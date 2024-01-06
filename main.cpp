#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <numeric>
#include <optional>
#include <vector>

#include <fmt/format.h>

#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

// #include <date/date.h>
#include <date/tz_private.h>
#include <opencv2/opencv.hpp>
#include <rawdevpp/dng.hpp>
#include <zstd.h>
namespace fs = std::filesystem;

#define USE_PRE_MEDIAN
#define OUTPUT_MERGED_FRAME
#define OUTPUT_FRAME
#define OUTPUT_DEBUG_INFO
// #define LOAD_DATA_ONLY
// #define POST_PROCESS_ONLY

static const std::string dataFolder = "./data/yog/data";
static const std::string TimeBegin = "2023-12-21T00:00:00-CST";
static const std::string TimeEnd = "2023-12-22T00:00:00-CST";
static const std::string debugFileFolder = ".";
static const std::string mergedFileFolder = "merged";
static const std::string outputFileFolder = "frames";
// only used without load-data
static const size_t ROWS = 3040;
static const size_t COLS = 4056;

#ifndef LOAD_DATA_ONLY
#define POST_PROCESS
#endif
#ifndef POST_PROCESS_ONLY
#define LOAD_DATA
#endif

cv::Mat3d camToProPhotoRGB(const cv::Mat3d &image, size_t wide, const Eigen::Matrix3d &matrix)
{
    cv::Mat ret = image.clone();
    ret.convertTo(ret, CV_64F);

    auto pixels = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor>>(reinterpret_cast<double *>(ret.data), 3, ret.rows * ret.cols);
    pixels = (matrix * pixels).eval();

    ret = cv::max(0, cv::min(std::pow(2, wide) - 1, ret));

    return ret;
}

cv::Mat3d RGB2BGR(const cv::Mat3d &image)
{
    cv::Mat3d ret;
    cv::Mat1d ch[3];
    cv::split(image, ch);
    std::swap(ch[0], ch[2]);
    cv::merge(ch, 3, ret);
    return ret;
}

cv::Mat1d easyGray(const cv::Mat3d &image)
{
    cv::Mat1d channel[3];
    cv::split(image, channel);
    return channel[0] * 0.114 + channel[1] * 0.587 + channel[2] * 0.299;
}

cv::Mat3i mergeImage(const std::vector<cv::Mat> &origin, int offset, int validBits, int dilateRange, int range)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dilateRange, dilateRange});

    cv::Mat3d total = cv::Mat3d::zeros(origin[0].size());
    size_t K = std::pow(2, offset);
    size_t Khalf = std::pow(2, offset / 2);
    size_t MAX = std::pow(2, validBits);
    int bits = validBits - offset;
    for (size_t i = 0; i < origin.size(); i++)
    {
        total *= K;
        bits += offset;
        cv::Mat3d temp;
        origin[i].convertTo(temp, CV_64F);

        cv::Mat mask = cv::Mat3b(total >= MAX);
        cv::dilate(mask, mask, kernel);

        mask.convertTo(mask, CV_64F);
        mask /= 255;
        cv::GaussianBlur(mask, mask, {range, range}, 0);

        total = total.mul(mask) + temp.mul(cv::Scalar{1.0, 1.0, 1.0} - mask);
        cv::Mat3i mid;
        total.convertTo(mid, CV_32S);
        mid.convertTo(total, CV_64F);
    }

    cv::Mat3i ret;
    total.convertTo(ret, CV_32S);
    return ret;
}

struct CompressLightInfo
{
    size_t minTrim;
    size_t maxTrim;
    size_t peak;
    std::vector<size_t> reHist;
};

CompressLightInfo calcCompressLightSingle(const cv::Mat1f &origin, size_t originWide, size_t aimWide, size_t minBins, double step = 1)
{
    size_t Row = originWide / step;
    cv::Mat1f hist = cv::Mat1f::zeros(1, Row);
    for (size_t i = 0; i < Row; i++)
    {
        double min = std::pow(2, i * step);
        double max = std::pow(2, (i + 1) * step);
        cv::Mat1b mask = (origin >= min) & (origin < max);
        mask /= 255;
        hist(0, i) = cv::sum(mask)[0];
    }

    int minPos = 0, maxPos = 0;
    double min = DBL_MAX, max = 0;
    for (size_t i = 0; i < Row; i++)
    {
        if (hist(0, i) < min)
        {
            min = hist(0, i);
            minPos = i;
        }
        if (hist(0, i) > max)
        {
            max = hist(0, i);
            maxPos = i;
        }
    }

    {
        size_t minTrim = 0;
        size_t maxTrim = originWide;
        for (size_t i = 0; i < Row; i++)
        {
            minTrim = i;
            if (hist(0, i) > 0)
            {
                break;
            }
        }
        for (size_t i = Row - 1; i >= 0; i--)
        {
            maxTrim = i;
            if (hist(0, i) > 0)
            {
                break;
            }
        }
        size_t dynamic = std::pow(2, aimWide) - (maxTrim - minTrim + 1) * minBins - 1;
        std::vector<size_t> reHist;
        size_t total = 0;
        for (size_t i = 0; i < Row; i++)
        {
            reHist.push_back(0);
            if (i >= minTrim && i <= maxTrim)
            {
                total += hist(0, i);
            }
        }
        size_t reTotal = 0;
        for (size_t i = 0; i < Row; i++)
        {
            if (i >= minTrim && i <= maxTrim)
            {
                size_t N = std::round(dynamic * double(hist(0, i)) / total);
                if (reTotal + N > dynamic)
                {
                    N = dynamic - reTotal;
                }
                reTotal += N;
                reHist[i] = minBins + N;
            }
        }
        if (reTotal < dynamic)
        {
            reHist[maxPos] += dynamic - reTotal;
        }

        CompressLightInfo ret;
        ret.minTrim = minTrim;
        ret.maxTrim = maxTrim;
        ret.peak = maxPos;
        ret.reHist = reHist;
        return ret;
    }
}

cv::Mat applyCompressLightSingle(const cv::Mat1f &origin, const CompressLightInfo &info, size_t aimWide, double step = 1)
{
    cv::Mat1f ret = cv::Mat1f::zeros(origin.size());

    double used = 1;
    for (size_t i = info.minTrim; i <= info.maxTrim; i++)
    {
        double base = std::pow(2, i * step);
        double top = std::pow(2, (i + 1) * step);
        double range = top - base;
        double k = double(info.reHist[i]) / range;
        cv::Mat1f((origin - cv::Scalar{base}) * k + cv::Scalar{used}).copyTo(ret, (origin >= base) & (origin < top));
        used += info.reHist[i];
    }
    ret.setTo(std::pow(2, aimWide) - 1, origin >= std::pow(2, (info.maxTrim + 1) * step));

    ret.setTo(0, ret < 0);
    size_t maxValue = std::pow(2, aimWide) - 1;
    ret.setTo(maxValue, ret > maxValue);
    return ret;
}

class PID
{
    double Kp;
    double Ki;
    double Kd;
    size_t length;
    std::list<double> cache;
    double value;

public:
    PID(double Kp, double Ki, double Kd, size_t length, std::optional<double> init = {}) : Kp(Kp), Ki(Ki), Kd(Kd), length(length)
    {
        if (init.has_value())
        {
            value = init.value();
            cache.push_front(0);
        }
    }

    double update(double aim)
    {
        if (cache.size() == 0)
        {
            value = aim;
            this->cache.push_front(0);
            return value;
        }
        double err = aim - value;
        cache.push_front(err);
        if (cache.size() > length)
            cache.pop_back();
        double delta = Kp * err + Ki * std::accumulate(cache.begin(), cache.end(), 0) + Kd * (*cache.begin() - *(cache.begin()++));
        value += delta;
        return value;
    }
};

int main(int, char **)
{
    std::vector<fs::path> captures;

#ifdef LOAD_DATA
    std::vector<fs::path> cameras;
    fs::path camFolder = dataFolder;
    fs::directory_iterator it_end;
    for (fs::directory_iterator it(camFolder); it != it_end; it++)
    {
        cameras.push_back(it->path());
    }
    std::vector<date::local_seconds> timeList;
    const date::tzdb &tzdb = date::get_tzdb();
    auto tz = tzdb.current_zone();

    std::stringstream ss;
    ss = std::stringstream(TimeBegin);
    date::local_seconds beginTime;
    ss >> date::parse("%FT%T-%Z", beginTime);
    ss = std::stringstream(TimeEnd);
    date::local_seconds endTime;
    ss >> date::parse("%FT%T-%Z", endTime);

    for (fs::directory_iterator it(cameras[0]); it != it_end; it++)
    {
        captures.push_back(it->path());
        std::istringstream ss(it->path().filename().string());
        std::chrono::sys_seconds time;
        ss >> date::parse("%FT%T-%Z", time);
        auto local = tz->to_local(time);

        if (local < beginTime || local >= endTime)
            continue;

        timeList.push_back(local);
    }

#ifdef USE_PRE_MEDIAN
    rawdevpp::Color::XY rawShots = rawdevpp::Color::XY::Zero(2, timeList.size());
    for (size_t index = 0; index < timeList.size(); index++)
    {
        std::vector<std::string> files;
        for (fs::directory_iterator it(captures[index]); it != it_end; it++)
        {
            files.push_back(it->path().string());
        }
        std::fstream file;
        file.open(files[0], std::ios::in | std::ios::binary);
        auto image = rawdevpp::Decoder::DNG::parse(file, 3);
        auto rawShot = rawdevpp::Color::XYZ2XY(image.images[0].asShotNeutral.value().array());
        file.close();
        rawShots.col(index) = rawShot.col(0);
    }
    const int filterR = 10;
    std::vector<rawdevpp::Color::XY> rawShotsFiltered;
    for (int i = 0; i < timeList.size(); i++)
    {
        int left = std::max(0, i - filterR);
        int right = std::min((int)timeList.size(), i + filterR + 1);
        auto slice = rawShots.middleCols(left, right - left);
        std::vector<double> xList, yList;
        size_t N = slice.cols();
        for (size_t n = 0; n < N; n++)
        {
            xList.push_back(slice(0, n));
            yList.push_back(slice(1, n));
        }
        std::sort(xList.begin(), xList.end());
        std::sort(yList.begin(), yList.end());
        rawdevpp::Color::XY item = rawdevpp::Color::XY::Zero(2, 1);
        item(0, 0) = xList[N / 2];
        item(1, 0) = yList[N / 2];
        rawShotsFiltered.push_back(item);
    }
#endif

    const double Kp = 0.3;
    const double Ki = Kp * 0.1;
    const double Kd = Kp * 0;
    const size_t length = 30;
    PID balanceX(Kp, Ki, Kd, length);
    PID balanceY(Kp, Ki, Kd, length);
#else
    fs::directory_iterator it_end;
    for (fs::directory_iterator it(mergedFileFolder); it != it_end; it++)
    {
        if (it->path().extension() == ".zst")
            captures.push_back(it->path());
    }
#endif

    const double step = 0.5;

#ifdef LOAD_DATA
#ifdef OUTPUT_DEBUG_INFO
    std::ofstream shotInfoDbg(fmt::format("{}/shotInfo.json", debugFileFolder));
    shotInfoDbg << "[" << std::endl;
#endif
#ifdef OUTPUT_MERGED_FRAME
    std::ofstream mergedInfo;
    mergedInfo.open(fmt::format("{}/merged-info.bin", debugFileFolder), std::ios::out | std::ios::binary);
#endif
#else
    std::ifstream mergedInfo;
    mergedInfo.open(fmt::format("{}/merged-info.bin", debugFileFolder), std::ios::in | std::ios::binary);
#endif

#ifdef POST_PROCESS
    cv::Mat1d compressDbg = cv::Mat1d::zeros(32 / step, captures.size());
#endif

    for (size_t index = 0; index < captures.size(); index++)
    {
        std::optional<Eigen::Matrix3d> balanceMatrix;
        cv::Mat merged;
#ifdef LOAD_DATA
        using namespace std::chrono_literals;
        std::cout << date::format("%FT%T", timeList[index]) << "[" << date::format("%FT%T", beginTime) << "," << date::format("%FT%T", endTime) << "]" << std::endl;

        std::vector<std::string> files;
        for (fs::directory_iterator it(captures[index]); it != it_end; it++)
        {
            files.push_back(it->path().string());
        }

        std::vector<cv::Mat> mats;
        rawdevpp::Context ctx;
        for (const auto &f : files)
        {
            std::fstream file;
            file.open(f, std::ios::in | std::ios::binary);
            auto image = rawdevpp::Decoder::DNG::parse(file, 3);

            auto rootImage = image.images[0].base.getDirectoryBySubfileType(file, image.tiff.byteswap, 0).value();
            auto data = rootImage.readImage<uint16_t>(rootImage.getImageInfo(file, image.tiff.byteswap), file, image.tiff.byteswap);
            cv::Mat bayer = cv::Mat1w(std::get<2>(data)).clone();
            bayer = bayer.reshape(1, std::get<1>(data));
            cv::Mat raw;
            cv::cvtColor(bayer, raw, cv::COLOR_BayerRG2RGB);

            raw.convertTo(raw, CV_64F);
            double black;
            {
                auto blackLevel = image.images[0].blackLevel.value();
                if (blackLevel.type == (uint16_t)rawdevpp::Decoder::TIFF::DEDataType::SHORT)
                    black = blackLevel.value<std::vector<uint16_t>>(file, image.tiff.byteswap)[0];
                else if (blackLevel.type == (uint16_t)rawdevpp::Decoder::TIFF::DEDataType::LONG)
                    black = blackLevel.value<std::vector<uint32_t>>(file, image.tiff.byteswap)[0];
                else
                {
                    auto value = blackLevel.value<std::vector<std::pair<uint32_t, uint32_t>>>(file, image.tiff.byteswap)[0];
                    black = value.first / (double)value.second;
                }
            }
            raw = cv::max(raw - cv::Scalar{black, black, black}, 0);
            raw.convertTo(raw, CV_16U);
            file.close();

            mats.push_back(raw);

            if (!balanceMatrix.has_value())
            {
#ifdef USE_PRE_MEDIAN
                auto rawShot = rawShotsFiltered.at(index);
#else
                auto rawShot = rawdevpp::Color::XYZ2XY(image.images[0].asShotNeutral.value().array());
#endif
                rawdevpp::Color::XY processedShot = rawdevpp::Color::XY::Zero(2, 1);
                processedShot(0, 0) = balanceX.update(rawShot(0, 0));
                processedShot(1, 0) = balanceY.update(rawShot(1, 0));
                balanceMatrix = image.images[0].matrixCamera2ProPhotoRGB(ctx, rawdevpp::Color::XY2XYZ(processedShot));

#ifdef OUTPUT_DEBUG_INFO
                {
                    auto oriShot = image.images[0].asShotNeutral.value().array();
                    shotInfoDbg << fmt::format(R"({{"raw":[{},{}],"median":[{},{}],"pid":[{},{}]}},)",
                                               oriShot(0, 0), oriShot(1, 0),
                                               rawShot(0, 0), rawShot(1, 0),
                                               processedShot(0, 0), processedShot(1, 0))
                                << std::endl;
                }
#endif
            }
        }

        merged = mergeImage(mats, 4, 12, 17, 21);

#ifdef OUTPUT_MERGED_FRAME
        {
            std::vector<char> buffer;
            buffer.resize(merged.elemSize() * merged.rows * merged.cols);
            size_t compressedSize = ZSTD_compress(buffer.data(), buffer.size(), merged.data, buffer.size(), 10);
            std::ofstream outputFile;
            outputFile.open(fmt::format("{}/{}.zst", mergedFileFolder, index), std::ios::out | std::ios::binary);
            outputFile.write(buffer.data(), compressedSize);
            outputFile.close();
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    double value = balanceMatrix.value()(r, c);
                    mergedInfo.write((char *)&value, sizeof(double));
                }
        }
#endif
#else
        {
            std::cout << captures[index].generic_string() << std::endl;
            merged = cv::Mat3i::zeros(ROWS, COLS);
            Eigen::Matrix3d matrix;
            std::ifstream file(captures[index].generic_string(), std::ios::in | std::ios::binary);
            file.seekg(0, std::ios::end);
            size_t N = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> zstdData;
            zstdData.resize(N);
            file.read(zstdData.data(), N);
            file.close();
            size_t parsed = ZSTD_decompress(merged.data, merged.elemSize() * merged.rows * merged.cols, zstdData.data(), N);

            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                {
                    double value;
                    mergedInfo.read((char *)&value, sizeof(double));
                    matrix(r, c) = value;
                }
            balanceMatrix = matrix;
        }
#endif

#ifdef POST_PROCESS
        merged = camToProPhotoRGB(merged, 28, balanceMatrix.value());

        merged /= std::pow(2, 28);
        merged.convertTo(merged, CV_32F);
        cv::cvtColor(merged, merged, cv::COLOR_RGB2HLS);

        cv::Mat split[3];
        cv::split(merged, split);
        split[1] *= std::pow(2, 28);
        auto info = calcCompressLightSingle(split[1], 28, 8, 1, step);
        // std::cout << fmt::format("[{},{},{}]", info.minTrim, info.peak, info.maxTrim);
        split[1] = applyCompressLightSingle(split[1], info, 8, step);
        split[1] = split[1] / 256;
        cv::merge(split, 3, merged);

        cv::cvtColor(merged, merged, cv::COLOR_HLS2RGB);
        merged = RGB2BGR(merged);
        merged *= 255;

#ifdef OUTPUT_FRAME
        cv::Mat output = cv::min(255, cv::max(0, merged));
        output.convertTo(output, CV_8U);
        cv::imwrite(fmt::format("{}/{}.png", outputFileFolder, index), output);
#endif

#ifdef OUTPUT_DEBUG_INFO
        {
            for (size_t i = info.minTrim; i <= info.maxTrim; i++)
            {
                compressDbg(i, index) = info.reHist[i];
            }
        }
#endif
#endif
    }

#ifdef LOAD_DATA
#ifdef OUTPUT_MERGED_FRAME
    mergedInfo.close();
#endif
#ifdef OUTPUT_DEBUG_INFO
    shotInfoDbg << "null]";
    shotInfoDbg.close();
#endif
#endif

#ifdef POST_PROCESS
#ifdef OUTPUT_DEBUG_INFO
    {
        cv::Mat1b output;
        compressDbg.convertTo(output, CV_8U);
        cv::imwrite(fmt::format("{}/compressDbg.png", debugFileFolder), output);
    }
#endif
#endif

    return 0;
}
