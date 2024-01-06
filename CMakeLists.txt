cmake_minimum_required(VERSION 3.0.0)
project(photo-merger VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_TZ_LIB ON)
set(USE_SYSTEM_TZ_DB ON)
add_subdirectory("third_party/date")

add_compile_options(-fopenmp)

include(CTest)
enable_testing()

find_package(OpenCV)
find_package(Eigen3)
find_package(fmt)
find_package(rawdevpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${rawdevpp_INCLUDE_DIRS} /usr/include/mkl)
include_directories("third_party/date/include")

add_executable(photo-merger main.cpp)

target_link_libraries(photo-merger ${OpenCV_LIBRARIES} date date-tz mkl_rt rawdevpp fmt::fmt zstd)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
