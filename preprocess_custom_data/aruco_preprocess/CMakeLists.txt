cmake_minimum_required(VERSION 3.11.4)
project(PieceSeg)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_library(CNPY_LIB cnpy)
include_directories(/usr/local/include)


add_executable(calibration calibration.cpp)
target_link_libraries(calibration ${OpenCV_LIBS} ${CNPY_LIB})