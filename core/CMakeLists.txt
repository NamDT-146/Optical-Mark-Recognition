cmake_minimum_required(VERSION 3.10)
project(OMR_System_Core)

# Sử dụng C++ 17 để tận dụng các tính năng mới
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tìm OpenCV (Yêu cầu bạn đã cài OpenCV trên máy)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Gom tất cả file source lại
file(GLOB SOURCES "src/*.cpp")

# Tạo file thực thi (Cho console test CP1 & CP2)
add_executable(omr_engine ${SOURCES})

# Liên kết thư viện OpenCV
target_link_libraries(omr_engine ${OpenCV_LIBS})