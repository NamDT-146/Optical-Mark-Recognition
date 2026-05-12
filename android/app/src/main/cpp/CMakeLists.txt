cmake_minimum_required(VERSION 3.22.1)
project("omrcore")

set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CORE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core")

if(NOT DEFINED OpenCV_DIR)
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../libs/OpenCV-android-sdk/sdk/native/jni")
endif()

message(STATUS "Using OpenCV_DIR=${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

add_library(omrcore SHARED
        jni_bridge.cpp
        ${CORE_ROOT_DIR}/src/scanner.cpp
        ${CORE_ROOT_DIR}/src/grader.cpp
        ${CORE_ROOT_DIR}/src/utils.cpp
)

target_include_directories(omrcore PRIVATE
        ${CORE_ROOT_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(omrcore PRIVATE cxx_std_17)

target_link_libraries(omrcore
    ${OpenCV_LIBS}
    android
    log
    jnigraphics
)