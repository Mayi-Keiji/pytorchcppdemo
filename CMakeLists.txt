cmake_minimum_required(VERSION 3.5)

project(cppdemo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BASE_DEPS_PATH D:\\AI\\Learn\\engineer)
message(${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CMAKE_PREFIX_PATH ${BASE_DEPS_PATH}\\libtorch_win_debug_1_13_1_cpu ${BASE_DEPS_PATH}\\opencv-4.7.0-windows\\opencv\\build)
   set(Torch_DIR ${BASE_DEPS_PATH}\\libtorch_win_debug_1_13_1_cpu\\share\\cmake\\Torch)
else()
    set(CMAKE_PREFIX_PATH ${BASE_DEPS_PATH}\\libtorch_win1_13_1_cpu ${BASE_DEPS_PATH}\\opencv-4.7.0-windows\\opencv\\build)
    set(Torch_DIR ${BASE_DEPS_PATH}\\libtorch_win1_13_1_cpu\\share\\cmake\\Torch)
endif()

set(Opencv_DIR ${BASE_DEPS_PATH}\\opencv-4.7.0-windows\\opencv\\build)

find_package(Torch REQUIRED)
find_package(Opencv REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE HEADERS ./*.h)
aux_source_directory(./ SOURCES)
add_executable(cppdemo ${SOURCES} ${HEADERS})

target_link_options(${PROJECT_NAME} PRIVATE /INCLUDE:?ignore_this_library_placeholder@@YAHXZ)
target_link_libraries(cppdemo ${TORCH_LIBRARIES}  ${OpenCV_LIBS})
set_property(TARGET cppdemo PROPERTY CXX_STANDARD 14)
