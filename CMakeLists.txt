# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# project(pytorchv5)

# set(CMAKE_PREFIX_PATH "/home/wangmao/libtorch/")

# find_package(Torch REQUIRED)

# add_executable(main main.cpp)
# target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")
# set_property(TARGET main PROPERTY CXX_STANDARD 14)




cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(pytorchv5)
set(CMAKE_PREFIX_PATH "/home/wangmao/libtorch/")

find_package(Torch REQUIRED)
find_package(OpenCV  REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV  not found.")
endif()
add_executable(main main.cpp)
# target_link_libraries(main  PRIVATE "" )
target_link_libraries(main   "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" )
set_property(TARGET main PROPERTY CXX_STANDARD 14)