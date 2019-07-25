cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(neural_network)

find_package(Torch REQUIRED)

add_executable(neural_network neural_network.cpp)
target_link_libraries(neural_network "${TORCH_LIBRARIES}")
set_property(TARGET neural_network PROPERTY CXX_STANDARD 17)
