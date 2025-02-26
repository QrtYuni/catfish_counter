cmake_minimum_required(VERSION 3.8)

project (seat C CXX)

find_package(OpenCV REQUIRED)

set (CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(include)

add_executable(run src/main.cpp)
target_link_libraries(run
    ${OpenCV_LIBS}

)   