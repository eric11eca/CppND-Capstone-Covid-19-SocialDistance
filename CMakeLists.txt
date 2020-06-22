cmake_minimum_required( VERSION 3.7 )
set(CMAKE_CXX_STANDARD 14)
add_definitions( -std=c++14 )
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 -pthread")

project( SocialDistance )
find_package( OpenCV REQUIRED NO_MODULE PATHS /usr/local/lib/cmake/opencv4 NO_DEFAULT_PATH)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable( SocialDistance ${SOURCES})
target_link_libraries( SocialDistance ${OpenCV_LIBS} )
