cmake_minimum_required(VERSION 3.16)

# find_program(CCACHE_PROGRAM ccache)
# if(CCACHE_PROGRAM)
#     message(STATUS "Found CCACHE_PROGRAM")
#     set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
# endif()
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

project(depth2normal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(depth2normal depth2normal.cpp pcl_utils.hpp)
target_link_libraries(depth2normal ${PCL_LIBRARIES} ${OpenCV_LIBS} ${VTK_LIBRARIES})