cmake_minimum_required (VERSION 2.6)

include(CMakeToolsHelpers OPTIONAL)

set (CMAKE_CXX_STANDARD 11)

project (optical_flow_cpu_version)

find_package(OpenCV)
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_LIBS})
endif(OpenCV_FOUND)


find_package(Boost COMPONENTS filesystem)
find_package(X11 REQUIRED)
find_package(ALSA REQUIRED)

file(GLOB_RECURSE optical_flow_cpu_version_SOURCES "src/*.cpp")
file(GLOB_RECURSE optical_flow_cpu_version_HEADERS "src/*.hpp")

set (optical_flow_cpu_version_INCLUDE_DIRS "")
foreach (_headerFile ${optical_flow_cpu_version_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND optical_flow_cpu_version_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES optical_flow_cpu_version_INCLUDE_DIRS)

add_executable(optical_flow_cpu_version ${optical_flow_cpu_version_SOURCES})
target_include_directories(optical_flow_cpu_version PRIVATE ${optical_flow_cpu_version_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${X11_INCLUDE_DIR})
target_link_libraries(optical_flow_cpu_version ${OpenCV_LIBS} ${Boost_LIBRARIES} nana ${X11_LIBRARIES} rt ${X11_Xft_LIB} ${ALSA_LIBRARIES} fontconfig stdc++fs)