CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(img_equalization)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

FIND_PACKAGE(OpenCV REQUIRED )

include_directories ("${OpenCV_INCLUDE_DIRS}")

add_executable(img_equalization img_equalization.cpp common_code.cpp clahe.cpp)

target_link_libraries(img_equalization ${OpenCV_LIBS})
add_executable(test_common_code test_common_code.cpp common_code.cpp)

target_link_libraries(test_common_code ${OpenCV_LIBS})

set_target_properties(test_common_code PROPERTIES OUTPUT_NAME "test_common_code")