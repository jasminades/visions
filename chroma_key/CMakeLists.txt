CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(chroma_key)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

FIND_PACKAGE(OpenCV REQUIRED)

include_directories("${OpenCV_INCLUDE_DIRS}")

# Define the executables
add_executable(chroma_key main.cpp common_code.cpp)
add_executable(chroma_key_test_common_code test_common_code.cpp common_code.cpp)

# Link the executables with OpenCV libraries
target_link_libraries(chroma_key ${OpenCV_LIBS})
target_link_libraries(chroma_key_test_common_code ${OpenCV_LIBS})

# Set the output name for the test executable
set_target_properties(chroma_key_test_common_code PROPERTIES OUTPUT_NAME "test_common_code")
