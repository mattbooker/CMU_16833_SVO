cmake_minimum_required(VERSION 2.8)
project( SVO )

# add_compile_options(-Wall -Wextra -Werror)

# Add packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
FetchContent_MakeAvailable(googletest)

# Include dirs
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Set sourcefiles
list(APPEND SOURCEFILES
  src/DepthFilter.cpp)

# Create svo library
add_library(svo SHARED ${SOURCEFILES})
target_link_libraries(svo ${OpenCV_LIBS})

# Testing
enable_testing()

# Set sourcefiles
list(APPEND TESTFILES
  test/TestDepthFilter.cpp)

add_executable(
  svo_test
  ${TESTFILES}
)
target_link_libraries(
  svo_test
  svo
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(svo_test)
