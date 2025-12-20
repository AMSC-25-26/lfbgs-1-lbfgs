cmake_minimum_required(VERSION 3.14)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG        v1.0.3
)


# tests and python bindings require external modules
# such as Catch2, so we disable them
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(autodiff)

add_executable(main_app src/main.cpp)
add_executable(test_runner tests/main.cpp)

target_link_libraries(main_app PRIVATE autodiff)
target_link_libraries(test_runner PRIVATE autodiff)

target_include_directories(main_app PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/lib)

