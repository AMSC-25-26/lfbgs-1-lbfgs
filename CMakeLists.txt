cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

include_directories(lib)

add_executable(main.o src/main.cpp)

target_link_libraries(main.o Eigen3::Eigen)

file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp")

add_custom_target(format
    COMMAND clang_format -i ${SOURCE_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formattazione del codice con clang-format..."
)