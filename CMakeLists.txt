cmake_minimum_required(VERSION 3.11)
project(HTJ2K_OpenCL VERSION 1.0.0)

# for clangd to work properly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

include_directories(HTJ2K_OpenCL PRIVATE include)

file(GLOB SOURCES src/*.cpp)
# add_definitions(-DRESOURCE_DIR="${CMAKE_SOURCE_DIR}/resources/")
add_executable(main ${SOURCES})

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)


target_link_libraries(main PRIVATE OpenCL ${OpenCV_LIBS})
