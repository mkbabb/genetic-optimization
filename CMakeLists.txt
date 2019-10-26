cmake_minimum_required(VERSION 3.2)
project(erate_analysis VERSION 0.1.0)


set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/8.0.1/bin/clang++")

set(COMPILE_FLAGS "-fcoroutines-ts -stdlib=libc++ -std=c++2a")

# -O1 -fno-signed-zeros -freciprocal-math -ffp-contract=fast -fno-trapping-math -ffast-math -ffinite-math-only


add_definitions(${COMPILE_FLAGS})
# add_definitions(${LINK_FLAGS})

add_executable(${PROJECT_NAME} src/tests.cpp)
# target_link_libraries(${PROJECT_NAME} PRIVATE sqlite3)



