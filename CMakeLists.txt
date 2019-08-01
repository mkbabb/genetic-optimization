cmake_minimum_required(VERSION 3.2)
project(erate VERSION 0.1.0)


set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/8.0.0_1/bin/clang++")

set(COMPILE_FLAGS "-fcoroutines-ts -stdlib=libc++ -std=c++2a")
# set(LINK_FLAGS "${LINK_FLAGS} -stdlib=libc++")

# include_directories(/usr/local/Cellar/llvm/8.0.0_1/include/c++/v1)




add_definitions(${COMPILE_FLAGS})
# add_definitions(${LINK_FLAGS})

add_executable(${PROJECT_NAME} include/itertools/tests.cpp)
# target_link_libraries(${PROJECT_NAME} PRIVATE sqlite3)



