cmake_minimum_required(VERSION 3.3)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_subdirectory(xor)
add_subdirectory(serialization)

