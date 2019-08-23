cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)

set (CMAKE_CXX_STANDARD 11)

set (MultiNEAT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

find_package(Threads REQUIRED)

find_package(Boost
        COMPONENTS
        date_time
        system
        filesystem
        serialization
        REQUIRED
        )

set(SOURCE_FILES
        src/Genome.cpp
        src/Innovation.cpp
        src/NeuralNetwork.cpp
        src/Parameters.cpp
        src/Population.cpp
        src/Random.cpp
        src/Species.cpp
        src/Substrate.cpp
        src/Utils.cpp
        src/Traits.cpp)


add_library(MultiNEAT STATIC ${SOURCE_FILES})

set(Boost_USE_STATIC_LIBS       OFF) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

target_include_directories(MultiNEAT
        PRIVATE
        ${Boost_INCLUDE_DIRS}
        PUBLIC
        $<INSTALL_INTERFACE:${MultiNEAT_SRC_DIR}/include>
        )

target_link_libraries(MultiNEAT
        PUBLIC
        Boost::date_time
        Boost::system
        Boost::filesystem
        Boost::serialization
        Threads::Threads
        )

target_compile_definitions(MultiNEAT
        PUBLIC
        USE_BOOST_RANDOM
        )

add_executable(MultiNEAT_Test
        src/Main.cpp
        )

target_link_libraries(MultiNEAT_Test
        MultiNEAT
        $<$<BOOL:${WIN32}>:ws2_32>
        )

if(GENERATE_PYTHON_BINDINGS)
    find_package(Python3
        COMPONENTS
            Development
        REQUIRED
    )

    message("found python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

    set(3 ${Python3_VERSION_MAJOR})
    set(3x "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
    set(PY3x "-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}") # ubuntu version

    Python3_add_library(multineat_python
            MODULE
            ${SOURCE_FILES}
            src/PythonBindings.cpp
            )

    target_compile_definitions(multineat_python
            PUBLIC
            USE_BOOST_RANDOM
            USE_BOOST_PYTHON
            )


    find_package(Boost COMPONENTS
            python${3}
            python${3x}
            python${PY3x}
            numpy${3}
            numpy${3x}
            numpy${PY3x}
            QUIET
            )

    target_include_directories(multineat_python
            PRIVATE
            ${Boost_INCLUDE_DIRS}
            PUBLIC
            $<INSTALL_INTERFACE:${MultiNEAT_SRC_DIR}/include>
            )

    target_link_libraries(multineat_python
            PUBLIC
            PUBLIC
            Boost::date_time
            Boost::system
            Boost::filesystem
            Boost::serialization
            # Python3::Python # already included by default
            $<$<BOOL:${Boost_PYTHON${3}_FOUND}>:${Boost_PYTHON${3}_LIBRARY}>
            $<$<BOOL:${Boost_PYTHON${3x}_FOUND}>:${Boost_PYTHON${3x}_LIBRARY}>
            $<$<BOOL:${Boost_PYTHON${PY3x}_FOUND}>:${Boost_PYTHON${PY3x}_LIBRARY}>
            $<$<BOOL:${Boost_NUMPY${3}_FOUND}>:${Boost_NUMPY${3}_LIBRARY}>
            $<$<BOOL:${Boost_NUMPY${3x}_FOUND}>:${Boost_NUMPY${3x}_LIBRARY}>
            $<$<BOOL:${Boost_NUMPY${PY3x}_FOUND}>:${Boost_NUMPY${PY3x}_LIBRARY}>
            )

endif()