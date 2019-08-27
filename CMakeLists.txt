cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)

set (CMAKE_CXX_STANDARD 11)

set (MultiNEAT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()


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
        src/Traits.cpp
        )


set(Boost_USE_STATIC_LIBS       OFF) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

if (NOT SKBUILD)

    add_library(MultiNEAT STATIC ${SOURCE_FILES})

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

endif(NOT SKBUILD)

if(GENERATE_PYTHON_BINDINGS)
    find_package(Python3
        COMPONENTS
            Development
        REQUIRED
    )
    if (SKBUILD)
        find_package(PythonExtensions REQUIRED)
    endif()

    message("found python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

    set(3 ${Python3_VERSION_MAJOR})
    set(3x "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
    set(PY3x "-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}") # ubuntu version

    find_package(Boost COMPONENTS
            python${3}
            python${3x}
            python${PY3x}
            numpy${3}
            numpy${3x}
            numpy${PY3x}
            QUIET
            )

    if (Boost_python${3}_FOUND OR Boost_PYTHON${3}_FOUND)
        message("Boost python${3} found")
    elseif(Boost_python${3x}_FOUND OR Boost_PYTHON${3x}_FOUND)
        message("Boost python${3x} found")
    elseif(Boost_python${PY3x}_FOUND OR Boost_PYTHON${PY3x}_FOUND)
        message("Boost python${PY3x} found")
    else()
        message(FATAL_ERROR "Boost python3 not found")
    endif()

    if (Boost_numpy${3}_FOUND OR Boost_NUMPY${3}_FOUND)
        message("Boost python-numpy${3} found")
    elseif(Boost_numpy${3x}_FOUND OR Boost_NUMPY${3x}_FOUND)
        message("Boost python-numpy${3x} found")
    elseif(Boost_numpy${PY3x}_FOUND OR Boost_NUMPY${PY3x}_FOUND)
        message("Boost python-numpy${PY3x} found")
    else()
        message(FATAL_ERROR "Boost python numpy 3 not found")
    endif()

    if (SKBUILD)
        add_library(_multineat
                MODULE
                ${SOURCE_FILES}
                src/PythonBindings.cpp)
        python_extension_module(_multineat)
    else()
        Python3_add_library(_multineat
                MODULE
                ${SOURCE_FILES}
                src/PythonBindings.cpp
                )
    endif()


    target_compile_definitions(_multineat
            PUBLIC
            USE_BOOST_RANDOM
            USE_BOOST_PYTHON
            )

    target_include_directories(_multineat
            PRIVATE
            ${Boost_INCLUDE_DIRS}
            )

    set(MULTINEAT_PYTHON_LIBRARIES
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

            #SKBUILD VERSION
            $<$<BOOL:${Boost_python${3}_FOUND}>:${Boost_PYTHON${3}_LIBRARY}>
            $<$<BOOL:${Boost_python${3x}_FOUND}>:${Boost_PYTHON${3x}_LIBRARY}>
            $<$<BOOL:${Boost_python${PY3x}_FOUND}>:${Boost_PYTHON${PY3x}_LIBRARY}>
            $<$<BOOL:${Boost_numpy${3}_FOUND}>:${Boost_NUMPY${3}_LIBRARY}>
            $<$<BOOL:${Boost_numpy${3x}_FOUND}>:${Boost_NUMPY${3x}_LIBRARY}>
            $<$<BOOL:${Boost_numpy${PY3x}_FOUND}>:${Boost_NUMPY${PY3x}_LIBRARY}>
            )

    if (SKBUILD)
        target_link_libraries(_multineat
                ${MULTINEAT_PYTHON_LIBRARIES})
        set(site_packages "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
        install(TARGETS _multineat LIBRARY DESTINATION "${site_packages}/multineat")
    else()
        target_link_libraries(_multineat PUBLIC
                ${MULTINEAT_PYTHON_LIBRARIES})
    endif()

endif()