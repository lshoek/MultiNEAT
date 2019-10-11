# C++ test
add_executable(serialize_genome
        serialize_genome.cpp)

target_link_libraries(serialize_genome
        MultiNEAT
        Boost::unit_test_framework)

add_test(serialize_genome serialize_genome)

if(GENERATE_PYTHON_BINDINGS)
    # python to C++ test
    add_executable(analize_genome_file
            analize_genome_file.cpp)

    target_link_libraries(analize_genome_file
            MultiNEAT)

    ## C++ to python test
    #add_executable(generate_genome_file
    #        generate_genome_file.cpp)
    #
    #target_link_libraries(generate_genome_file
    #        MultiNEAT)

    add_test(python_to_c++_to_python test.py)
endif()
