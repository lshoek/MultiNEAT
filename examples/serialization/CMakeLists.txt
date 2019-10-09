add_executable(serialize_genome
        main.cpp)

target_link_libraries(serialize_genome
        MultiNEAT
        Boost::unit_test_framework)

add_test(serialize_genome serialize_genome)
