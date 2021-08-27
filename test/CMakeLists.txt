

add_executable(test_rpc test_rpc.cc)
target_include_directories(test_rpc PUBLIC "../src")
target_link_libraries(test_rpc moorpc)


add_executable(test_multinode_allreduce test_multinode_allreduce.cc)
target_include_directories(test_multinode_allreduce PUBLIC "../src")
target_link_libraries(test_multinode_allreduce moorpc)


