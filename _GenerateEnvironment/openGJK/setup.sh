cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build build 
cmake -E chdir build/examples/c ./example_lib_opengjk_ce