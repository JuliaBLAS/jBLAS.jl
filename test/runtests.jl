# Need to turn this to actual tests.





using OhMyREPL, jBLAS, BenchmarkTools, LinearAlgebra

BLAS.set_num_threads(1)
A128_128 = mrandn(128,128);
B128_126 = mrandn(128,126);
C128_126 = mrandn(126,128);
jmul!(C128_128, A128_128, B128_126)
mul!(C128_128, A128_128, B128_126)
@benchmark jmul!($C128_128, $A128_128, $B128_126)
@benchmark mul!($C128_128, $A128_128, $B128_126)

A800_900 = mrandn(16*50, 900);
B900_840 = mrandn(900,14*60);
C800_840 = mrandn(16*50,14*60);
jmul!(C800_840, A800_900, B900_840)
mul!(C800_840, A800_900, B900_840)
@benchmark jmul!($C800_840, $A800_900, $B900_840)
@benchmark mul!($C800_840, $A800_900, $B900_840)
