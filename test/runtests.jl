# Need to turn this to actual tests.





using OhMyREPL, jBLAS, BenchmarkTools, LinearAlgebra


A16_32 = mrandn(16,32);
B32_14 = mrandn(32,14);
C16_14 = mrandn(16,14);
@benchmark jmul!($C16_14, $A16_32, $B32_14)


A32_32 = mrandn(32,32);
B32_28 = mrandn(32,28);
C32_28 = mrandn(32,28);
@benchmark jmul!($C32_28, $A32_32, $B32_28)


BLAS.set_num_threads(1)
A128_128 = mrandn(128,128);
B128_126 = mrandn(128,126);
C128_126 = mrandn(128,126);
jmul!(C128_126, A128_128, B128_126)
mul!(C128_126, A128_128, B128_126)
@benchmark jmul!($C128_126, $A128_128, $B128_126)
@benchmark mul!($C128_126, $A128_128, $B128_126)

A800_900 = mrandn(16*50, 900);
B900_840 = mrandn(900,14*60);
C800_840 = mrandn(16*50,14*60);
jmul!(C800_840, A800_900, B900_840)
mul!(C800_840, A800_900, B900_840)
@benchmark jmul!($C800_840, $A800_900, $B900_840)
@benchmark mul!($C800_840, $A800_900, $B900_840)


#
# using Random, StaticArrays
# jBLAS.pick_kernel_size(::Type{Core.VecElement{Float64}}) = jBLAS.pick_kernel_size(Float64)
# function Random.rand!(x::MArray{M,N,Core.VecElement{T}}) where {M,N,T}
#     @inbounds for i ∈ eachindex(x)
#         x[i] = Core.VecElement(randn())
#     end
#     x
# end
# function vmrandn(M,N)
#     out = MMatrix{M,N,Core.VecElement{Float64}}(undef)
#     rand!(out)
# end
#
#
# vA32_32 = vmrandn(32,32);
# vB32_28 = vmrandn(32,28);
# vC32_28 = vmrandn(32,28);
# @benchmark jmul!($vC32_28, $vA32_32, $vB32_28)
#
# BLAS.set_num_threads(1)
# vA128_128 = vmrandn(128,128);
# vB128_126 = vmrandn(128,126);
# vC128_126 = vmrandn(128,126);
# jmul!(vC128_126, vA128_128, vB128_126)
# mul!(vC128_126, vA128_128, vB128_126)
# @benchmark jmul!($vC128_126, $vA128_128, $vB128_126)
# @benchmark mul!($vC128_126, $vA128_128, $vB128_126)
#
# vA800_900 = vmrandn(16*50, 900);
# vB900_840 = vmrandn(900,14*60);
# vC800_840 = vmrandn(16*50,14*60);
# vjmul!(vC800_840, vA800_900, vB900_840)
# vmul!(vC800_840, vA800_900, vB900_840)
# @benchmark jmul!($vC800_840, $vA800_900, $vB900_840)
# @benchmark mul!($vC800_840, $vA800_900, $vB900_840)
