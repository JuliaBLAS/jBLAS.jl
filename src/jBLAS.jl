# __precompile__()
module jBLAS

using CpuId, StaticArrays, LinearAlgebra, Random, SIMD

export mrandn,
        jmul!

const register_size = CpuId.simdbytes()

# include("simd.jl")
include("gemm.jl")
include("randmat.jl") # Currently commits type piracy. Maybe I should try to push those changes.


end # module
