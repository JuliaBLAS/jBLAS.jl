# __precompile__()
module jBLAS

using CpuId, StaticArrays, LinearAlgebra, Random, SIMD

export mrandn,
        jmul!

const register_size = CpuId.simdbytes()

# include("simd.jl") # Maybe use that as a fallback for custom number types? Eg, so it works with min-plus algebra?
include("cpu_info.jl")
include("memory_management.jl")
include("kernel_structure.jl")
include("gemm.jl")
include("randmat.jl") # Currently commits type piracy. Maybe I should try to push those changes.


end # module
