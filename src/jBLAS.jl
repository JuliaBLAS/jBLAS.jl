# __precompile__()
module jBLAS

using StaticArrays, LinearAlgebra, Random, SIMDPirates, Base.Cartesian

export mrandn,
        jmul!,
        prefetch

# include("simd.jl") # Maybe use that as a fallback for custom number types? Eg, so it works with min-plus algebra?
include("cpu_info.jl")
include("miscellaneous.jl")
include("memory_management.jl")
include("kernel_structure.jl")
include("kernels.jl")
include("gemm.jl")
include("randmat.jl") # Currently commits type piracy. Maybe I should try to push those changes.

@generated VecType(::Type{T}) where T = :(Vec{$(REGISTER_SIZE÷sizeof(T)),$T})

end # module
