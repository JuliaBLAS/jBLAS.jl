"""
This function should eventually be used to pick the kernel sizes.
However, right now I think the math is wrong.
Eg,
1) it doesn't realize that 16x14 works without spilling for AVX512, instead saying 10 is the most columns that fit with 16 rows
2) It picks sizes that are much too skiny, like 32x6.
Once this is improved, we can update pick_kernel_size to calculate the best size using the same logic as this function
(without storing all results in a matrix, just tracking the best)

Worth looking into, maybe we do want relatively non-square kenerls, and the problem
is one of memory movement? That is, our square blocks should try to balance
the number of rows and columns...
...what is the impact on memory movement of different shapes?
"""
function kernel_size_summary(avx512, ::Type{T} = Float64) where T
    register_size, register_count = avx512 ? (64, 32) : (32, 16)
    t_size = sizeof(T)
    num_per_register = register_size ÷ t_size
    max_total = num_per_register * register_count
    cache_line = CACHELINE_SIZE ÷ t_size
    num_cache_lines = cld(max_total, cache_line)
    summary = Matrix{Float64}(undef, 5, num_cache_lines)
    for num_row_cachelines ∈ 1:num_cache_lines
        num_rows = num_row_cachelines * cache_line
        a_loads = cld(num_rows, num_per_register)
        num_cols = (register_count - a_loads - 2) ÷ a_loads
        summary[:, num_row_cachelines] .= (num_rows, num_cols, num_rows * num_cols, num_cols + a_loads, num_rows * num_cols / (num_cols + a_loads))
        if num_cols == 0
            println("A * X = D; nrow(D), ncol(D), length(D), NumRegeristersLoaded")
            return summary[:, 1:num_row_cachelines]
        end
    end
    println("A * X = D; nrow(D), ncol(D), length(D), NumRegeristersLoaded")
    summary
end

"""
Returns number of elements per register,
and the number of rows and columns of the kernel.
Ie, 8, 16, 14 means that 8 elements fit per register,
and the optimal kernel is
16x14 = 16xN * Nx14
matrix multiplication.
"""
function pick_kernel_size(::Type{Float64} = Float64; D_count = 1, A_count = 1, X_count = 1) where T
    # register_size == 32 ? (4,8,6) : (8,32,6)  #assumes size is either 32 or 64
    REGISTER_SIZE == 32 ? (4,8,6) : (8,16,14)  #assumes size is either 32 or 64
end
pick_kernel_size(::Type{Core.VecElement{T}}) where T = pick_kernel_size(T)
