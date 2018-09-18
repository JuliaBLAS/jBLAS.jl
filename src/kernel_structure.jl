
"""
Base.@pure Kernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}()

The kernel is typed based on M and P. M is a critical component of vector length,
while the kernel is unrolled across P. It simply loops over N.
"""
struct Kernel{Mₖ,Pₖ,stride_AD,stride_X,N} end
Base.@pure Kernel(Mₖ,N,Pₖ,stride_AD,stride_X) = Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}()
# struct Kernel{T}
#     M::Int
#     N::Int
#     P::Int
# end

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
function kernel_size_summary(::Type{T} = Float64) where T
    T_size = sizeof(T)
    num_per_register = REGISTER_SIZE ÷ T_size
    cache_line = num_per_register 
    # cache_line = CACHELINE_SIZE ÷ T_size
    max_total = num_per_register * REGISTER_COUNT
    num_cache_lines = cld(max_total, cache_line)
    summary = Matrix{Float64}(undef, 5, num_cache_lines)
    for a_loads ∈ 1:num_cache_lines
        num_rows = a_loads * REGISTER_SIZE
        num_cols = (REGISTER_COUNT - a_loads - 1) ÷ a_loads # assumes we need only a single B
        length_D = num_rows * num_cols
        num_loads = num_cols + a_loads
        summary[:, a_loads] .= (num_rows, num_cols, length_D, num_loads, length_D / num_loads)
        if num_cols == 0
            println("A * X = D; nrow(D), ncol(D), length(D), NumRegeristersLoaded")
            return summary[:, 1:num_row_cachelines]
        end
    end
    println("A * X = D; nrow(D), ncol(D), length(D), NumRegeristersLoaded")
    summary
end

function num_cols_and_loads(num_rows, elements_per_register)
    a_loads = cld(num_rows, elements_per_register)
    (REGISTER_COUNT - a_loads - 2) ÷ a_loads, a_loads
end

"""
num_cols = elements_per_register * ( REGISTER_COUNT - 2 ) / num_rows - 1
# square, assume approx equal
num_rows = elements_per_register * ( REGISTER_COUNT - 2 ) / num_rows - 1
0 = num_rows^2 + num_rows - elements_per_register * ( REGISTER_COUNT - 2 )


Returns number of elements per register,
and the number of rows and columns of the kernel.
Ie, 8, 16, 14 means that 8 elements fit per register,
and the optimal kernel is
16x14 = 16xN * Nx14
matrix multiplication.

Rather than all the noise above, we just pick something close to square because:
(L1_cache_size - rows*colums) ÷ (rows + columns)
will be maximized with relatively square blocks, making that friendliest for the cache.
"""
function pick_kernel_size(::Type{T}; D_count = 1, A_count = 1, X_count = 1) where T
    T_size = sizeof(T)
    elements_per_register = REGISTER_SIZE ÷ T_size
    cache_line = elements_per_register
    # cache_line = CACHELINE_SIZE ÷ T_size
    max_total = elements_per_register * REGISTER_COUNT
    num_cache_lines = cld(max_total, cache_line)
    prev_num_rows, prev_num_cols = 0, 0
    prev_ratio = -Inf
    for a_loads ∈ 1:num_cache_lines
        num_rows = a_loads * elements_per_register
        num_cols = (REGISTER_COUNT - a_loads - 1) ÷ a_loads # assumes we need only a single B
        length_D = num_rows * num_cols
        num_loads = num_cols + a_loads
        next_ratio = length_D / num_loads
        if next_ratio < prev_ratio
            break
        else
            prev_ratio = next_ratio
            prev_num_rows, prev_num_cols = num_rows, num_cols
        end
    end
    elements_per_register, prev_num_rows, prev_num_cols
end
# function pick_kernel_size(::Type{T}; D_count = 1, A_count = 1, X_count = 1) where T
#     T_size = sizeof(T)
#     elements_per_register = REGISTER_SIZE ÷ T_size
#     elements_in_cacheline = elements_per_register
#     # elements_in_cacheline = CACHELINE_SIZE ÷ T_size
#     elements_in_registers = elements_per_register * REGISTER_COUNT
#     approx = sqrt(0.25 + elements_per_register * (REGISTER_COUNT-2)) - 0.5
#     many_rows = round_x_to_nearest_y(approx, elements_in_cacheline, RoundUp)
#     few_rows  = round_x_to_nearest_y(approx, elements_in_cacheline, RoundDown)
#     mr_cols, mr_al = num_cols_and_loads(many_rows, elements_per_register)
#     few_rows < 1 && return elements_per_register, many_rows, mr_cols
#     fr_cols, fr_al = num_cols_and_loads(few_rows,  elements_per_register)
#     if many_rows * mr_cols / (mr_cols + mr_al) < few_rows * fr_cols / (fr_cols + fr_al)
#         return elements_per_register, few_rows, fr_cols
#     else
#         return elements_per_register, many_rows, mr_cols
#     end
# end
# pick_kernel_size(::Type{Core.VecElement{T}}) where T = pick_kernel_size(T)
