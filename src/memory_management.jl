using Core.Intrinsics: llvmcall

struct PrefetchA
    A::Int
end
struct PrefetchX
    X::Int
end
struct PrefetchAX
    A::Int
    X::Int
end

# Base.:+(ptr::Ptr, offset::Prefetch) = ptr + offset.offset
# Base.:+(offset::Prefetch, ptr::Ptr) = ptr + offset.offset

# args are address, read/write, locality, cache type
@generated function prefetch(address, ::Val{Locality} = Val(1), ::Val{RorW} = Val(0)) where {Locality, RorW}
    prefetch_call_string = """%addr = inttoptr i64 %0 to i8*
    call void @llvm.prefetch(i8* %addr, i32 $RorW, i32 $Locality, i32 1)
    ret void"""
    quote
        $(Expr(:meta, :inline))
        llvmcall(("declare void @llvm.prefetch(i8* , i32 , i32 , i32 )",
        $prefetch_call_string), Cvoid, Tuple{Ptr{Cvoid}}, address)
    end
end

"""
Given matrices of size M, N, and P...
This matrix assumes 3 cache levels. This means it is not especially portable, beyond x86_64.
Level 1 and 2 is assumed to be local to each core, while level 3 is assumed shared.

How do I want it to work? Each level should divide well into the next. Means for one thing
I probably want to iterate over cache_size backwards?

Goal is to minimize actual data movement that goes on.
D (+)= A * X
Looking at the extreme of calculating a single kernel from D in full at a time,
we see that it
a) Minimizes loading and unloading D from registers.
    1) Does this maximize kernel performance? Kernels are then (m x N) * (N x p).
b)

Should add support for how many copies of each of the matrices we have, so that we can perform calculations such as
    D = A*X + C
    or
    D = A*(X + C)
in one step, without as much un/reloading of memory.
"""
function blocking_structure(M, N, P, ::Type{T} = Float64; cache_size::NTuple{3,Int} = CACHE_SIZE, D_count = 1, A_count = 1, X_count = 1) where T
    total_elements = M*N*D_count + N*P*A_count + M*P*X_count
    L1, L2, L3 = cache_size .÷ sizeof(T)
    if L1 > total_elements
        epr, m_1, p_1 = pick_kernel_size(T, D_count = D_count, A_count = A_count, X_count = X_count)
        # if m_1 <= M && p_1 <= P
        #     return ((M,N,P),(M,N,P),(M,N,P)),-1
        # else
        return ((min(m_1,M),N,min(p_1,P)),(M,N,P),(M,N,P)),0
        # end
        # return ((M,N,P),(M,N,P),(M,N,P)),0
    end

    epr, m_1, p_1 = pick_kernel_size(T, D_count = D_count, A_count = A_count, X_count = X_count)
    # @show m_1, p_1, L1, L2, L3
    Dmp_1 = m_1 * p_1

    n_1 = (L1 - Dmp_1) ÷ (m_1 + p_1)

    # I need to consider the case where
    # m_1 or p_1 can be some multiple of themselves due to N being too small.
    # n_2 = n_1 = min(N, n_1)
    if n_1 > N
        n_2 = n_1 = N
        # m_1, p_1 = divide_into_rough_square(L1, M, P, n_1, m_1, p_1)#, D_count = D_count, A_count = A_count, X_count = X_count)
    else
        n_2 = n_1
    end

    # else # currently not bothering to handle the "else".

    # end
    # num_elements = cache_size[i+1] ÷ sizeofT
    # 0 = m^2 + 2m*n_2 - L2

    if L2 > total_elements
        return ((m_1, n_1, p_1), (M, N, P), (M, N, P)),1
    end

    # Try to upper bound size of m_2, p_2
    # Consider safety factors, for other things (instructions?) in this cache?
    m_2, p_2 = divide_into_rough_square(L2, M, P, n_2, m_1, p_1)#, D_count = D_count, A_count = A_count, X_count = X_count)

    if L3 > total_elements
        return ((m_1, n_1, p_1), (m_2, n_2, p_2), (M, N, P)),2
    end

    Dmp_2 = m_2 * p_2
    n_3 = (L3 - Dmp_2) ÷ (m_2 + p_2)
    if n_3 > N
        m_3, p_3 = divide_into_rough_square(L3, M, P, n_3, m_2, p_2)#, D_count = D_count, A_count = A_count, X_count = X_count)
    else
        m_3, p_3 = m_2, p_2
    end

    (Base.Cartesian.@ntuple 3 i -> (m_i, n_i, p_i)),3

end

function divide_into_rough_square(L, M, P, n, mbase, pbase)
    L_upper_bound = floor(Int, sqrt(abs2(n) + L) - n)
    m_2 = max(round_x_to_nearest_y(L_upper_bound, mbase), mbase)
    if m_2 > M
        m_2 = M
        p_2 = min(P, (L - m_2*n) ÷ (m_2 + n) )
    else
        p_2 = L_upper_bound ÷ pbase * pbase
        if p_2 > P
            p_2 = P
            m_2 = min(M, (L - p_2*n) ÷ (p_2 + n) )
        end
    end
    m_2, p_2
end



function prefetch_storage(V::Type{Vec{L,T}}, CL, cache_loads, rows, cols, M, maxrc, maxcc, loc = 2, D = :pD) where {L,T}
    q1 = quote end
    q2 = quote end
    st = sizeof(T)
    for c ∈ 1:cols, cl ∈ 1:cache_loads
        push!(q1.args, :(prefetch($D + $((cl-1)*CL*st + M*(c - 1)*st + rows*st) + rc*$(rows*st) + cc*$(cols*M*st), Val($loc), Val(1))))
        push!(q2.args, :(prefetch($D + $((cl-1)*CL*st + M*(c - 1)*st + cols*M*st) + cc*$(cols*M*st), Val($loc), Val(1))))
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    quote
        if rc != $maxrc
            $q1
        elseif cc != $maxcc
            $q2
        end
    end
end

"""
Could introducing branches in these prefetch statements slow things down?
What are the consequences of these?
Should the new row prefetch be reformulated into an ifelse statement?
"""
function prefetch_load_Aiq(V::Type{Vec{L,T}}, CL, cache_loads, rows, cols, M,N, maxrc,
                            prefetch_freq, loc = 2, A = :pA) where {L,T}
    q1 = quote end
    q2 = quote end
    st = sizeof(T)
    for cl ∈ 0:cache_loads-1
        push!(q1.args, :(prefetch($(A) + $(cl*CL*st-M*st+M*st*prefetch_freq) + rc*$(rows*st), Val($loc), Val(0))))
        push!(q2.args, :(prefetch($(A) + $(cl*CL*st-(N-prefetch_freq+1)*M*st+rows*st) + rc*$(rows*st), Val($loc), Val(0))))
    end
    quote
        if 0 < $(N-prefetch_freq) # Don't jump to new row
            $q1
        elseif rc != $maxrc # Don't prefetch if there're no more rows to prefetch
            $q2
        end
    end
end
function prefetch_load_Aq(V::Type{Vec{L,T}}, CL, cache_loads, rows, cols, M,N, maxrc,
                            prefetch_freq, loc = 2, A = :pA) where {L,T}
    q1 = quote end
    q2 = quote end
    st = sizeof(T)
    for cl ∈ 0:cache_loads-1
        push!(q1.args, :(prefetch($(A) + $(cl*CL*st-M*st+M*st*prefetch_freq) + rc*$(rows*st) + n*$(M*st), Val($loc), Val(0))) )
        push!(q2.args, :(prefetch($(A) + $(cl*CL*st-(N-prefetch_freq+1)*M*st+rows*st) + rc*$(rows*st) + n*$(M*st), Val($loc), Val(0))) )
    end
    quote
        if n < $(N-prefetch_freq) # Don't jump to new row
            $q1
        elseif rc != $maxrc # Don't prefetch if there're no more rows to prefetch
            $q2
        end
    end
end
function prefetch_load_Xq(V::Type{Vec{L,T}}, CL, rows, cols, N, maxrc, maxcc,
                            prefetch_freq, loc = 2, X = :pX) where {L,T}
    q1 = quote end
    q2 = quote end
    st = sizeof(T)
    # for c ∈ 1:cols
    #     push!(q.args, :( $(Symbol(X,:_,c)) = ($V)($(X)[n,$c + cc*$cols]) ) )
    # end
    for c ∈ 1:cols
        push!(q1.args, :(prefetch($X + ($(c*N+prefetch_freq*CL) + cc*$(cols*N) + pxn)*$st , Val($loc), Val(0))))
        push!(q2.args, :(prefetch($X + ($((cols+c-1)*N+prefetch_freq*CL) + cc*$(cols*N) + pxn)*$st , Val($loc), Val(0))))
    end
    quote
        if pxn < $(N-prefetch_freq*CL)
            $q1
        elseif cc != $maxcc # plus (cols - 1) * N, because jumping cols to the right, and starting at top
            $q2
        end
    end
end
function prefetch_load_Xiq(V::Type{Vec{L,T}}, CL, rows, cols, N, maxrc, maxcc,
                            prefetch_freq, loc = 2, X = :pX) where {L,T}
    q1 = quote end
    q2 = quote end
    st = sizeof(T)
    # for c ∈ 1:cols
    #     push!(q.args, :( $(Symbol(X,:_,c)) = ($V)($(X)[n,$c + cc*$cols]) ) )
    # end
    for c ∈ 1:cols
        push!(q1.args, :(prefetch($X + ($(c*N+prefetch_freq*CL) + cc*$(cols*N) )*$st , Val($loc), Val(0))))
        push!(q2.args, :(prefetch($X + ($((cols+c-1)*N+prefetch_freq*CL) + cc*$(cols*N))*$st , Val($loc), Val(0))))
    end
    quote
        if 0 < $(N-prefetch_freq*CL)
            $q1
        elseif cc != $maxcc # plus (cols - 1) * N, because jumping cols to the right, and starting at top
            $q2
        end
    end
end
