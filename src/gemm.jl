

function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:cols, r ∈ 1:row_loads
        push!(q.args, :( vstore( $(Symbol(D,:_,r,:_,c)), $D + $((r-1)*L*st + M*(c-1)*st ) + rc*$(rows*st) + cc*$(cols*M*st)) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore($(Symbol(D,:_,r,:_,c)), $D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st + cc*cols*M*st)) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Symbol, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore($(Symbol(D,:_,r,:_,c)), $D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st) + $cc*$(cols*M*st)) ) )
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Symbol, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore($(Symbol(D,:_,r,:_,c)), $D + $((r-1)*L*st + M*(c-1)*st + cc*cols*M*st) + $rc*$(rows*st) ) ) )
    end
    q
end
function load_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:cols, r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = vstore($V, $D + $((r-1)*L*st + M*(c-1)*st ) + rc*$(rows*st) + cc*$(cols*M*st)) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function load_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = vload($V, $D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st + cc*cols*M*st)) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function load_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Symbol, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :($(Symbol(D,:_,r,:_,c)) = vload($V, $D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st) + $cc*$(cols*M*st)) ) )
    end
    q
end
function load_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Symbol, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = vload($V, $D + $((r-1)*L*st + M*(c-1)*st + cc*cols*M*st) + $rc*$(rows*st) ) ) )
    end
    q
end
function initialize_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N,
            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X, pX::Symbol = :pX) where {L,T}
    q = quote end
    st = sizeof(T)
    # prefetch() && push!(q.args, :(prefetch($(pX) + $(c*N) + cc*$(cols*N), Val(3), Val(0))))
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st) + rc*$(rows*st) )) )
        # prefetch() && push!(q.args, :(prefetch($(A) + $((r-1)*L*st-M*st + 2*M*st) + rc*$(rows*st), Val(3), Val(0))))
        # push!(q.args, :( @show $(Symbol(A,:_,r)) ))
    end
    for c ∈ 1:cols
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[1,$c + cc*$cols])) )
        # push!(q.args, :( $(Symbol(X,:_,c)) = ($V)(($X)[1,$c + cc*$cols]) ) )
        # push!(q.args, :( @show $(Symbol(X,:_,c)) ))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = $(Symbol(A,:_,r))*$(Symbol(X,:_,c)) ) )
            # push!(q.args, :( @show $(Symbol(D,:_,r,:_,c)) ))
        end
    end
    q
end
function initialize_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Int, cc::Symbol, pₖ::Int = cols,
            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X, pX::Symbol = :pX) where {L,T}
    q = quote end
    st = sizeof(T)
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st + rc*rows*st) )) )
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[1,$c + $cc*$cols]) ))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = $(Symbol(A,:_,r))*$(Symbol(X,:_,c)) ) )
        end
    end
    q
end
function initialize_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Symbol, cc::Int, pₖ::Int = cols,
            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X, pX::Symbol = :pX) where {L,T}
    q = quote end
    st = sizeof(T)
    # prefetch() && push!(q.args, :(prefetch($(pX) + $(c*N) + cc*$(cols*N), Val(3), Val(0))))
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st) + $rc*$(rows*st) )) )
        # prefetch() && push!(q.args, :(prefetch($(A) + $((r-1)*L*st-M*st + 2*M*st) + rc*$(rows*st), Val(3), Val(0))))
        # push!(q.args, :( @show $(Symbol(A,:_,r)) ))
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[1,$(c + cc*cols)]) ))
        # push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = ($V)(($X)[1,$(c + cc*cols)]) ) )
        # push!(q.args, :( @show $(Symbol(X,:_,c)) ))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = $(Symbol(A,:_,r))*$(Symbol(X,:_,c)) ) )
            # push!(q.args, :( @show $(Symbol(D,:_,r,:_,c)) ))
        end
    end
    q
end
function initialize_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Int, cc::Int, pₖ::Int = cols,
            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X, pX::Symbol = :pX) where {L,T}
    q = quote end
    st = sizeof(T)
    # prefetch() && push!(q.args, :(prefetch($(pX) + $(c*N) + cc*$(cols*N), Val(3), Val(0))))
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st + rc*rows*st) )) )
        # prefetch() && push!(q.args, :(prefetch($(A) + $((r-1)*L*st-M*st + 2*M*st) + rc*$(rows*st), Val(3), Val(0))))
        # push!(q.args, :( @show $(Symbol(A,:_,r)) ))
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[1,$(c + cc*cols)]) ))
        # push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = ($V)(($X)[1,$(c + cc*cols)]) ) )
        # push!(q.args, :( @show $(Symbol(X,:_,c)) ))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = $(Symbol(A,:_,r))*$(Symbol(X,:_,c)) ) )
            # push!(q.args, :( @show $(Symbol(D,:_,r,:_,c)) ))
        end
    end
    q
end
function fma_increment_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N,
                        D::Symbol = :D, A::Symbol = :A, X::Symbol = :X) where {L,T}
    q = quote end
    st = sizeof(T)
    # prefetch() && push!(q.args, :(prefetch($(pX) + $(c*N) + cc*$(cols*N), Val(3), Val(0))))
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st-M*st) + rc*$(rows*st) + n*$(M*st) )) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(0))))
        # + $(r*vector_length*st-M*st) + rc*$(rows*st) + n*$(M*st) )) )
        # push!(q.args, :( @show $(Symbol(A,:_,r)) ))
    end
    for c ∈ 1:cols
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[n,$c + cc*$cols])) )
        # push!(q.args, :( $(Symbol(X,:_,c)) = ($V)($(X)[n,$c + cc*$cols]) ) )
        # push!(q.args, :( @show $(Symbol(X,:_,c)) ))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = fma( $(Symbol(A,:_,r)),$(Symbol(X,:_,c)),$(Symbol(D,:_,r,:_,c))) ) )
            # push!(q.args, :( @show $(Symbol(D,:_,r,:_,c)) ))
        end
    end
    q
end
function fma_increment_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Symbol, cc::Int, pₖ::Int = cols,
                            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X) where {L,T}
    q = quote end
    st = sizeof(T)
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st-M*st) + n*$(M*st) + $rc*$(rows*st)  ) ))
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[n,$(c + cc*cols)])) )
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = fma( $(Symbol(A,:_,r)),$(Symbol(X,:_,c)),$(Symbol(D,:_,r,:_,c))) ) )
        end
    end
    q
end
function fma_increment_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Int, cc::Symbol, pₖ::Int = cols,
                            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X) where {L,T}
    q = quote end
    st = sizeof(T)
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st-M*st + rc*rows*st) + n*$(M*st) )) )
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[n, $c + $cc*$cols])))
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = fma( $(Symbol(A,:_,r)),$(Symbol(X,:_,c)),$(Symbol(D,:_,r,:_,c))) ) )
        end
    end
    q
end

function fma_increment_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, N, rc::Int, cc::Int, pₖ::Int = cols,
                            D::Symbol = :D, A::Symbol = :A, X::Symbol = :X) where {L,T}
    q = quote end
    st = sizeof(T)
    for r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(A,:_,r)) = vload($V, $(A) + $((r-1)*L*st-M*st + rc*rows*st) + n*$(M*st) )) )
    end
    for c ∈ 1:pₖ
        push!(q.args, :( @inbounds $(Symbol(X,:_,c)) = $V($X[n,$(c + cc*cols)])) )
        for r ∈ 1:row_loads
            push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = fma( $(Symbol(A,:_,r)),$(Symbol(X,:_,c)),$(Symbol(D,:_,r,:_,c))) ) )
        end
    end
    q
end



# Was deprecated in 0.7
# @inline sub2ind((M,N)::Tuple{Int,Int}, i, j) = i + (j-1)*M

# @generated function smul!(D::SizedArray{Tuple{M,P},T}, A::SizedArray{Tuple{M,N},T}, X::SizedArray{Tuple{N,P},T}) where {T,M,N,P}

"""
jmul!(D::MMatrix{M,P,T}, A::MMatrix{M,N,T}, X::MMatrix{N,P,T},
    ::Val{Aprefetch_freq} = Val(7), ::Val{Xprefetch_freq} = Val(7),
    ::Val{A_loc} = Val(3), ::Val{X_loc} = Val(3), ::Val{D_loc} = Val(3))

It may take some playing with the prefetching arguments.

jmul!(D, A, X)

calculates D = A * X


jmul!(D, A, X, Val(Aprefetch), Val(Xprefetch))

The prefetch val options determine the prefetch lag. That is, how far in advance will a prefetch instruction be sent?

Prefetching needs serious work on Haswell processors.
Likely, on every intel non-avx512 processors.
"""
@generated function jmul!(D::MMatrix{M,P,T}, A::MMatrix{M,N,T}, X::MMatrix{N,P,T},
    ::Val{Aprefetch_freq} = Val(7), ::Val{Xprefetch_freq} = Val(7),
    ::Val{A_loc} = Val(3), ::Val{X_loc} = Val(3), ::Val{D_loc} = Val(3)) where {T,M,N,P,Aprefetch_freq,Xprefetch_freq,A_loc,X_loc,D_loc}#,Aprefetch_freq2,Xprefetch_freq2,A_loc2,X_loc2}
    # ,::Val{Aprefetch_freq2} = Val(8), ::Val{Xprefetch_freq2} = Val(8),
    # ::Val{A_loc2} = Val(2), ::Val{X_loc2} = Val(2)) where {T,M,N,P,Aprefetch_freq,Xprefetch_freq,A_loc,X_loc,D_loc,Aprefetch_freq2,Xprefetch_freq2,A_loc2,X_loc2}

    # Preftech freq actually very architecture dependendent!!!
    # Going to have to pick that somehow, too.
    # For now, if it isn't AVX512, we'll assume it should be much less aggressive.

# @generated function smul!(D::SizedArray{Tuple{M,P},T}, A::SizedArray{Tuple{M,N},T}, X::SizedArray{Tuple{N,P},T}) where {T,M,N,P}
    # t_size = sizeof(T)
    # cacheline_size = 64 ÷ t_size
    # N = avx512 ? 64 ÷ t_size : 32 ÷ t_size
    vector_length, rows, cols = pick_kernel_size(T) #VL = VecLength

    if architecture == :Ryzen #Should make
        Aprefetch_freq *= 8
        Xprefetch_freq *= 8
    end


    row_chunks, row_remainder = divrem(M, rows)
    col_chunks, col_remainder = divrem(P, cols)

    row_loads = rows ÷ vector_length
    # col_loads = cols ÷
    V = Vec{vector_length, T}

    init_block = initialize_block(V, row_loads, rows, cols, M, N, :pD, :pA)#, :pX)
    fma_block = fma_increment_block(V, row_loads, rows, cols, M, N, :pD, :pA)#, :pX)
    store = store_block(V, row_loads, rows, cols, M, :pD)
        # init_block = initialize_block(V, row_loads, rows, cols)#, :pD, :pA, :pX)
        # fma_block = fma_increment_block(V, row_loads, rows, cols, M)#, :pD, :pA, :pX)
        # store = store_block(V, row_loads, rows, cols, M)#, :pD)

    cache_length = CACHELINE_SIZE ÷ sizeof(T)
    cache_loads = rows ÷ cache_length
    # prefetch_freq = 8
    maxrc, maxcc = row_chunks-1, col_chunks-1
    prefetch_store = prefetch_storage(V, cache_length, cache_loads, rows, cols, M, maxrc, maxcc, D_loc)


    prefetch_load_A = prefetch_load_Aq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq, A_loc)
    prefetch_load_Ai = prefetch_load_Aiq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq, A_loc)
    prefetch_load_X = prefetch_load_Xq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq, X_loc)
    prefetch_load_Xi = prefetch_load_Xiq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq, X_loc)


    # prefetch_load_A2 = prefetch_load_Aq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq2, A_loc2)
    # prefetch_load_Ai2 = prefetch_load_Aiq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq2, A_loc2)
    # prefetch_load_X2 = prefetch_load_Xq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq2, X_loc2)
    # prefetch_load_Xi2 = prefetch_load_Xiq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq2, X_loc2)

    # purpose of this is to break iteration of N up into these chunks, to prefetch X vectors.
    Nd, Nr = divrem(N-1, cache_length)
    Nr += 1
    if Nr < cache_length ÷ 3 && Nd > 0
        Nr += cache_length
        Nd -= 1
    end


    q = quote
        # pD = pointer(D)
        pA = pointer(A)
        pX = pointer(X)
        pD = pointer(D)
        @inbounds begin
            for cc ∈ 0:$(col_chunks-1), rc ∈ 0:$(row_chunks-1) #row_chunks total
                $prefetch_load_Xi
                $prefetch_load_Ai
                # $prefetch_load_Xi2
                # $prefetch_load_Ai2
                $init_block
                for n ∈ 2:$Nr
                    $prefetch_load_A
                    # $prefetch_load_A2
                    $fma_block
                end
                for nd ∈ 1:$Nd
                    pxn = (nd-1)*$cache_length+$Nr
                    $prefetch_load_X
                    # $prefetch_load_X2
                    for n ∈ pxn+1:pxn+$cache_length
                        $prefetch_load_A
                        # $prefetch_load_A2
                        $fma_block
                    end
                end
                $store
                $prefetch_store
            end
        end
        D
    end
    # push!(q.args[6].args[3],
    # quote
    #     for cc ∈ $(P-col_remainder+1:P), rc ∈ $(M-row_remainder+1:M)# rest, should do some remaining cache line size first
    #
    #     end
    # end)

    q
end
#
# struct UnsafeArray{T,N}
#     pointer::Ptr{T}
#     UnsafeArray(p::Ptr{T},::Val{N}) where {T,N} = new{T,N}(p)
# end
#
# Base.getindex(x::UnsafeArray,i) = unsafe_load(x.pointer, i)
# Base.getindex(x::UnsafeArray{T,N},i,j) where {T,N} = unsafe_load(x.pointer, i+(j-1)*N)
# #
# @generated function jmul!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T},::Val{M},::Val{N},::Val{P},
#     ::Val{Aprefetch_freq} = Val(7), ::Val{Xprefetch_freq} = Val(7),
#     ::Val{A_loc} = Val(3), ::Val{X_loc} = Val(3), ::Val{D_loc} = Val(3)) where {T,M,N,P,Aprefetch_freq,Xprefetch_freq,A_loc,X_loc,D_loc}#,Aprefetch_freq2,Xprefetch_freq2,A_loc2,X_loc2}
#     # ,::Val{Aprefetch_freq2} = Val(8), ::Val{Xprefetch_freq2} = Val(8),
#     # ::Val{A_loc2} = Val(2), ::Val{X_loc2} = Val(2)) where {T,M,N,P,Aprefetch_freq,Xprefetch_freq,A_loc,X_loc,D_loc,Aprefetch_freq2,Xprefetch_freq2,A_loc2,X_loc2}
#
#     # Preftech freq actually very architecture dependendent!!!
#     # Going to have to pick that somehow, too.
#     # For now, if it isn't AVX512, we'll assume it should be much less aggressive.
#
# # @generated function smul!(D::SizedArray{Tuple{M,P},T}, A::SizedArray{Tuple{M,N},T}, X::SizedArray{Tuple{N,P},T}) where {T,M,N,P}
#     # t_size = sizeof(T)
#     # cacheline_size = 64 ÷ t_size
#     # N = avx512 ? 64 ÷ t_size : 32 ÷ t_size
#     vector_length, rows, cols = pick_kernel_size(T) #VL = VecLength
#
#     if vector_length == 4
#         Aprefetch_freq *= 8
#         Xprefetch_freq *= 8
#     end
#
#
#     row_chunks, row_remainder = divrem(M, rows)
#     col_chunks, col_remainder = divrem(P, cols)
#
#     row_loads = rows ÷ vector_length
#     # col_loads = cols ÷
#     V = Vec{vector_length, T}
#
#     init_block = initialize_block(V, row_loads, rows, cols, M, N, :pD, :pA)#, :pX)
#     fma_block = fma_increment_block(V, row_loads, rows, cols, M, N, :pD, :pA)#, :pX)
#     store = store_block(V, row_loads, rows, cols, M, :pD)
#         # init_block = initialize_block(V, row_loads, rows, cols)#, :pD, :pA, :pX)
#         # fma_block = fma_increment_block(V, row_loads, rows, cols, M)#, :pD, :pA, :pX)
#         # store = store_block(V, row_loads, rows, cols, M)#, :pD)
#
#     cache_length = 64 ÷ sizeof(T)
#     cache_loads = rows ÷ cache_length
#     # prefetch_freq = 8
#     maxrc, maxcc = row_chunks-1, col_chunks-1
#     prefetch_store = prefetch_storage(V, cache_length, cache_loads, rows, cols, M, maxrc, maxcc, D_loc)
#
#
#     prefetch_load_A = prefetch_load_Aq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq, A_loc)
#     prefetch_load_Ai = prefetch_load_Aiq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq, A_loc)
#     prefetch_load_X = prefetch_load_Xq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq, X_loc)
#     prefetch_load_Xi = prefetch_load_Xiq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq, X_loc)
#
#
#     # prefetch_load_A2 = prefetch_load_Aq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq2, A_loc2)
#     # prefetch_load_Ai2 = prefetch_load_Aiq(V, cache_length, cache_loads, rows, cols, M,N, maxrc, Aprefetch_freq2, A_loc2)
#     # prefetch_load_X2 = prefetch_load_Xq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq2, X_loc2)
#     # prefetch_load_Xi2 = prefetch_load_Xiq(V, cache_length, rows, cols, N, maxrc, maxcc, Xprefetch_freq2, X_loc2)
#
#     # purpose of this is to break iteration of N up into these chunks, to prefetch X vectors.
#     Nd, Nr = divrem(N-1, cache_length)
#     Nr += 1
#     if Nr < cache_length ÷ 3 && Nd > 0
#         Nr += cache_length
#         Nd -= 1
#     end
#
#
#     q = quote
#         # pD = Base.unsafe_convert(Ptr{$V}, pointer(D))
#         # pA = Base.unsafe_convert(Ptr{$V}, pointer(A))
#         # pX = Base.unsafe_convert(Ptr{$V}, pointer(X))
#         X = UnsafeArray(pX,Val{$N}())
#         # pD = Base.unsafe_convert(Ptr{$V}, pointer(D))
#         @inbounds begin
#             for cc ∈ 0:$(col_chunks-1), rc ∈ 0:$(row_chunks-1) #row_chunks total
#                 $prefetch_load_Xi
#                 $prefetch_load_Ai
#                 # $prefetch_load_Xi2
#                 # $prefetch_load_Ai2
#                 $init_block
#                 for n ∈ 2:$Nr
#                     $prefetch_load_A
#                     # $prefetch_load_A2
#                     $fma_block
#                 end
#                 for nd ∈ 1:$Nd
#                     pxn = (nd-1)*$cache_length+$Nr
#                     $prefetch_load_X
#                     # $prefetch_load_X2
#                     for n ∈ pxn+1:pxn+$cache_length
#                         $prefetch_load_A
#                         # $prefetch_load_A2
#                         $fma_block
#                     end
#                 end
#                 $store
#                 $prefetch_store
#             end
#         end
#         # D
#     end
#     # push!(q.args[6].args[3],
#     # quote
#     #     for cc ∈ $(P-col_remainder+1:P), rc ∈ $(M-row_remainder+1:M)# rest, should do some remaining cache line size first
#     #
#     #     end
#     # end)
#
#     q
# end
#
#
#
# # D = randn(32*20, 36*20);
# # A = randn(32*20, 32*25);
# # X = randn(32*25, 36*20);
