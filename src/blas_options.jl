
struct BLASOptions{T}

    mₖ::Int
    pₖ::Int

end

function store_block(opt::BLASOptions{T}, V::Type{Vec{L,T}}, row_loads, rows, cols, M, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:opt.pₖ, r ∈ 1:opt.mₖ
        push!(q.args, :( vstore!( $D + $((r-1)*L*st + M*(c-1)*st ) + rc*$(rows*st) + cc*$(cols*M*st), $(Symbol(D,:_,r,:_,c))) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end


function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:cols, r ∈ 1:row_loads
        push!(q.args, :( vstore!( $D + $((r-1)*L*st + M*(c-1)*st ) + rc*$(rows*st) + cc*$(cols*M*st), $(Symbol(D,:_,r,:_,c))) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore!($D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st + cc*cols*M*st), $(Symbol(D,:_,r,:_,c))) ) )
        # prefetch() && push!(q.args, :(prefetch($(Symbol(:p, A,:_,r)) + $(M*st) , Val(3), Val(1))))
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Int, cc::Symbol, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore!($D + $((r-1)*L*st + M*(c-1)*st + rc*rows*st) + $cc*$(cols*M*st), $(Symbol(D,:_,r,:_,c))) ) )
    end
    q
end
function store_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, rc::Symbol, cc::Int, pₖ::Int = cols, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:pₖ, r ∈ 1:row_loads
        push!(q.args, :( vstore!($D + $((r-1)*L*st + M*(c-1)*st + cc*cols*M*st) + $rc*$(rows*st), $(Symbol(D,:_,r,:_,c))) ) )
    end
    q
end
function load_block(V::Type{Vec{L,T}}, row_loads, rows, cols, M, D::Symbol = :D) where {L,T}
    q = quote end
    st = sizeof(T)
    for c ∈ 1:cols, r ∈ 1:row_loads
        push!(q.args, :( $(Symbol(D,:_,r,:_,c)) = vstore!($D + $((r-1)*L*st + M*(c-1)*st ) + rc*$(rows*st) + cc*$(cols*M*st), $V) ) )
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
