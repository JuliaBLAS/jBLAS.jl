function kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T)
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    quote
        @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        for n ∈ 0:$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
        end
        @nexprs $Pₖ p -> @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        nothing
    end
end

@generated function kernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T)
end
function initkernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T)
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    quote
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> begin
                vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
                Dx_p_q = vA_q * vX
            end
        end
        for n ∈ 1:$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
        end
        @nexprs $Pₖ p -> @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        nothing
    end
end
@generated function initkernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    initkernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T)
end
@generated function kernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchAX) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        for n₁ ∈ 0:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> begin
                        vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                        Dx_p_q = fma(vA_q, vX, Dx_p_q)
                    end
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end
@generated function initkernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchAX) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> begin
                vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
                Dx_p_q = vA_q * vX
            end
        end
        @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        for n ∈ 1:$(C-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> begin
                        vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                        Dx_p_q = fma(vA_q, vX, Dx_p_q)
                    end
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end

@generated function kernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchA) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        for n ∈ 0:$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        nothing
    end
end
@generated function initkernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchA) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> begin
                vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
                Dx_p_q = vA_q * vX
            end
        end
        @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        for n ∈ 1:$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        nothing
    end
end

@generated function kernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, ::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchX) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        for n₁ ∈ 0:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> begin
                        vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                        Dx_p_q = fma(vA_q, vX, Dx_p_q)
                    end
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end
@generated function initkernel!(pD::Ptr{T}, pA::Ptr{T}, pX::Ptr{T}, K::Kernel{Mₖ,Pₖ,stride_AD,stride_X,N}, pf::PrefetchX) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = Mₖ ÷ C
    quote
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> begin
                vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
                Dx_p_q = vA_q * vX
            end
        end
        for n ∈ 1:$(C-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> begin
                        vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                        Dx_p_q = fma(vA_q, vX, Dx_p_q)
                    end
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> begin
                    vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end
