mask_expr(W, r) = :($(Expr(:tuple, [i > r ? Core.VecElement{Bool}(false) : Core.VecElement{Bool}(true) for i ∈ 1:W]...)))

function mulinit(V, Q, Pₖ, X_stride, r, mask_expr, pfA_1)
    quote
        $(r == 0 ?
            :(@nexprs $Q q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1)))
                :
            :(@nexprs $(Q-1) q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
            $(Symbol(:vA_, Q)) = vload($V, pA + $(REGISTER_SIZE*(Q-1)),$mask_expr)))
        @nexprs $Pₖ p -> begin
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = vmul(vA_q, vX)
        end
        $pfA_1
    end
end
function gemminit(V, Q, Pₖ, AD_stride, r, mask_expr)
    if r == 0
        q = quote
            @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
    else
        q = quote end
        for p ∈ 1:Pₖ
            for q ∈ 1:Q-1
                push!(q.args, :($(Symbol(:Dx_,p,:_,q)) = vload($V, pD + $(REGISTER_SIZE*(q-1) + AD_stride*(p-1)))))
            end
            push!(q.args, :($(Symbol(:Dx_,p,:_,Q)) = vload($V, pD + $(REGISTER_SIZE*(Q-1) + AD_stride*(p-1)),$mask_expr)))
        end
    end
    q
end

function kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,init,pf::PrefetchAX = nothing) where {Mₖ,Pₖ,stride_AD,stride_X,N,T}
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    while W > 2Mₖ
        W >>= 1
    end
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    if r == 0
        mask = :(nothing)
    else
        Q += 1
        mask = mask_expr(W, r)
    end

    # if Q > 0
    #     r == 0 || throw("Number of rows $Mₖ not a multiple of register size: $REGISTER_SIZE.")
    # else
    #     W = r
    #     Q = 1
    # end
    V = Vec{W,T}
    C = CACHELINE_SIZE ÷ T_size
    Qₚ = cld(Mₖ, C)
    # Check whether we are prefetching A and/or X.
    pfA_1, pfA_2 = prefetch_A(pf)
    pfX_1, pfX_2, pfX_3 = prefetch_X(pf)
    if init
        q = mulinit(V, Q, Pₖ, X_stride, pfA_1)
    else
        q = gemminit(V, Q, Pₖ, AD_stride)
    end
    if pfX_1 == nothing
        push!(q.args,
        quote
            for n ∈ $(init ? 1 : 0):$(N-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vfma(vA_q, vX, Dx_p_q)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
            nothing
        end)
    else
        push!(q.args,
        quote
            # @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            for n ∈ $(init ? 1 : 0):$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vfma(vA_q, vX, Dx_p_q)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            # @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
            $pfX_1

            for n₁ ∈ $C:$C:$(N-C)
                for n ∈ n₁:n₁+$(C-1)
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                        @nexprs $Q q -> Dx_p_q = vfma(vA_q, vX, Dx_p_q)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
                # @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_2
            end
            for n ∈ $(N - (N % C)):$(N-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vfma(vA_q, vX, Dx_p_q)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> begin
                # prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_3
                @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
            end
            nothing
        end
    end
    q
end



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
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
        @nexprs $Q q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = vA_q * vX
        end
        for n ∈ 1:$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
        @nexprs $Q q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = vA_q * vX
        end
        @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        for n ∈ 1:$(C-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
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
        @nexprs $Q q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
        @nexprs $Pₖ p -> begin
            vX = $V(unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = vA_q * vX
        end
        for n ∈ 1:$(C-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
            end
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = $V(unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = fma(vA_q, vX, Dx_p_q)
            end
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end
