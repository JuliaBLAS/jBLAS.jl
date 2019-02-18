mask_expr(W, r) = :($(Expr(:tuple, [i > r ? Core.VecElement{Bool}(false) : Core.VecElement{Bool}(true) for i ∈ 1:W]...)))

function mulinit(V, WT, Q, Pₖ, X_stride, r, mask_expr, inline_expr, pfA_1)
    # if r == 0
        q_load_expr = :(@nexprs $Q q -> vA_q = vload($V, pA + $WT*(q-1)))
    # else
    #     q_load_expr = quote
    #         @nexprs $(Q-1) q -> vA_q = vload($V, pA + $WT*(q-1))
    #       $(Symbol(:vA_, Q)) = vload($V, pA + $(WT*(Q-1)),$mask_expr)
    #   end
    # end

    quote
        $inline_expr
        $q_load_expr
        @nexprs $Pₖ p -> begin
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = SIMDPirates.vmul(vA_q, vX)
        end
        $pfA_1
    end
end
function gemminit(V, WT, Q, Pₖ, AD_stride, r, mask_expr, inline_expr)
    if r == 0
        q = quote
            $inline_expr
            @nexprs $Pₖ p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $WT*(q-1) + $AD_stride*(p-1))
        end
    else
        q = quote
            $inline_expr
            @nexprs $(Pₖ-1) p -> @nexprs $Q q -> Dx_p_q = vload($V, pD + $WT*(q-1) + $AD_stride*(p-1))
        end
        for q ∈ 1:Q-1
            push!(q.args, :($(Symbol(:Dx_,Pₖ,:_,q)) = vload($V, pD + $(WT*(q-1) + AD_stride*(Pₖ-1)))))
        end
        push!(q.args, :($(Symbol(:Dx_,Pₖ,:_,Q)) = vload($V, pD + $(WT*(Q-1) + AD_stride*(Pₖ-1)),$mask_expr)))
    end
    q
end

function kernel_quote(Mₖ,Pₖ,stride_AD,stride_X,N,T,init,inline = false, pf = nothing)
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    X_stride = stride_X * T_size
    W = REGISTER_SIZE ÷ T_size
    while W >= 2Mₖ
        W >>= 1
    end
    WT = W * T_size
    Q, r = divrem(Mₖ, W) #Assuming Mₖ is a multiple of W
    V = Vec{W,T}
    if r == 0
        mask = :()
        A_load_expr = :(@nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1)))
        D_store1 = :(@nexprs $Q q -> vstore(Dx_p_q, pD + $WT*(q-1) + $AD_stride*(p-1)))
        D_store2 = :(@nexprs $Q q -> vstore($(Symbol(:Dx_,Pₖ,:_q)), pD + $WT*(q-1) + $(AD_stride*(Pₖ-1))))
    else
        mask = mask_expr(W, r)
        if Q == 0
            Q = 1
            A_load_expr = :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*AD_stride) + $(WT*(Q-1)), $mask))
        else
            A_load_expr = quote
                @nexprs $Q q -> vA_q = vload($V, pA + $((N-1)*AD_stride) + $WT*(q-1))
            end
            Q += 1
            push!(A_load_expr.args, :($(Symbol(:vA_, Q)) = vload($V, pA + $((N-1)*AD_stride) + $(WT*(Q-1)), $mask)))
        end
        # D_store1 = quote
        #             @nexprs $(Q-1) q -> vstore(Dx_p_q, pD + $WT*(q-1) + $AD_stride*(p-1))
        #             vstore($(Symbol(:Dx_p_, Q)), pD + $(WT*(Q-1)) + $AD_stride*(p-1), $mask)
        #         end
        D_store1 = :(@nexprs $Q q -> vstore(Dx_p_q, pD + $WT*(q-1) + $AD_stride*(p-1)))
        D_store2 = quote
            @nexprs $(Q-1) q -> vstore($(Symbol(:Dx_,Pₖ,:_q)), pD + $WT*(q-1) + $(AD_stride*(Pₖ-1)))
            vstore($(Symbol(:Dx_, Pₖ, :_, Q)), pD + $(WT*(Q-1) + AD_stride*(Pₖ-1)), $mask)
        end
    end
    C = min(CACHELINE_SIZE ÷ T_size,N)
    Qₚ = cld(Mₖ, C)
    # Check whether we are prefetching A and/or X.
    pfA_1, pfA_2, pfA_3 = prefetch_A(pf, N)
    pfX_1, pfX_2, pfX_3, pfX_4 = prefetch_X(pf, N, Pₖ)
    inline_expr = inline ? Expr(:meta, :inline) : :(nothing)
    if init
        q = mulinit(V, WT, Q, Pₖ, X_stride, r, mask, inline_expr, pfA_1)
    else
        q = gemminit(V, WT, Q, Pₖ, AD_stride, r, mask, inline_expr)
    end

    if pfX_1 == nothing
        push!(q.args,
        quote
            for n ∈ $(init ? 1 : 0):$(r == 0 ? N-1 : N-2 )
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
        end)
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + $((N-1)*T_size) + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                $pfA_3
                @nexprs $Pₖ p -> $D_store1
                nothing
            end )
        else
            push!(q.args,
            quote
                @nexprs $(Pₖ-1) p -> $D_store1
                $D_store2
                nothing
            end)
        end
    else
        push!(q.args,
        quote
            # @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            for n ∈ $(init ? 1 : 0):$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                $pfA_2
                # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            # @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
            $pfX_1
        end)
        if (N - (N % C) == N) && (r > 0)
            C_upper_bound = N - 2C
            must_finish_iter = true
            remaining_iterations = N-2C+1:N-C-1
        else
            C_upper_bound = N - C
            must_finish_iter = N - (N % C) < (r == 0 ? N-1 : N-2 )
            remaining_iterations = (N - (N % C)):(r == 0 ? N-1 : N-2 )
        end
        push!(q.args,
        quote
            for n₁ ∈ $C:$C:$C_upper_bound
                for n ∈ n₁:n₁+$(C-1)
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                        @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
                # @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_2
            end
        end)
        if must_finish_iter
            push!(q.args,
            quote
                for n ∈ $remaining_iterations
                    @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $WT*(q-1))
                    @nexprs $Pₖ p -> begin
                        vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                        @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                    end
                    $pfA_2
                    # @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
                end
            end)
        end
        if r > 0
            push!(q.args,
            quote
                $A_load_expr
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + $((N-1)*T_size) + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                $pfA_3
            end )
        end

        push!(q.args,
        quote
            @nexprs $(Pₖ-1) p -> begin
                # prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
                $pfX_3
                $D_store1
            end
            $pfX_4
            $D_store2
            nothing
        end)
    end
    q
end
@generated function fastmul!(D::MMatrix{M,P,T},A::MMatrix{M,N,T},X::MMatrix{N,P,T}) where {M,N,P,T}
    quote
        pD, pA, pX = pointer(D), pointer(A), pointer(X)
        # Mₖ,Pₖ,stride_AD,stride_X,N,T,init
        $(kernel_quote(M,P,M,N,N,T,true,true))
    end
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
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
        $(Expr(:meta,:inline))
        @nexprs $Q q -> vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
        @nexprs $Pₖ p -> begin
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = SIMDPirates.evmul(vA_q, vX)
        end
        for n ∈ 1:$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = SIMDPirates.evmul(vA_q, vX)
        end
        @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        for n ∈ 1:$(C-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
            end
            @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
                @nexprs $Qₚ q -> prefetch(pA + pf.A + n*$AD_stride + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> begin
                vA_q = vload($V, pA + $REGISTER_SIZE*(q-1))
                Dx_p_q = SIMDPirates.evmul(vA_q, vX)
            end
        end
        @nexprs $Qₚ q -> prefetch(pA + pf.A + $CACHELINE_SIZE*(q-1), Val(3), Val(0))
        for n ∈ 1:$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
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
            vX = vbroadcast($V, unsafe_load(pX + (p-1)*$X_stride))
            @nexprs $Q q -> Dx_p_q = SIMDPirates.evmul(vA_q, vX)
        end
        for n ∈ 1:$(C-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
            end
        end
        @nexprs $Pₖ p -> prefetch(pX + pf.X + (p-1)*$X_stride, Val(3), Val(0))
        for n₁ ∈ $C:$C:$(N-C)
            for n ∈ n₁:n₁+$(C-1)
                @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
                @nexprs $Pₖ p -> begin
                    vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                    @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
                end
            end
            @nexprs $Pₖ p -> prefetch(pX + pf.X + n₁*$T_size + (p-1)*$X_stride, Val(3), Val(0))
        end
        for n ∈ $(N - (N % C)):$(N-1)
            @nexprs $Q q -> vA_q = vload($V, pA + n*$AD_stride + $REGISTER_SIZE*(q-1))
            @nexprs $Pₖ p -> begin
                vX = vbroadcast($V, unsafe_load(pX + n*$T_size + (p-1)*$X_stride))
                @nexprs $Q q -> Dx_p_q = vmuladd(vA_q, vX, Dx_p_q)
            end
        end
        @nexprs $Pₖ p -> begin
            prefetch(pX + pf.X + $(N*T_size) + (p-1)*$X_stride, Val(3), Val(0))
            @nexprs $Q q -> vstore(Dx_p_q, pD + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
        end
        nothing
    end
end
