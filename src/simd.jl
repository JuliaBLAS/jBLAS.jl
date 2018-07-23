

# using TriangularMatrices

"""
Vec{N,T} is just a wrapper for NTuple{N,T}, to not commit type piracy.
I am not having it wrap NTuple{N,Core.VecElement{T}}, because I've gotten segfaults doing that.
"""
struct Vec{N,T} <: AbstractVector{T}
    # data::NTuple{N,Core.VecElement{T}}
    data::NTuple{N,T}
    Vec(data::Vararg{T,N}) where {N,T} = new{N,T}(data)
end


@inline Base.getindex(x::Vec, i) = x.data[i]
@inline Base.length(::Vec{N}) where N = N
@inline Base.size(::Vec{N}) where N = (N,)
@inline Base.eltype(::Vec{N,T}) where {N,T} = T

"""
If you're writing performance sensitive coat, the prefered interface would be converting your datastructure to the
appropriate Ptr{Vec{N,T}} type, eg

A = randn(4,5); # Matrix{Float64}
pA = Base.unsafe_convert(Ptr{Vec{4,Float64}}, pointer(A))
unsafe_load(pA, 1) # first vector of 4
unsafe_load(pA, 2) # second vector of 4
unsafe_load(pA, 3) # third vector of 4

The vload functions instead index with respect to element number, so the above is equivalent to
vload(Vec{4,Float64}, A, 0)
vload(Vec{4,Float64}, A, 4)
vload(Vec{4,Float64}, A, 8)
or
vload(pA, 0)
vload(pA, 4)
vload(pA, 8)

In case that interface is more convenient.
"""
@inline function vload(::Type{Vec{N,T}}, x::AbstractArray{T}, i) where {N,T}
    unsafe_load(Base.unsafe_convert(Ptr{Vec{N,T}}, pointer(x)) + sizeof(T)*i)
end
@inline function vload(::Type{Vec{N,T}}, px::Ptr, i) where {N,T}
    unsafe_load(Base.unsafe_convert(Ptr{Vec{N,T}}, px) + sizeof(T)*i)
end
@inline function vload(px::Ptr{Vec{N,T}}, i) where {N,T}
    # unsafe_load(p, i)
    unsafe_load(px + sizeof(T) * i)
end
@inline function vstore!(x::AbstractArray{T}, v::Vec{N,T}, i) where {N,T}
    unsafe_store!(Base.unsafe_convert(Ptr{Vec{N,T}}, pointer(x)) + sizeof(T)*i, v)
end
@inline function vstore!(x::Ptr{T}, v::Vec{N,T}) where {N,T}
    unsafe_store!(Base.unsafe_convert(Ptr{Vec{N,T}}, x), v)
end
@inline function vstore!(x::Ptr{T}, v::T) where {N,T}
    unsafe_store!(x, v)
end
@inline function vstore!(x::Ptr{Vec{N,T}}, v::Vec{N,T}) where {N,T}
    unsafe_store!(x, v)
end
function create_quote()
    q = quote
        @inbounds @fastmath begin
            $(Expr(:meta, :inline))
            Vec()
        end
    end
    # q, q.args[2].args[3].args[3].args
    q, q.args[2].args[3].args[3].args[4].args
end
@generated function Base.:+(x::Vec{N,T}, y::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] + y[$n]))
    end
    q
end
@generated function Base.:+(x::T, y::Vec{N,T}) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x + y[$n]))
    end
    q
end
@generated function Base.:+(x::Vec{N,T}, y::T) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] + y))
    end
    q
end
@generated function Base.:-(x::Vec{N,T}, y::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] - y[$n]))
    end
    q
end
@generated function Base.:-(x::T, y::Vec{N,T}) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x - y[$n]))
    end
    q
end
@generated function Base.:-(x::Vec{N,T}, y::T) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] - y))
    end
    q
end
@generated function Base.:*(x::Vec{N,T}, y::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] * y[$n]))
    end
    q
end
@generated function Base.:*(x::T, y::Vec{N,T}) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x * y[$n]))
    end
    q
end
@generated function Base.:*(x::Vec{N,T}, y::T) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] * y))
    end
    q
end
@generated function Base.:/(x::Vec{N,T}, y::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] / y[$n]))
    end
    q
end
@generated function Base.:/(x::T, y::Vec{N,T}) where {N,T<:Number}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x / y[$n]))
    end
    q
end
@generated function Base.:/(x::Vec{N,T}, y::T) where {N,T<:Number}
    q, qa = create_quote()
    push!(qa, :(yi = 1/y))
    for n ∈ 1:N
        push!(qa, :(x[$n] * yi))
    end
    q
end
@generated function Base.fma(x::Vec{N,T}, y::Vec{N,T}, z::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] * y[$n] + z[$n]))
    end
    q
end
@generated function Base.fma(x::T, y::Vec{N,T}, z::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x * y[$n] + z[$n]))
    end
    q
end
@generated function Base.fma(x::Vec{N,T}, y::T, z::Vec{N,T}) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] * y + z[$n]))
    end
    q
end
@generated function Base.fma(x::Vec{N,T}, y::Vec{N,T}, z::T) where {N,T}
    q, qa = create_quote()
    for n ∈ 1:N
        push!(qa, :(x[$n] * y[$n] + z))
    end
    q
end
# @generated function Base.fma(x::Vec{N,T}, y::T, z::T) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(x[$n] * y + z))
#     end
#     q
# end
# @generated function Base.fma(x::T, y::Vec{N,T}, z::T) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(x * y[$n] + z))
#     end
#     q
# end
# @generated function Base.fma(x::T, y::T, z::Vec{N,T}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(x * y + z[$n]))
#     end
# #     q
# # end
#
#
# @generated function Base.:+(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value + y[$n].value)))
#     end
#     q
# end
# @generated function Base.:+(x::T, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x + y[$n].value)))
#     end
#     q
# end
# @generated function Base.:+(x::Vec{N,Core.VecElement{T}}, y::T) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value + y)))
#     end
#     q
# end
# @generated function Base.:+(x::Vec{N,Core.VecElement{T}}, y::Core.VecElement{T}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value + y.value)))
#     end
#     q
# end
# @generated function Base.:-(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value - y[$n].value)))
#     end
#     q
# end
# @generated function Base.:-(x::T, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x - y[$n].value)))
#     end
#     q
# end
# @generated function Base.:-(x::Core.VecElement{T}, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x.value - y[$n].value)))
#     end
#     q
# end
# @generated function Base.:-(x::Vec{N,Core.VecElement{T}}, y::T) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value - y)))
#     end
#     q
# end
# @generated function Base.:-(x::Vec{N,Core.VecElement{T}}, y::Core.VecElement{T}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value - y.value)))
#     end
#     q
# end
# @generated function Base.:*(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y[$n].value)))
#     end
#     q
# end
# @generated function Base.:*(x::T, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x * y[$n].value)))
#     end
#     q
# end
# @generated function Base.:*(x::Core.VecElement{T}, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x.value * y[$n].value)))
#     end
#     q
# end
# @generated function Base.:*(x::Vec{N,Core.VecElement{T}}, y::T) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y)))
#     end
#     q
# end
# @generated function Base.:*(x::Vec{N,Core.VecElement{T}}, y::Core.VecElement{T}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y.value)))
#     end
#     q
# end
# @generated function Base.:/(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value / y[$n].value)))
#     end
#     q
# end
# @generated function Base.:/(x::T, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x / y[$n].value)))
#     end
#     q
# end
# @generated function Base.:/(x::Core.VecElement{T}, y::Vec{N,Core.VecElement{T}}) where {N,T<:Number}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x.value / y[$n].value)))
#     end
#     q
# end
# @generated function Base.:/(x::Vec{N,Core.VecElement{T}}, y::T) where {N,T<:Number}
#     q, qa = create_quote()
#     push!(qa, :(yi = 1/y))
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * yi)))
#     end
#     q
# end
# @generated function Base.:/(x::Vec{N,Core.VecElement{T}}, y::Core.VecElement{T}) where {N,T<:Number}
#     q, qa = create_quote()
#     push!(qa, :(yi = 1/y.value))
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * yi)))
#     end
#     q
# end
# @generated function Base.fma(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}, z::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y[$n].value + z[$n].value)))
#     end
#     q
# end
# @generated function Base.fma(x::T, y::Vec{N,Core.VecElement{T}}, z::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x * y[$n].value + z[$n].value)))
#     end
#     q
# end
# @generated function Base.fma(x::Core.VecElement{T}, y::Vec{N,Core.VecElement{T}}, z::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x.value * y[$n].value + z[$n].value)))
#     end
#     q
# end
# @generated function Base.fma(x::Vec{N,Core.VecElement{T}}, y::T, z::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y + z[$n].value)))
#     end
#     q
# end
# @generated function Base.fma(x::Vec{N,Core.VecElement{T}}, y::Core.VecElement{T}, z::Vec{N,Core.VecElement{T}}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y.value + z[$n].value)))
#     end
#     q
# end
# @generated function Base.fma(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}, z::T) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y[$n].value + z)))
#     end
#     q
# end
# @generated function Base.fma(x::Vec{N,Core.VecElement{T}}, y::Vec{N,Core.VecElement{T}}, z::Core.VecElement{T}) where {N,T}
#     q, qa = create_quote()
#     for n ∈ 1:N
#         push!(qa, :(Core.VecElement(x[$n].value * y[$n].value + z.value)))
#     end
#     q
# end
