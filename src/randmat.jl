# Current StaticArrays functions are ill suited for creating large matrices.
# At large sizes, generating unrolled expressions takes an eternity (I haven't been patient enough to find out how long),
# while for loops compile instantly.

function Random.rand!(x::MArray)
    @inbounds for i ∈ eachindex(x)
        x[i] = randn()
    end
    x
end
function mrandn(M,N)
    out = MMatrix{M,N,Float64}(undef)
    rand!(out)
end

function Base.fill!(D::MArray, x)
    @inbounds for i ∈ eachindex(D)
        D[i] = x
    end
    D
end
