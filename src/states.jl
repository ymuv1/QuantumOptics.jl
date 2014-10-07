module states

using ..bases

export StateVector, Bra, Ket,
       tensor, dagger, ⊗,
       basis_bra, basis_ket,
       zero!, coherent_state


abstract StateVector

type Bra <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
end

type Ket <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
end

Bra(b::Basis) = Bra(b, zeros(Complex, length(b)))
Ket(b::Basis) = Ket(b, zeros(Complex, length(b)))


check = check_multiplicable

*(a::Bra, b::Ket) = (check(a.basis, b.basis); sum(a.data.*b.data))
*{T<:StateVector}(a::Number, b::T) = T(b.basis, complex(a)*b.data)
*{T<:StateVector}(a::T, b::Number) = T(a.basis, complex(b)*a.data)

/{T<:StateVector}(a::T, b::Number) = T(a.basis, a.data/complex(b))

+{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data+b.data) : throw(IncompatibleBases()))

-{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data-b.data) : throw(IncompatibleBases()))


tensor{T<:StateVector}(a::T, b::T) = T(compose(a.basis, b.basis), kron(a.data, b.data))
⊗(a,b) = tensor(a,b)

dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))

function basis_vector(shape::Vector{Int}, index::Vector{Int})
    x = zeros(Complex, shape...)
    x[index] = Complex(1.)
    reshape(x, prod(shape))
end

basis_bra(b::Basis, index::Array{Int}) = Bra(b, basis_vector(b.shape, index))
basis_bra(b::Basis, index::Int) = basis_bra(b, [index])
basis_ket(b::Basis, index::Array{Int}) = Ket(b, basis_vector(b.shape, index))
basis_ket(b::Basis, index::Int) = basis_ket(b, [index])

zero!(a::StateVector) = fill!(a.data, zero(eltype(a.data)))

function coherent_state(b::FockBasis, alpha)
    alpha = complex(alpha)
    x = zeros(Complex128, b.N1)
    x[1] = complex(1.)
    a = exp(-abs2(alpha)/2)
    for n=2:b.N1
        x[n] = x[n-1]*alpha/sqrt(n)
        x[n-1] *= a
    end
    x[end] *= a
    return Ket(b, x)
end

end