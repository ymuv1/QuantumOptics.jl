module states

import Base: ==, +, -, *, /

using ..bases

export StateVector, Bra, Ket,
       tensor, dagger, ⊗,
       basis_bra, basis_ket, basis,
       normalize, normalize!


"""
Abstract base class for Bra and Ket.
"""
abstract StateVector

type Bra <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
    Bra(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

type Ket <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
    Ket(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

Bra(b::Basis) = Bra(b, zeros(Complex, length(b)))
Ket(b::Basis) = Ket(b, zeros(Complex, length(b)))

Base.eltype(x::StateVector) = Complex128
Base.zero{T<:StateVector}(x::T) = T(x.basis)

=={T<:StateVector}(x::T, y::T) = (x.basis == y.basis) && (x.data == y.data)


# Arithmetic operations
*(a::Bra, b::Ket) = (check_multiplicable(a.basis, b.basis); sum(a.data.*b.data))
*{T<:StateVector}(a::Number, b::T) = T(b.basis, complex(a)*b.data)
*{T<:StateVector}(a::T, b::Number) = T(a.basis, complex(b)*a.data)

/{T<:StateVector}(a::T, b::Number) = T(a.basis, a.data/complex(b))

+{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data+b.data) : throw(IncompatibleBases()))

-{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data-b.data) : throw(IncompatibleBases()))

"""
Tensor product of given bras or kets.
"""
tensor{T<:StateVector}(a::T, b::T) = T(compose(a.basis, b.basis), kron(a.data, b.data))
tensor{T<:StateVector}(states::T...) = reduce(tensor, states)
tensor() = error("Tensor function needs at least one argument.")
⊗(a,b) = tensor(a,b)

"""
Hermitian conjugate of given state vector.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))


# Normalization functions
"""
Norm of given state vector.
"""
Base.norm(x::StateVector, p=2) = norm(x.data, p)

"""
Normalized copy of the given state vector.
"""
normalize(x::StateVector, p=2) = x/norm(x, p)

"""
Normalize the given state vector.
"""
function normalize!(x::StateVector, p=2)
    u = 1./norm(x, p)
    for i=1:length(x.data)
        x.data[i]*=u
    end
    return x
end


# Creation of basis states.
function basis_vector(shape::Vector{Int}, indices::Vector{Int})
    x = zeros(Complex, shape...)
    x[indices] = Complex(1.)
    reshape(x, prod(shape))
end

basis_bra(b::Basis, indices::Array{Int}) = Bra(b, basis_vector(b.shape, indices))
basis_bra(b::Basis, index::Int) = basis_bra(b, [index])
basis_ket(b::Basis, indices::Array{Int}) = Ket(b, basis_vector(b.shape, indices))
basis_ket(b::Basis, index::Int) = basis_ket(b, [index])


end # module
