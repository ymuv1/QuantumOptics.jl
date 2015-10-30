module superoperators

import Base: ==, *, /, +, -, expm
import ..operators.check_samebases

using ..bases, ..operators, ..operators_sparse

export SuperOperator, SparseSuperOperator, spre, spost, liouvillian


abstract AbstractSuperOperator

type SuperOperator <: AbstractSuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::Matrix{Complex128}
    function SuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::Matrix{Complex128})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) || length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

type SparseSuperOperator <: AbstractSuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::SparseMatrixCSC{Complex128}
    function SparseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::SparseMatrixCSC{Complex128})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) || length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

=={T<:AbstractSuperOperator}(a::T, b::T) = (a.basis_l == b.basis_l) && (a.basis_r == b.basis_r) && (a.data == b.data)

operators.check_samebases(a::AbstractSuperOperator, b::AbstractSuperOperator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)

function *{T<:AbstractSuperOperator}(a::T, b::Operator)
    if a.basis_r[1] != b.basis_l || a.basis_r[2] != b.basis_r
        throw(DimensionMismatch())
    end
    data = a.data*reshape(b.data, length(b.data))
    return Operator(a.basis_l[1], a.basis_l[2], reshape(data, length(a.basis_l[1]), length(a.basis_l[2])))
end

function *{T<:AbstractSuperOperator}(a::T, b::T)
    if a.basis_r != b.basis_l
        throw(DimensionMismatch())
    end
    return T(a.basis_l, b.basis_r, a.data*b.data)
end

/{T<:AbstractSuperOperator}(a::T, b::Number) = T(a.basis_l, a.basis_r, a.data/complex(b))

+{T<:AbstractSuperOperator}(a::T, b::T) = (operators.check_samebases(a, b); T(a.basis_l, a.basis_r, a.data+b.data))

-{T<:AbstractSuperOperator}(a::T, b::T) = (operators.check_samebases(a, b); T(a.basis_l, a.basis_r, a.data-b.data))
-{T<:AbstractSuperOperator}(a::T) = T(a.basis_l, a.basis_r, -a.data)


spre(op::Operator) = SuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), tensor(identity(op), op).data)
spre(op::SparseOperator) = SparseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), tensor(identity(op), op).data)

spost(op::Operator) = SuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), kron(transpose(op.data), identity(op).data))
spost(op::SparseOperator) = SparseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r),  kron(transpose(op.data), identity(op).data))


function _check_input(H::AbstractOperator, J::Vector, Jdagger::Vector, Gamma::Union{Vector{Float64}, Matrix{Float64}})
    for j=J
        @assert typeof(j) <: AbstractOperator
        operators.check_samebases(H, j)
    end
    for j=Jdagger
        @assert typeof(j) <: AbstractOperator
        operators.check_samebases(H, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(Gamma) == Matrix{Float64}
        @assert size(Gamma, 1) == size(Gamma, 2) == length(J)
    elseif typeof(Gamma) == Vector{Float64}
        @assert length(Gamma) == length(J)
    else
        error()
    end
end

function liouvillian{T<:AbstractOperator}(H::T, J::Vector{T};
            Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
            Jdagger::Vector{T}=map(dagger, J))
    _check_input(H, J, Jdagger, Gamma)
    L = spre(-1im*H) + spost(1im*H)
    if typeof(Gamma) == Matrix{Float64}
        for i=1:length(J), j=1:length(J)
            jdagger_j = Gamma[i,j]/2*Jdagger[j]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(Gamma[i,j]*J[i]) * spost(Jdagger[j])
        end
    elseif typeof(Gamma) == Vector{Float64}
        for i=1:length(J)
            jdagger_j = Gamma[i]/2*Jdagger[i]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(Gamma[i]*J[i]) * spost(Jdagger[i])
        end
    else
        error()
    end
    return L
end

Base.expm{T<:AbstractSuperOperator}(op::T) = T(op.basis_l, op.basis_r, expm(op.data))

end # module
