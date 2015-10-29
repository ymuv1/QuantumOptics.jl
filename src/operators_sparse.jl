module operators_sparse

import Base: ==, *, /, +, -
import ..operators

using ..bases, ..states, ..operators, ..sparsematrix

export SparseOperator, sparse_identity


"""
Sparse array implementation of AbstractOperator.

The matrix is stored as the julia built-in type SparseMatrixCSC
in the data field.
"""
type SparseOperator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::SparseMatrixCSC{Complex128}
end

SparseOperator(b::Basis, data::SparseMatrixCSC{Complex128}) = SparseOperator(b, b, data)
SparseOperator(b::Basis, data::Matrix{Complex128}) = SparseOperator(b, sparse(data))
SparseOperator(op::Operator) = SparseOperator(op.basis_l, op.basis_r, sparse(op.data))

SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, spzeros(Complex128, length(b1), length(b2)))
SparseOperator(b::Basis) = SparseOperator(b, b)

operators.full(a::SparseOperator) = Operator(a.basis_l, a.basis_r, full(a.data))

==(x::SparseOperator, y::SparseOperator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


*(a::SparseOperator, b::SparseOperator) = (check_multiplicable(a.basis_r, b.basis_l); SparseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::SparseOperator) = SparseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::SparseOperator, b::SparseOperator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? SparseOperator(a.basis_l, a.basis_r, a.data+b.data) : throw(IncompatibleBases()))

-(a::SparseOperator, b::SparseOperator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? SparseOperator(a.basis_l, a.basis_r, a.data-b.data) : throw(IncompatibleBases()))

# Fast in-place multiplication implementations
operators.gemm!{T<:Complex}(alpha::T, M::SparseOperator, b::Operator, beta::T, result::Operator) = sparsematrix.gemm!(alpha, M.data, b.data, beta, result.data)
operators.gemm!{T<:Complex}(alpha::T, a::Operator, M::SparseOperator, beta::T, result::Operator) = sparsematrix.gemm!(alpha, a.data, M.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, M::SparseOperator, b::Ket, beta::T, result::Ket) = sparsematrix.gemv!(alpha, M.data, b.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, b::Bra, M::SparseOperator, beta::T, result::Bra) = sparsematrix.gemv!(alpha, b.data, M.data, beta, result.data)


operators.tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))

operators.dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')

operators.norm(op::SparseOperator, p) = norm(op.data, p)
operators.trace(op::SparseOperator) = trace(op.data)

sparse_identity(b::Basis) = SparseOperator(b, b, speye(Complex128, length(b)))
sparse_identity(b1::Basis, b2::Basis) = SparseOperator(b1, b2, speye(Complex128, length(b1), length(b2)))
Base.identity(op::SparseOperator) = sparse_identity(op.basis_l, op.basis_r)


function operators.embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{SparseOperator})
    @assert length(indices) == length(operators)
    @assert length(basis.bases) >= maximum(indices)
    if length(basis.bases) == 1
        return SparseOperator(basis, basis, deepcopy(operators[1].data))
    end
    op_total = (1 in indices ? operators[findfirst(indices, 1)] : sparse_identity(basis.bases[1]))
    for i=2:length(basis.bases)
        op = (i in indices ? operators[findfirst(indices, i)] : sparse_identity(basis.bases[i]))
        op_total = tensor(op_total, op)
    end
    return op_total
end


end # module
