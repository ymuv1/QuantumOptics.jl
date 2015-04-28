module operators_sparse

using ..bases, ..states, ..sparsematrix

importall ..operators

export SparseOperator, sparse_identity


type SparseOperator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::sparsematrix.SparseMatrix{Complex128}
end

SparseOperator(b::Basis, data::sparsematrix.SparseMatrix) = SparseOperator(b, b, data)
SparseOperator(b::Basis, data::Matrix) = SparseOperator(b, sparsematrix.SparseMatrix(data))
SparseOperator(Operator) = SparseOperator(Operator.basis_l, Operator.basis_r, sparsematrix.SparseMatrix(Operator.data))

SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, sparsematrix.SparseMatrix([prod(b1.shape), prod(b2.shape)], Int[], Int[], Complex128[]))
SparseOperator(b::Basis) = SparseOperator(b, b)

Operator(a::SparseOperator) = Operator(a.basis_l, a.basis_r, full(a.data))

# *(a::SparseOperator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
# *(a::Bra, b::SparseOperator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::SparseOperator, b::SparseOperator) = (check_multiplicable(a.basis_r, b.basis_l); SparseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Operator) = (check_multiplicable(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::Operator, b::SparseOperator) = (check_multiplicable(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::SparseOperator) = SparseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::SparseOperator, b::SparseOperator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? SparseOperator(a.basis_l, a.basis_r, a.data+b.data) : throw(IncompatibleBases()))
-(a::SparseOperator, b::SparseOperator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? SparseOperator(a.basis_l, a.basis_r, a.data-b.data) : throw(IncompatibleBases()))


tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))

dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')
Base.full(x::SparseOperator) = Operator(x.basis_l, x.basis_r, full(x.data))

# Base.norm(op::SparseOperator, p) = norm(op.data, p)
# Base.trace(op::SparseOperator) = trace(op.data)

sparse_identity(b::Basis) = SparseOperator(b, b, sparse_eye(Complex128, length(b)))
sparse_identity(b1::Basis, b2::Basis) = SparseOperator(b1, b2, sparse_eye(Complex128, length(b1), length(b2)))
# number(b::Basis) = Operator(b, b, diagm(map(Complex, 0:(length(b)-1))))
# destroy(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),1))
# create(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),-1))

# const spinbasis = GenericBasis([2])
# const sigmax = Operator(spinbasis, [0 1;1 0])
# const sigmay = Operator(spinbasis, [0 -1im;1im 0])
# const sigmaz = Operator(spinbasis, [1 0;0 -1])
# const sigmap = Operator(spinbasis, [0 0;1 0])
# const sigmam = Operator(spinbasis, [0 1;0 0])

# check_equal_bases(a::AbstractOperator, b::AbstractOperator) = (check_equal(a.basis_l,b.basis_l); check_equal(a.basis_r,b.basis_r))

# set!(a::Operator, b::Operator) = (check_equal_bases(a, b); set!(a.data, b.data); a)
# zero!(a::Operator) = fill!(a.data, zero(eltype(a.data)))

function operators.gemm!{T<:Complex}(alpha::T, M::SparseOperator, b::Operator, beta::T, result::Operator)
    sparsematrix.gemm!(alpha, M.data, b.data, beta, result.data)
end
function operators.gemm!{T<:Complex}(alpha::T, a::Operator, M::SparseOperator, beta::T, result::Operator)
    sparsematrix.gemm!(alpha, a.data, M.data, beta, result.data)
end
function operators.gemv!{T<:Complex}(alpha::T, M::SparseOperator, b::Ket, beta::T, result::Ket)
    sparsematrix.gemv!(alpha, M.data, b.data, beta, result.data)
end

function embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{SparseOperator})
    op_total = (1 in indices ? operators[findfirst(indices, 1)] : sparse_identity(basis.bases[1]))
    for i=2:length(basis.bases)
        op = (i in indices ? operators[findfirst(indices, i)] : sparse_identity(basis.bases[i]))
        op_total = tensor(op_total, op)
    end
    return op_total
end

end # module
