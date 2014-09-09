module operators_sparse

using ..bases, ..states
using Base.LinAlg.BLAS

importall ..operators

export SparseOperator


type SparseOperator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::SparseMatrixCSC{Complex128}
end

SparseOperator(b::Basis, data) = SparseOperator(b, b, data)
SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, spzeros(Complex128, length(b1), length(b2)))
SparseOperator(b::Basis) = SparseOperator(b, b)


*(a::SparseOperator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::SparseOperator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
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
#Base.full(x::Operator) = x

Base.norm(op::SparseOperator, p) = norm(op.data, p)
Base.trace(op::SparseOperator) = trace(op.data)

spidentity(b::Basis) = SparseOperator(b, b, speye(Complex128, length(b)))
spidentity(b1::Basis, b2::Basis) = Operator(b1, b2, eye(Complex128, length(b1), length(b2)))
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

function operators.gemm!{T<:Complex}(alpha::T, A::SparseOperator, b::Matrix{T}, beta::T, result::Matrix{T})
    
    a = A.data
    if abs(beta)<1e-14
        fill!(result, zero(T))
    end
    nzv = a.nzval
    rv = a.rowval
    colptr = a.colptr
    for multivec_col=1:size(b,2)::Int, col=1:a.n::Int
        Xc = b[col::Int, multivec_col::Int]
        @inbounds for k = colptr[col::Int]::Int : (colptr[col::Int+1]::Int-1)
           result[rv[k::Int]::Int, multivec_col::Int]::Complex128 += alpha::Complex128 * nzv[k::Int]::Complex128 * Xc::Complex128
        end
    end
    return nothing
end
function operators.gemm!{T<:Complex}(alpha::T, X::Matrix{T}, a::SparseOperator, beta::T, result::Matrix{T})
    A = a.data
    if abs(beta)<1e-14
        fill!(result, zero(T))
    end
    @inbounds for multivec_row=1:size(X,1), col=1:A.n::Int, k=A.colptr[col::Int]::Int:(A.colptr[col::Int+1]::Int-1)
        result[multivec_row::Int, col::Int] += alpha::Complex128 * X[multivec_row::Int, A.rowval[k::Int]::Int]::Complex128 * A.nzval[k::Int]::Complex128
    end
    return nothing
end
#gemm!{T<:Complex}(alpha::T, a::SparseOperator, b::Matrix{T}, beta::T, result::Matrix{T}) = gemm!(alpha, a.data, b, beta, result)
#gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::SparseOperator, beta::T, result::Matrix{T}) = gemm!(alpha, a, b.data, beta, result)


end