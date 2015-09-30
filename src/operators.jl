module operators

import Base: *, /, +, -
import ..states

using Base.LinAlg.BLAS
using Base.Cartesian

using ..bases, ..states

export AbstractOperator, Operator,
       tensor, dagger, expect, embed, normalize, normalize!


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
basis_l and basis_r defining the left and right side bases.

For fast time evolution also at least the function
gemv!(alpha, op::AbstractOperator, x::Ket, beta, result::Ket) should begin
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract AbstractOperator

"""
Dense array implementation of AbstractOperator.

The matrix consisting of complex floats is stored in the data field.
"""
type Operator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex{Float64}}
    Operator(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

Operator(b::Basis, data) = Operator(b, b, data)
Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(Complex, length(b1), length(b2)))
Operator(b::Basis) = Operator(b, b)
Operator(op::AbstractOperator) = full(op)

"""
Converting an arbitrary AbstractOperator into an Operator.
"""
Base.full(x::Operator) = deepcopy(x)
Base.full(op::AbstractOperator) = op*identity(op.basis_r)

Base.eltype(x::AbstractOperator) = Complex128
Base.zero{T<:AbstractOperator}(x::T) = T(x.basis_l, x.basis_r)


# Arithmetic operations for dense Operators
check_samebases(a::AbstractOperator, b::AbstractOperator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)

*(a::Operator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::Operator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::Operator, b::Operator) = (check_multiplicable(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::Operator, b::Operator) = (check_samebases(a,b); Operator(a.basis_l, a.basis_r, a.data+b.data))

-(a::Operator) = Operator(a.basis_l, a.basis_r, -a.data)
-(a::Operator, b::Operator) = (check_samebases(a,b); Operator(a.basis_l, a.basis_r, a.data-b.data))


# Fast in-place multiplication implementations
gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::Matrix{T}, beta::T, result::Matrix{T}) = BLAS.gemm!('N', 'N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Matrix{T}, b::Vector{T}, beta::T, result::Vector{T}) = BLAS.gemv!('N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Vector{T}, b::Matrix{T}, beta::T, result::Vector{T}) = BLAS.gemv!('T', alpha, b, a, beta, result)

gemm!(alpha, a::Operator, b::Operator, beta, result::Operator) = gemm!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::Operator, b::Ket, beta, result::Ket) = gemv!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::Bra, b::Operator, beta, result::Bra) = gemv!(alpha, a.data, b.data, beta, result.data)


"""
Tensor product of operators.
"""
states.tensor(a::Operator, b::Operator) = Operator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))
states.tensor(ops::Operator...) = reduce(tensor, ops)

"""
Tensor product of a ket and a bra results in an operator.
"""
states.tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))

"""
Hermitian conjugate of the given operator.
"""
states.dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')


"""
p-norm of given operator.
"""
Base.norm(op::Operator, p) = norm(op.data, p)

"""
Trace of given operator.
"""
Base.trace(op::Operator) = trace(op.data)

"""
Normalized copy of given operator (trace is 1.).
"""
states.normalize(op::Operator) = op/trace(op)

"""
Normalize the given operator.
"""
function states.normalize!(op::Operator)
    u = 1./trace(op)
    for j=1:size(op.data,2), i=1:size(op.data,1)
        op.data[i,j] *= u
    end
    return op
end


"""
Expectation value of the given operator for the specified state(s).
"""
expect(op::AbstractOperator, state::Operator) = trace(op*state)
expect(op::AbstractOperator, states::Vector{Operator}) = [expect(op, state) for state=states]
expect(op::AbstractOperator, state::Ket) = dagger(state)*(op*state)
expect(op::AbstractOperator, states::Vector{Ket}) = [expect(op, state) for state=states]

"""
Identity operator.
"""
identity(b::Basis) = Operator(b, b, eye(Complex, length(b)))
identity(b1::Basis, b2::Basis) = Operator(b1, b2, eye(Complex, length(b1), length(b2)))


# Multiplication for AbstractOperators in terms of their gemv! implementation
function gemm!(alpha, M::AbstractOperator, b::Operator, beta, result::Operator)
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        gemv!(alpha, M, bket, beta, resultket)
        result.data[:,i] = resultket.data
    end
end

function gemm!(alpha, b::Operator, M::AbstractOperator, beta, result::Operator)
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        gemv!(alpha, bbra, M, beta, resultbra)
        result.data[i,:] = resultbra.data
    end
end

function *(op1::AbstractOperator, op2::Operator)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = Operator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op1::Operator, op2::AbstractOperator)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = Operator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op::AbstractOperator, psi::Ket)
    check_multiplicable(op.basis_r, psi.basis)
    result = Ket(op.basis_l)
    gemv!(Complex(1.), op, psi, Complex(0.), result)
    return result
end

function *(psi::Bra, op::AbstractOperator)
    check_multiplicable(psi.basis, op.basis_l)
    result = Bra(op.basis_r)
    gemv!(Complex(1.), psi, op, Complex(0.), result)
    return result
end

Base.prod{B<:Basis, T<:AbstractArray}(basis::B, operators::T) = (length(operators)==0 ? identity(basis) : prod(operators))

"""
Tensor product of operators where all missing indices are identity operators.

Arguments
---------
    * basis: CompositeBasis of the resuting operator.
    * indices: Indices of the subsystems in which the given operators live.
    * operators: Operators defined in the subsystems.
"""
embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector) = tensor([prod(basis.bases[i], operators[find(indices.==i)]) for i=1:length(basis.bases)]...)
embed{T<:AbstractOperator}(basis::CompositeBasis, index::Int, op::T) = embed(basis, Int[index], T[op])


# Partial trace for dense operators.
function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[N] = 1
    for m=N-1:-1:1
        S[m] = S[m+1]*shape[m+1]
    end
    return S
end

@generated function _ptrace{RANK}(rank::Array{Int,RANK}, a::Matrix{Complex128},
                                  shape_l::Vector{Int}, shape_r::Vector{Int},
                                  indices::Vector{Int})
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = deepcopy(shape_l)
        result_shape_l[indices] = 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = deepcopy(shape_r)
        result_shape_r[indices] = 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(Complex128, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                #println("Jl_0: ", Jl_0, "; Jr_0: ", Jr_0, "; Il_0: ", Il_0, "; Ir_0: ", Ir_0)
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

"""
Partial trace of the given operator over the specified indices.
"""
function bases.ptrace(a::Operator, indices::Vector{Int})
    rank = zeros(Int, [0 for i=1:length(a.basis_l.shape)]...)
    result = _ptrace(rank, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

bases.ptrace(a::Operator, indices::Int) = bases.ptrace(a, Int[indices])

"""
Partial trace of the given state vector over the specified indices.
"""
bases.ptrace(a::Ket, indices) = bases.ptrace(tensor(a, dagger(a)), indices)
bases.ptrace(a::Bra, indices) = bases.ptrace(tensor(dagger(a), a), indices)


end # module
