module operators_lazy

using Base.Cartesian
using ..bases, ..states

importall ..operators

export LazyOperator, LazyTensor, LazySum, LazyProduct


abstract LazyOperator <: AbstractOperator

type LazyTensor <: LazyOperator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::Dict{Int,AbstractOperator}

    function LazyTensor(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int,AbstractOperator}, factor::Number=1.)
        N = length(basis_l)
        @assert N==length(basis_r)
        @assert maximum(keys(operators))<=N
        @assert 1<=minimum(keys(operators))
        for (i,op) = operators
            @assert op.basis_l==basis_l.bases[i] && op.basis_r==basis_r.bases[i]
        end
        new(basis_l, basis_r, complex(factor), operators)
    end
end

function LazyTensor{T<:AbstractOperator}(basis_l::CompositeBasis, basis_r::CompositeBasis, indices::Vector{Int}, operators::Vector{T})
    @assert length(indices) == length(Set(indices))
    LazyTensor(basis_l, basis_r, Dict{Int,AbstractOperator}([i=>op for (i,op)=zip(indices, operators)]))
end

LazyTensor(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Int, operator::AbstractOperator) = LazyTensor(basis_l, basis_r, Dict{Int,AbstractOperator}(index=>operator))


type LazySum <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    operators::Vector{AbstractOperator}

    function LazySum(operators::AbstractOperator...)
        @assert length(operators)>1
        for i = 2:length(operators)
            @assert operators[1].basis_l == operators[i].basis_l
            @assert operators[1].basis_r == operators[i].basis_r
        end
        new(operators[1].basis_l, operators[1].basis_r, AbstractOperator[operators...])
    end
end


type LazyProduct <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    factor::Complex128
    operators::Vector{AbstractOperator}
    function LazyProduct(factor::Complex128, operators::AbstractOperator...)
        @assert length(operators)>1
        for i = 2:length(operators)
            @assert multiplicable(operators[i-1].basis_r, operators[i].basis_l)
        end
        new(operators[1].basis_l, operators[end].basis_r, factor, AbstractOperator[operators...])
    end
end
LazyProduct(operators::AbstractOperator...) = LazyProduct(Complex(1.), operators...)

function Base.full(x::LazyTensor)
    op_list = Operator[]
    for i=1:length(x.basis_l.bases)
        if i in x.indices
            push!(op_list, full(x.operators[first(find(x.indices.==i))]))
        else
            push!(op_list, identity(x.basis_l.bases[i], x.basis_r.bases[i]))
        end
    end
    return x.factor*reduce(tensor, op_list)
end


Base.trace(op::LazyTensor) = op.factor*prod([(haskey(op.operators,i) ? trace(op.operators[i]): prod(op.basis_l.shape)) for i=1:length(op.basis_l)])
Base.trace(op::LazySum) = sum([trace(x) for x=op.operators])

function *(a::LazyTensor, b::LazyTensor)
    check_multiplicable(a.basis_r, b.basis_l)
    a_indices = keys(a.operators)
    b_indices = keys(b.operators)
    D = Dict{Int, AbstractOperator}()
    for i=intersect(a_indices, b_indices)
        D[i] = a.operators[i]*b.operators[i]
    end
    for i=setdiff(a_indices, b_indices)
        D[i] = a.operators[i]
    end
    for i=setdiff(b_indices, a_indices)
        D[i] = b.operators[i]
    end
    return LazyTensor(a.basis_l, b.basis_r, D, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a.basis_l, a.basis_r, a.operators, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b.basis_l, b.basis_r, b.operators, a*b.factor)

*(a::LazySum, b::LazySum) = LazySum([op1*op2 for op1=a.operators, op2=b.operators]...)
*(a::LazySum, b::Operator) = sum([op*b for op=a.operators])
*(a::Operator, b::LazySum) = sum([a*op for op=b.operators])
*(a::LazySum, b::Ket) = sum([op*b for op=a.operators])
*(a::Bra, b::LazySum) = sum([a*op for op=b.operators])
*(a::LazyProduct, b::LazyProduct) = LazyProduct(a.factor*b.factor, a.operators..., b.operators...)
*(a::LazyProduct, b::Ket) = a.factor*foldr(*, b, a.operators)
*(a::Bra, b::LazyProduct) = b.factor*foldl(*, a, b.operators)
*(a::LazyProduct, b::Number) = LazyProduct(a.factor*b, a.operators...)
*(a::Number, b::LazyProduct) = LazyProduct(a*b.factor, b.operators...)
*(a::LazySum, b) = LazySum([op*b for op=a.operators]...)
*(a, b::LazySum) = LazySum([a*op for op=b.operators]...)


+(a::LazyOperator, b::LazyOperator) = LazySum([a,b]...)
+(a::LazyOperator, b::AbstractOperator) = LazySum(a, b)
+(a::AbstractOperator, b::LazyOperator) = LazySum(a, b)
+(a::LazySum, b::LazySum) = LazySum([a.operators, b.operators]...)
+(a::LazySum, b::LazyOperator) = LazySum([a.operators, b]...)
+(a::LazyOperator, b::LazySum) = LazySum([a, b.operators]...)
+(a::LazySum, b::AbstractOperator) = LazySum([a.operators, b]...)
+(a::AbstractOperator, b::LazySum) = LazySum([a, b.operators]...)

# +(a::LazySum, b::AbstractOperator) = LazySum([a.operators, b]...)
# +(a::AbstractOperator, b::LazySum) = LazySum([a, b.operators]...)

-(a::LazyTensor) = LazyTensor(a.basis_l, a.basis_r, a.operators, -a.factor)
-(a::LazySum) = LazySum([-op for op=a.operators]...)

-(a::LazyOperator, b::LazyOperator) = LazySum(a, -b)
-(a::LazyOperator, b::AbstractOperator) = LazySum(a, -b)
-(a::AbstractOperator, b::LazyOperator) = LazySum(a, -b)
-(a::LazySum, b::LazySum) = LazySum([a.operators, (-b).operators]...)
-(a::LazySum, b::LazyOperator) = LazySum([a.operators, -b]...)
-(a::LazyOperator, b::LazySum) = LazySum([a, (-b).operators]...)
-(a::LazySum, b::AbstractOperator) = LazySum([a.operators, -b]...)
-(a::AbstractOperator, b::LazySum) = LazySum([a, (-b).operators]...)

# @generated function _lazytensor_gemv!{RANK}(rank::Array{Int, RANK}, alpha,
#                                             op::AbstractOperator, index::Int,
#                                             b::Ket, beta, result::Ket)
#     N0 = index-1
#     N1 = RANK-index
#     return quote
#         @nloops $N0
#     end
# end
#
# function gemv!(alpha, a::LazyTensor, b::Ket, beta, result::Ket)
#     gemv!(alpha, a.operators[1], b, beta, result)
#     for i=2:length(a.operators)
#         gemv!(alpha, a.operators[i], b, Complex(1.), result)
#     end
# end

function operators.gemv!(alpha, a::LazySum, b::Ket, beta, result::Ket)
    operators.gemv!(alpha, a.operators[1], b, beta, result)
    for i=2:length(a.operators)
        operators.gemv!(alpha, a.operators[i], b, Complex(1.), result)
    end
end

function operators.gemv!(alpha, a::Bra, b::LazySum, beta, result::Bra)
    operators.gemv!(alpha, a, b.operators[1], beta, result)
    for i=2:length(b.operators)
        operators.gemv!(alpha, a, b.operators[i], Complex(1.), result)
    end
end

function operators.gemv!(alpha, a::LazyProduct, b::Ket, beta, result::Ket)
    tmp1 = Ket(a.operators[end].basis_l)
    operators.gemv!(Complex(1.), a.operators[end], b, Complex(0.), tmp1)
    for i=length(a.operators)-1:-1:2
        tmp2 = Ket(a.operators[i].basis_l)
        operators.gemv!(Complex(1.), a.operators[i], tmp1, Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, a.operators[1], tmp1, beta, result)
end
# function operators.gemv!(alpha, a::LazyProduct, b::Ket, beta, result::Ket)
#     tmp = Ket(b.basis)
#     operators.gemv!(alpha, a.operators[end], b, Complex(0.), result)
#     for i=length(a.operators)-1:-1:1
#         result, tmp = tmp, result
#         operators.gemv!(Complex(1.), a.operators[i], tmp, Complex(0.), result)
#     end
#     tmp.data[:] = result.data[:]
#     nothing
# end


function operators.gemv!(alpha, a::Bra, b::LazyProduct, beta, result::Bra)
    tmp1 = Bra(b.operators[1].basis_r)
    operators.gemv!(Complex(1.), a, b.operators[1], Complex(0.), tmp1)
    for i=2:length(b.operators)-1
        tmp2 = Bra(b.operators[i].basis_r)
        operators.gemv!(Complex(1.), tmp1, b.operators[i], Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, tmp1, b.operators[end], beta, result)
end

end
