module operators_lazy

using Base.Cartesian
using ..bases, ..states

importall ..operators

export LazyTensor, LazySum


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
*(a::LazySum, b) = LazySum([op*b for op=a.operators]...)
*(a, b::LazySum) = LazySum([a*op for op=b.operators]...)

+(a::LazyOperator, b::LazyOperator) = LazySum([a,b]...)
+(a::LazySum, b::LazySum) = LazySum([a.operators, b.operators]...)
+(a::LazySum, b::LazyOperator) = LazySum([a.operators, b]...)
+(a::LazyOperator, b::LazySum) = LazySum([a, b.operators]...)

+(a::LazySum, b::AbstractOperator) = LazySum([a.operators, b]...)
+(a::AbstractOperator, b::LazySum) = LazySum([a, b.operators]...)

-(a::LazyTensor) = LazyTensor(a.basis_l, a.basis_r, a.operators, -a.factor)
-(a::LazySum) = LazySum([-op for op=a.operators]...)
-(a::LazyOperator, b::LazyOperator) = LazySum([a,-b]...)
-(a::LazySum, b::LazySum) = LazySum([a.operators, (-b).operators]...)
-(a::LazySum, b::LazyOperator) = LazySum([a.operators, -b]...)
-(a::LazyOperator, b::LazySum) = LazySum([a, (-b).operators]...)

-(a::LazySum, b::AbstractOperator) = LazySum([a.operators, -b]...)
-(a::AbstractOperator, b::LazySum) = LazySum([-a, b.operators]...)

end
