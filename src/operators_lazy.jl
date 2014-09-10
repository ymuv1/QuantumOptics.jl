module operators_lazy

using Cartesian
using ..bases, ..states

importall ..operators
import ..operators.gemm!

export LazyTensor, LazySum

function complementary_indices(N::Int, indices::Vector{Int})
    dual_indices = zeros(Int, N-length(indices))
    count = 1
    for i=1:N
        if !(i in indices)
            dual_indices[count] = i
            count += 1
        end
    end
    return dual_indices
end

abstract LazyOperator <: AbstractOperator

type LazyTensor <: LazyOperator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    indices::Array{Int}
    operators::Vector{AbstractOperator}

    function LazyTensor{T<:AbstractOperator}(basis_l::CompositeBasis, basis_r::CompositeBasis, indices::Array{Int}, operators::Vector{T})
        @assert length(basis_l)==length(basis_r)
        @assert length(basis_l)>=length(indices)>0
        @assert length(indices)==length(operators)
        for (i,op) = zip(indices, operators)
            @assert op.basis_l==basis_l.bases[i] && op.basis_r==basis_r.bases[i]
        end
        compl_indices = complementary_indices(length(basis_l.bases), indices)
        for i=compl_indices
            check_multiplicable(basis_r.bases[i], basis_l.bases[i])
        end
        new(basis_l, basis_r, indices, operators)
    end
end

type LazySum <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    operators::Array{AbstractOperator}

    function LazySum{T<:AbstractOperator}(operators::Vector{T})
        @assert length(operators)>1
        op1 = operators[1]
        for i = 2:length(operators)
            @assert op1.basis_l == operators[i].basis_l
            @assert op1.basis_r == operators[i].basis_r
        end
        new(op1.basis_l, op1.basis_r, operators)
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
    return reduce(tensor, op_list)
end


@ngenerate N Nothing function mul_suboperator_l{N}(a::Array{Complex128, 2}, b::Array{Complex128, N}, index::Int, result::Array{Complex128, N})
    for m=1:size(a,2)
        @nloops N i d->(1:size(b,N+1-d)) d->(if d==index && i_d!=m continue end) begin
            #println("a")
            for n=1:size(a,1)
                @inbounds (@nref N result d->(N+1-d==index ? n : i_{N+1-d})) += a[n,m] * (@nref N b d->(i_{N+1-d}))
                #@inbounds (@nref N result d->(N+1-d==index ? n : i_{N+1-d})) += a[n,m] * (@nref N b d->(i_{N+1-d}))
            end
        end
    end
    return nothing
end

@ngenerate N Nothing function mul_suboperator_r{N}(a::Array{Complex128, N}, b::Array{Complex128, 2}, index::Int, result::Array{Complex128, N})
    for n=1:size(b,1)
        @nloops N i d->(1:size(a,N+1-d)) d->(if d==index && i_d!=n continue end) begin
            for m=1:size(b,2)
                @inbounds (@nref N result d->(N+1-d==index ? m : i_{N+1-d})) += (@nref N a d->(i_{N+1-d})) * b[n,m]
            end
        end
    end
    return nothing
end


@ngenerate RANK Nothing function mul_suboperator_l2{RANK}(uninvolved_rank::Array{Int,RANK},
                                    uninvolved_shape::Vector{Int}, uninvolved_strides::Vector{Int}, involved_stride::Int,
                                    a::Matrix{Complex128}, b::Vector{Complex128}, result::Vector{Complex128})
    M = size(a,1)
    @nexprs 1 (d->(I_{RANK}=1))
    @inbounds @nloops RANK i (d->1:uninvolved_shape[d]) (d->(I_{d-1}=I_d)) (d->(I_d+=uninvolved_strides[d])) begin
        I_n = I_0
        for n=1:M
            I_m = I_0 
            for m=1:M
                result[I_m] += a[m,n]*b[I_n]
                I_m += involved_stride
            end
            I_n += involved_stride
        end        
    end
    return nothing
end

@ngenerate RANK Nothing function mul_suboperator_r2{RANK}(uninvolved_rank::Array{Int,RANK},
                                    uninvolved_shape::Vector{Int}, uninvolved_strides::Vector{Int}, involved_stride::Int,
                                    a::Matrix{Complex128}, b::Vector{Complex128}, result::Vector{Complex128})
    M = size(a,1)
    @nexprs 1 (d->(I_{RANK}=1))
    @inbounds @nloops RANK i (d->1:uninvolved_shape[d]) (d->(I_{d-1}=I_d)) (d->(I_d+=uninvolved_strides[d])) begin
        I_n = I_0
        for n=1:M
            I_m = I_0 
            for m=1:M
                result[I_n] += a[m,n]*b[I_m]
                I_m += involved_stride
            end
            I_n += involved_stride
        end        
    end
    return nothing
end


function *(a::LazyTensor, b::Ket)
    check_multiplicable(a.basis_r, b.basis)
    b_shape = reverse(b.basis.shape)
    b_data = reshape(b.data, b_shape...)
    for (op, operator_index) = zip(a.operators, a.indices)
        # TODO: Doesnt work like this if basis_l != basis_r
        tmp1 = zeros(eltype(b_data), b_shape...)
        mul_suboperator_l(op.data, b_data, operator_index, tmp1)
        b_data = tmp1
    end
    return Ket(a.basis_l, vec(b_data))
end

function *(a::Bra, b::LazyTensor)
    check_multiplicable(a.basis, b.basis_l)
    a_shape = reverse(a.basis.shape)
    a_data = reshape(a.data, a_shape...)
    for (op, operator_index) = zip(b.operators, b.indices)
        # TODO: Doesnt work like this if basis_l != basis_r
        tmp1 = zeros(eltype(a_data), a_shape...)
        mul_suboperator_r(a_data, op.data, operator_index, tmp1)
        a_data = tmp1
        #mul_suboperator_l(op.data', a_data', operator_index, tmp1)
        #a_data = tmp1'
    end
    return Bra(b.basis_r, vec(a_data))
end

# function *(a::LazyTensor, b::Ket)
#     check_multiplicable(a.basis_r, b.basis)
#     for (op, operator_index) = zip(a.operators, a.indices)
#         b = mul_suboperator(op, b, operator_index)
#     end
#     return b
# end


function *(a::LazyTensor, b::Operator)
    check_multiplicable(a.basis_r, b.basis_l)
    x = Operator(a.basis_l, b.basis_r)
    for j=1:length(b.basis_r)
        b_j = Ket(b.basis_l, b.data[:,j])
        x.data[:,j] = (a*b_j).data
    end
    return x
end

function *(a::Operator, b::LazyTensor)
    check_multiplicable(a.basis_r, b.basis_l)
    x = Operator(a.basis_l, b.basis_r)
    for j=1:length(a.basis_l)
        a_j = Bra(a.basis_l, vec(a.data[j,:]))
        x.data[j,:] = (a_j*b).data
    end
    return x
end



function gemm!{T<:Complex}(alpha::T, a::LazyTensor, b::Ket, beta::T, result::Vector{T})
    check_multiplicable(a.basis_r, b.basis)
    b_shape = reverse(b.basis.shape)
    b_data = reshape(b.data, b_shape...)
    for (op, operator_index) = zip(a.operators, a.indices)
        # TODO: Doesnt work like this if basis_l != basis_r
        tmp1 = zeros(eltype(b_data), b_shape...)
        #gemm_suboperator!(alpha, op.data, b_data, operator_index, beta, tmp1)
        mul_suboperator_l(op.data, b_data, operator_index, tmp1)
        b_data = tmp1
    end
    result[:] = beta*result[:] + alpha*vec(b_data)[:]
    #return Ket(a.basis_l, vec(b_data))
end

function gemm_r!(b::LazyTensor, tmp1, tmp2)
    for (op, operator_index) = zip(b.operators, b.indices)
        mul_suboperator_r(tmp1, op.data, operator_index, tmp2)
        tmp1, tmp2 = tmp2, tmp1
    end
end

function addtoresult_l(j::Int, alpha::Complex128, beta::Complex128, tmp::Array{Complex128}, result::Matrix{Complex128})
    for i=1:size(result,1)
        result[i,j] = beta*result[i,j] + alpha*tmp[i]
    end
    return nothing
end

function setto_l(j::Int, tmp::Array{Complex128}, x::Matrix{Complex128})
    for i=1:size(x,1)
        tmp[i] = x[i,j]
    end
    return nothing
end

function strides(shape::Vector{Int}, indices::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[N] = 1
    for m=N-1:-1:1
        S[m] = S[m+1]*shape[m+1]
    end
    # S_ = [zeros(Int, N-1) for i=indices]
    # for i=1:length(indices)
    #     counter = 1
    #     for m=1:N
    #         if m!=indices[i]
    #             S_[i][counter] = S[m]
    #             counter += 1
    #         end
    #     end
    # end
    # return S, S_#[[S[1:i-1], S[i+1:end]] for i=indices]
    return S, [[S[1:i-1], S[i+1:end]] for i=indices]
end

function shapes(shape::Vector{Int}, indices::Vector{Int})
    return [[shape[1:i-1], shape[i+1:end]] for i=indices]
end

function applyoperators_l2{RANK}(uninvolved_rank::Array{Int,RANK}, uninvolved_shapes::Vector{Vector{Int}}, uninvolved_strides::Vector{Vector{Int}}, S::Vector{Int}, lazy_op::LazyTensor, tmp1::Vector{Complex128}, tmp2::Vector{Complex128})
    for m = 1:length(lazy_op.operators)
        fill!(tmp2, Complex128(0., 0.))
        op_index = lazy_op.indices[m]
        a = lazy_op.operators[m].data
        u_shape = uninvolved_shapes[m]
        u_strides = uninvolved_strides[m]
        mul_suboperator_l2(uninvolved_rank, u_shape, u_strides, S[op_index], a, tmp1, tmp2)
        tmp1, tmp2 = tmp2, tmp1
    end
    return nothing
end

function gemm!{T<:Complex}(alpha::T, lazy_op::LazyTensor, x::Matrix{T}, beta::T, result::Matrix{T})
    tmp1 = zeros(eltype(x), size(x,1))
    tmp2 = zeros(eltype(x), size(x,1))
    S, uninvolved_strides = strides(lazy_op.basis_r.shape, lazy_op.indices)
    uninvolved_shapes = shapes(lazy_op.basis_r.shape, lazy_op.indices)
    N = length(lazy_op.basis_r.shape)-1
    uninvolved_rank = zeros(Int,[0 for i=1:N]...)
    for j=1:size(x,2)
        setto_l(j, tmp1, x)
        applyoperators_l2(uninvolved_rank, uninvolved_shapes, uninvolved_strides, S, lazy_op, tmp1, tmp2)
        if length(lazy_op.operators)%2==1
            addtoresult_l(j, alpha, beta, tmp2, result)
        else
            addtoresult_l(j, alpha, beta, tmp1, result)
        end
    end
end

function applyoperators_l(lazy_op::LazyTensor, tmp1::Array{Complex128}, tmp2::Array{Complex128})
    for m = 1:length(lazy_op.operators)
        fill!(tmp2, Complex128(0., 0.))
        mul_suboperator_l(lazy_op.operators[m].data, tmp1, lazy_op.indices[m], tmp2)
        tmp1, tmp2 = tmp2, tmp1
    end
    return nothing
end

function gemm2!{T<:Complex}(alpha::T, lazy_op::LazyTensor, x::Matrix{T}, beta::T, result::Matrix{T})
    tmp1 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
    tmp2 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
    for j=1:size(x,2)
        setto_l(j, tmp1, x)
        applyoperators_l(lazy_op, tmp1, tmp2)
        if length(lazy_op.operators)%2==1
            addtoresult_l(j, alpha, beta, tmp2, result)
        else
            addtoresult_l(j, alpha, beta, tmp1, result)
        end
    end
end

function addtoresult_r(j::Int, alpha::Complex128, beta::Complex128, tmp::Array{Complex128}, result::Matrix{Complex128})
    for i=1:size(result,1)
        result[j,i] = beta*result[j,i] + alpha*tmp[i]
    end
    return nothing
end

function setto_r(j::Int, tmp::Array{Complex128}, x::Matrix{Complex128})
    for i=1:size(x,1)
        tmp[i] = x[j,i]
    end
    return nothing
end

function applyoperators_r(lazy_op::LazyTensor, tmp1::Array{Complex128}, tmp2::Array{Complex128})
    for m = 1:length(lazy_op.operators)
        fill!(tmp2, Complex128(0., 0.))
        mul_suboperator_r(tmp1, lazy_op.operators[m].data, lazy_op.indices[m], tmp2)
        tmp1, tmp2 = tmp2, tmp1
    end
    return nothing
end

function applyoperators_r2(uninvolved_rank, uninvolved_shapes, uninvolved_strides, S, lazy_op::LazyTensor, tmp1::Vector{Complex128}, tmp2::Vector{Complex128})
    for m = 1:length(lazy_op.operators)
        fill!(tmp2, Complex128(0., 0.))
        op_index = lazy_op.indices[m]
        a = lazy_op.operators[m].data
        u_shape = uninvolved_shapes[m]
        u_strides = uninvolved_strides[m]
        mul_suboperator_r2(uninvolved_rank, u_shape, u_strides, S[op_index], a, tmp1, tmp2)
        tmp1, tmp2 = tmp2, tmp1
    end
    return nothing
end

function gemm!{T<:Complex}(alpha::T, x::Matrix{T}, lazy_op::LazyTensor, beta::T, result::Matrix{T})
    tmp1 = zeros(eltype(x), size(x,1))
    tmp2 = zeros(eltype(x), size(x,1))
    S, uninvolved_strides = strides(lazy_op.basis_r.shape, lazy_op.indices)
    uninvolved_shapes = shapes(lazy_op.basis_r.shape, lazy_op.indices)
    N = length(lazy_op.basis_r.shape)-1
    uninvolved_rank = zeros(Int,[0 for i=1:N]...)
    for j=1:size(x,2)
        setto_r(j, tmp1, x)
        applyoperators_r2(uninvolved_rank, uninvolved_shapes, uninvolved_strides, S, lazy_op, tmp1, tmp2)
        if length(lazy_op.operators)%2==1
            addtoresult_r(j, alpha, beta, tmp2, result)
        else
            addtoresult_r(j, alpha, beta, tmp1, result)
        end
    end
end

function gemm2!{T<:Complex}(alpha::T, x::Matrix{T}, lazy_op::LazyTensor, beta::T, result::Matrix{T})
    tmp1 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
    tmp2 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
    for j=1:size(x,2)
        setto_r(j, tmp1, x)
        applyoperators_r(lazy_op, tmp1, tmp2)
        if length(lazy_op.operators)%2==1
            addtoresult_r(j, alpha, beta, tmp2, result)
        else
            addtoresult_r(j, alpha, beta, tmp1, result)
        end
    end
end

# function gemm!{T<:Complex}(alpha::T, x::Matrix{T}, lazy_op::LazyTensor, beta::T, result::Matrix{T})
# #    result[:,:] = alpha*(a*Operator(a.basis_l, b)).data[:,:] + beta*result[:,:]
#     tmp1 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
#     tmp2 = zeros(eltype(x), reverse(lazy_op.basis_l.shape)...)
#     for j=1:size(x,1)
#         for i=1:size(x,2)
#             tmp1[i] = x[j,i]
#         end
#         for (op, operator_index) = zip(lazy_op.operators, lazy_op.indices)
#             for i=1:size(x,2)
#                 tmp2[i] = 0
#             end
#             mul_suboperator_r(tmp1, op.data, operator_index, tmp2)
#             tmp1, tmp2 = tmp2, tmp1
#         end
#         for i=1:size(x,2)
#             result[j,i] = beta*result[j,i] + alpha*tmp1[i]
#         end
#     end
# end

function gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::LazySum, beta::T, result::Matrix{T})
    #result[:,:] = alpha*(Operator(b.basis_l,a)*b).data[:,:] + beta*result[:,:]
    tmp = zeros(T, size(a)...)
    result[:] = beta*result[:]
    for op=b.operators
        gemm!(complex(1.), a, op, complex(0.), tmp)
        result[:] = result[:] + alpha*tmp[:]
    end
end

function gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::LazySum, beta::T, result::Matrix{T})
    firstrun = true
    for op=b.operators
        if firstrun
            gemm!(alpha, a, op, beta, result)
            firstrun = false
        else
            gemm!(alpha, a, op, Complex(1.), result)
        end
    end
end

# function gemm!{T<:Complex}(alpha::T, a::LazySum, b::Matrix{T}, beta::T, result::Matrix{T})
#     #result[:,:] = alpha*(a*Operator(a.basis_l, b)).data[:,:] + beta*result[:,:]
#     tmp = zeros(T, size(b)...)
#     result[:] = beta*result[:]
#     for op=a.operators
#         gemm!(complex(1.), op, b, complex(0.), tmp)
#         result[:] = result[:] + alpha*tmp[:]
#     end
# end

function gemm!{T<:Complex}(alpha::T, a::LazySum, b::Matrix{T}, beta::T, result::Matrix{T})
    firstrun = true
    for op=a.operators
        if firstrun
            gemm!(alpha, op, b, beta, result)
            firstrun = false
        else
            gemm!(alpha, op, b, Complex(1.), result)
        end
    end
end

*(a::LazySum, b::LazySum) = error()
*(a::LazySum, b) = sum([op*b for op=a.operators])
*(a, b::LazySum) = sum([a*op for op=b.operators])

+(a::LazyOperator, b::LazyOperator) = LazySum([a,b])
+(a::LazySum, b::LazySum) = LazySum([a.operators, b.operators])
+(a::LazySum, b::LazyOperator) = LazySum([a.operators, b])
+(a::LazyOperator, b::LazySum) = LazySum([a, b.operators])

+(a::LazySum, b::AbstractOperator) = LazySum([a.operators, b])
+(a::AbstractOperator, b::LazySum) = LazySum([a, b.operators])

end