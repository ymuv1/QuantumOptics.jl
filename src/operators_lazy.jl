module operators_lazy

using Iterators
using ..bases, ..states

importall ..operators

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

type LazyTensor <: AbstractOperator
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


function mul_suboperator(a::AbstractOperator, b::Ket, index::Int)
    @assert(issubtype(typeof(b.basis), CompositeBasis))
    N = length(b.basis.bases)
    @assert(0<index<=N)
    @assert(bases.multiplicable(a.basis_r, b.basis.bases[index]))
    uninvolved_indices = [1:index-1, index+1:N]
    uninvolved_shapes = b.basis.shape[uninvolved_indices]
    # println("operator: ", a.data)
    # println("|b>: ", b)
    # println("uninvolved_indices: ", uninvolved_indices)
    # println("uninvolved shapes: ", uninvolved_shapes)
    b_data = reshape(b.data, reverse(b.basis.shape)...)
    result_basis = CompositeBasis([b.basis.bases[1:index-1], a.basis_l, b.basis.bases[index+1:end]]...)
    result = Ket(result_basis)
    result_data = reshape(result.data, reverse(result.basis.shape)...)
    broad_index_result = {(1:n) for n=result.basis.shape}
    broad_index_b = {(1:n) for n=b.basis.shape}
    for uninvolved_index in Iterators.product([1:n for n = uninvolved_shapes]...)
        # println("uninvolved index: ", uninvolved_index)
        broad_index_result[uninvolved_indices] = [uninvolved_index...]
        broad_index_b[uninvolved_indices] = [uninvolved_index...]
        # println("broad_index_result: ", broad_index_result)
        # println("broad_index_b: ", broad_index_b)
        b_splice = Ket(b.basis.bases[index], vec(b_data[reverse(broad_index_b)...]))
        # println("bsplice: ", b_splice)
        # println("a*bsplice", a*b_splice)
        result_data[reverse(broad_index_result)...] = (a*b_splice).data
    end
    return Ket(result_basis, vec(result_data))
end

function *(a::LazyTensor, b::Ket)
    check_multiplicable(a.basis_r, b.basis)
    for (op, operator_index) = zip(a.operators, a.indices)
        b = mul_suboperator(op, b, operator_index)
    end
    return b
end

function *(a::LazyTensor, b::Operator)
    check_multiplicable(a.basis_r, b.basis_l)
    x = Operator(a.basis_l, b.basis_r)
    for j=1:length(b.basis_r)
        b_j = Ket(b.basis_l, b.data[:,j])
        x.data[:,j] = (a*b_j).data
    end
    return x
end


# type EmbeddedOperator <: AbstractOperator
#     basis_l::CompositeBasis
#     basis_r::CompositeBasis
#     indices_l::Array{Int}
#     indices_r::Array{Int}
#     operator::AbstractOperator

#     function EmbeddedOperator(basis_l::CompositeBasis, basis_r::CompositeBasis, indices_l::Array{Int}, indices_r::Array{Int}, operator::AbstractOperator)
#         if length(indices_l)==1
#             @assert(operator.basis_l==basis_l.bases[indices_l[1]])
#         else
#             @assert(operator.basis_l.bases==basis_l.bases[indices_l])
#         end
#         if length(indices_r)==1
#             @assert(operator.basis_r==basis_r.bases[indices_r[1]])
#         else
#             @assert(operator.basis_r.bases==basis_r.bases[indices_r])
#         end
#         compl_indices_l = complementary_indices(length(basis_l.shape), indices_l)
#         compl_indices_r = complementary_indices(length(basis_r.shape), indices_r)
#         @assert(basis_l.bases[compl_indices_l]==basis_r.bases[compl_indices_r])
#         new(basis_l, basis_r, indices_l, indices_r, operator)
#     end
# end

# function *(a::EmbeddedOperator, b::Ket)
#     check_multiplicable(a.basis_r, b.basis)
#     uninvolved_indices_l = complementary_indices(length(a.basis_l.shape), a.indices_l)
#     uninvolved_indices_r = complementary_indices(length(a.basis_r.shape), a.indices_r)
#     uninvolved_shape = a.basis_l.shape[uninvolved_indices_l]
#     x = Ket(a.basis_l)
#     data_x = reshape(x.data, x.basis.shape...)
#     broad_index_l = {(1:n) for n=a.basis_l.shape}
#     broad_index_r = {(1:n) for n=a.basis_r.shape}
#     #broad_index_l = zeros(Int, length(a.basis_l.shape))
#     #broad_index_r = zeros(Int, length(a.basis_r.shape))
#     for uninvolved_index in Iterators.product([1:n for n = uninvolved_shape]...)
#         # println("Uninvolved Index: ", uninvolved_index)
#         # println("Uninvolved Indices L: ", uninvolved_indices_l)
#         # println("Uninvolved Indices R: ", uninvolved_indices_r)
#         broad_index_l[uninvolved_indices_l] = [uninvolved_index...]
#         broad_index_r[uninvolved_indices_r] = [uninvolved_index...]
#         # println("Broad Index L: ", broad_index_l)
#         # println("Broad Index R: ", broad_index_r)
#         subdata_b = reshape(reshape(b.data, b.basis.shape...)[broad_index_r...], prod(a.operator.basis_r.shape))
#         sub_b = Ket(a.operator.basis_r, subdata_b)
#         sub_x = a.operator*sub_b
#         data_x[broad_index_l...] = reshape(sub_x.data, sub_x.basis.shape...)
#         # println("subdata_x: ", data_x[broad_index_l])
#         # println("x.data", x.data)
#     end
#     return x
# end

# type LazySum <: LazyOperator
#     basis_l::Basis
#     basis_r::Basis
#     operands::Array{AbstractOperator}
# end

end