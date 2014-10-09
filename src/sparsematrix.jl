module sparse

export SparseMatrix, sparse_eye

type SparseMatrix{T}
    shape::Vector{Int}
    index_l::Vector{Int}
    index_r::Vector{Int}
    values::Vector{T}
end

function SparseMatrix{T}(a::Matrix{T})
    index_l = Int[]
    index_r = Int[]
    values = T[]
    for j=1:size(a,2), i=1:size(a,1)
        if abs2(a[i,j]) > 1e-15
            push!(index_l, i)
            push!(index_r, j)
            push!(values, a[i,j])
        end
    end
    return SparseMatrix([size(a)...], index_l, index_r, values)
end

Base.size(a::SparseMatrix, i::Int) = a.shape[i]

sparse_eye(T, N::Int) = SparseMatrix(Int[N,N], Int[1:N], Int[1:N], T[one(T) for i=1:N])
sparse_eye(T, N1::Int, N2::Int) = (N = min(N1,N2); SparseMatrix(Int[N1,N2], Int[1:N], Int[1:N], T[one(T) for i=1:N]))

function perm!(S::SparseMatrix, perm)
    S.index_l = S.index_l[perm]
    S.index_r = S.index_r[perm]
    S.values = S.values[perm]
    return nothing
end

optimize_left!(S::SparseMatrix) = perm!(S, sortperm(S.index_l))
optimize_right!(S::SparseMatrix) = perm!(S, sortperm(S.index_r))

function add_element!{T}(a::SparseMatrix{T}, i::Int, j::Int, value::T)
    index = find((a.index_l.==i) & (a.index_r.==j))
    if length(index)==0
        push!(a.index_l, i)
        push!(a.index_r, j)
        push!(a.values, value)
    elseif length(index)==1
        a.values[index[1]] += value
    else
        error("Index not unique.")
    end
end

function Base.full{T}(a::SparseMatrix{T})
    result = zeros(T, a.shape...)
    for i=1:length(a.values)
        result[a.index_l[i], a.index_r[i]] = a.values[i]
    end
    return result
end

Base.ctranspose{T}(a::SparseMatrix{T}) = SparseMatrix(reverse(a.shape), deepcopy(a.index_r), deepcopy(a.index_l), conj(a.values))

function Base.kron{T1,T2}(a::SparseMatrix{T1}, b::SparseMatrix{T2})
    shape = [a.shape[1]*b.shape[1], a.shape[2]*b.shape[2]]
    result = SparseMatrix(shape, Int[], Int[], promote_type(T1,T2)[])
    for i=1:length(a.values), j=1:length(b.values)
        add_element!(result, (a.index_l[i]-1)*b.shape[1] + b.index_l[j], (a.index_r[i]-1)*b.shape[2] + b.index_r[j], a.values[i]*b.values[j])
    end
    return result
end

function +{T}(a::SparseMatrix{T}, b::SparseMatrix{T})
    @assert a.shape == b.shape
    result = deepcopy(a)
    for i=1:length(b.values)
        add_element!(result, b.index_l[i], b.index_r[i], b.values[i])
    end
    return result
end

function -{T}(a::SparseMatrix{T}, b::SparseMatrix{T})
    @assert a.shape == b.shape
    result = deepcopy(a)
    for i=1:length(b.values)
        add_element!(result, b.index_l[i], b.index_r[i], -b.values[i])
    end
    return result
end

function *{T1,T2}(a::SparseMatrix{T1}, b::SparseMatrix{T2})
    @assert size(a,2)==size(b,1)
    T = promote_type(T1, T2)
    index_l = Int[]
    index_r = Int[]
    values = T[]
    result = SparseMatrix([a.shape[1], b.shape[2]], index_l, index_r, values)
    for i=1:length(a.values), j=1:length(b.values)
        if a.index_r[i]==b.index_l[j]
            x = a.values[i]*b.values[j]
            add_element!(result, a.index_l[i], b.index_r[j], x)
        end
    end
    return result
end

function *{T1<:Number, T2}(a::T1, b::SparseMatrix{T2})
    a = convert(T2, a)
    if abs2(a) < 2*eps(abs2(a))
        return zero(b)
    end
    result = deepcopy(b)
    for i=1:length(b.values)
        result.values[i] *= a
    end
    return result
end

function *{T1, T2}(a::SparseMatrix{T1}, b::Matrix{T2})
    @assert size(a,2) == size(b,1)
    T = promote_type(T1, T2)
    result = zeros(T, size(a,1), size(b,2))
    for s=1:length(a.values)
        i = a.index_l[s]
        j = a.index_r[s]
        for k=1:size(b,2)
            result[i,k] += a.values[s]*b[j,k]
        end
    end
    return result
end

function *{T1, T2}(a::Matrix{T1}, b::SparseMatrix{T2})
    @assert size(a,2) == size(b,1)
    T = promote_type(T1, T2)
    result = zeros(T, size(a,1), size(b,2))
    for s=1:length(b.values)
        i = b.index_l[s]
        j = b.index_r[s]
        for k=1:size(a,1)
            result[k,j] += a[k,i]*b.values[s]
        end
    end
    return result
end

*{T}(a::SparseMatrix{T}, b) = convert(T,b)*a
/{T}(a::SparseMatrix{T}, b) = a*(one(T)/convert(T,b))

function _fMB{T}(j::Int, index_l::Vector{Int}, index_r::Vector{Int}, values::Vector{T}, alpha::T, B::Matrix{T}, result::Matrix{T})
    @inbounds for i=1:length(values)
        result[index_l[i], j] += alpha * values[i] * B[index_r[i], j]
    end
end

function gemm!{T}(alpha::T, M::SparseMatrix{T}, B::Matrix{T}, beta::T, result::Matrix{T})
    for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(B,1)
        _fMB(j, M.index_l, M.index_r, M.values, alpha, B, result)
    end
end

function _fBM{T}(index_l::Int, index_r::Int, value::T, alpha::T, B::Matrix{T}, result::Matrix{T})
    @inbounds for j=1:size(B,1)
        result[j, index_r] += alpha * value * B[j, index_l]
    end
end

function gemm!{T}(alpha::T, B::Matrix{T}, M::SparseMatrix{T}, beta::T, result::Matrix{T})
    for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for i=1:length(M.index_l)
        _fBM(M.index_l[i], M.index_r[i], M.values[i], alpha, B, result)
    end
end

function _fMv{T}(index_l::Vector{Int}, index_r::Vector{Int}, values::Vector{T}, alpha::T, v::Vector{T}, result::Vector{T})
    for i=1:length(values)
        result[index_l[i]] += alpha * values[i] * v[index_r[i]]
    end
end

function gemv!{T}(alpha::T, M::SparseMatrix{T}, v::Vector{T}, beta::T, result::Vector{T})
    for i=1:length(result)
        result[i] *= beta
    end
    _fMv(M.index_l, M.index_r, M.values, alpha, v, result)
end

end