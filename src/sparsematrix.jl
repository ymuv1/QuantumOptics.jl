module sparse

export SparseMatrix

type SparseMatrix{T}
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
    return SparseMatrix(index_l, index_r, values)
end

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


end