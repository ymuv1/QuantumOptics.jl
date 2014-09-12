module test_sparse

export SparseMatrix

type SparseMatrix{T}
    index_l::Vector{Int}
    index_r::Vector{Int}
    values::Vector{T}
end

function SparseMatrix{T}(A::Matrix{T})
    index_l = Int[]
    index_r = Int[]
    values = T[]
    for j=1:size(A,2), i=1:size(A,1)
        if abs2(A[i,j]) > 1e-15
            push!(index_l, i)
            push!(index_r, j)
            push!(values, A[i,j])
        end
    end
    return SparseMatrix(index_l, index_r, values)
end


function perm!(S::SparseMatrix, perm)
    S.index_l = S.index_l[perm]
    S.index_r = S.index_r[perm]
    S.values = S.values[perm]
    return nothing
end

optimize_left!(S::SparseMatrix) = perm!(S, sortperm(S.index_l))
optimize_right!(S::SparseMatrix) = perm!(S, sortperm(S.index_r))

function mul_SB_js{T}(S::SparseMatrix{T}, B::Matrix{T}, result::Matrix{T})
    @inbounds for j=1:size(B,1)
        for s=1:length(S.index_l)
            result[S.index_l[s], j] += S.values[s] * B[S.index_r[s], j]
        end
    end
    return nothing
end

function mul_SB_sj{T}(S::SparseMatrix{T}, B::Matrix{T}, result::Matrix{T})
    @inbounds for s=1:length(S.index_l)
        for j=1:size(B,1)
            result[S.index_l[s], j] += S.values[s] * B[S.index_r[s], j]
        end
    end
    return nothing
end

function mul_BS_js{T}(B::Matrix{T}, S::SparseMatrix{T}, result::Matrix{T})
    @inbounds for i=1:size(B,1)
        for s=1:length(S.index_l)
            result[i, S.index_r[s]] += B[i, S.index_l[s]] * S.values[s]
        end
    end
    return nothing
end

function mul_BS_sj{T}(B::Matrix{T}, S::SparseMatrix{T}, result::Matrix{T})
    @inbounds for s=1:length(S.index_l)
        for i=1:size(B,1)
            result[i, S.index_r[s]] += B[i, S.index_l[s]] * S.values[s]
        end
    end
    return nothing
end

function doprofile(N, fill=0.01, seed=0)
    srand(seed)
    Sfull = full(sprand(N, N, fill))
    S = SparseMatrix(Sfull)
    B = rand(N,N)

    #BS = B*Sfull
    #SB = Sfull*B

    result = zeros(eltype(B), N, N)
    # Run once for jit
    mul_SB_js(S, B, result)
    mul_SB_sj(S, B, result)
    mul_BS_js(B, S, result)
    mul_BS_sj(B, S, result)

    println("=========SxB=========")
    optimize_left!(S)
    fill!(result, 0)
    @time mul_SB_js(S, B, result)
    #println(norm(SB-result))
    fill!(result, 0)
    @time mul_SB_sj(S, B, result)
    #println(norm(SB-result))

    optimize_right!(S)
    fill!(result, 0)
    @time mul_SB_js(S, B, result)
    #println(norm(SB-result))
    fill!(result, 0)
    @time mul_SB_sj(S, B, result)
    #println(norm(SB-result))


    println("=========BxS=========")
    optimize_left!(S)
    fill!(result, 0)
    @time mul_BS_js(B, S, result)
    #println(norm(BS-result))
    fill!(result, 0)
    @time mul_BS_sj(B, S, result)
    #println(norm(BS-result))

    optimize_right!(S)
    fill!(result, 0)
    @time mul_BS_js(B, S, result)
    #println(norm(BS-result))
    fill!(result, 0)
    @time mul_BS_sj(B, S, result)
    #println(norm(BS-result))
end


doprofile(3000)

error()



function _f11{T}(j::Int, index_l::Vector{Int}, index_r::Vector{Int}, values::Vector{T}, alpha::T, B::Matrix{T}, result::Matrix{T})
    @inbounds for i=1:length(values)
        result[index_l[i], j] += alpha * values[i] * B[index_r[i], j]
    end
end

function gemm!{T}(alpha::T, M::SparseMatrix{T}, B::Matrix{T}, beta::T, result::Matrix{T})
    for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(B,1)
        _f11(j, M.index_l, M.index_r, M.values, alpha, B, result)
    end
end

function _f22{T}(j::Int, index_l::Vector{Int}, index_r::Vector{Int}, values::Vector{T}, alpha::T, B::Matrix{T}, result::Matrix{T})
    @inbounds for i=1:length(values)
        result[j, index_r[i]] += alpha * values[i] * B[j, index_l[i]]
    end
end

function gemm!{T}(alpha::T, B::Matrix{T}, M::SparseMatrix{T}, beta::T, result::Matrix{T})
    for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(B,1)
        _f22(j, M.index_l, M.index_r, M.values, alpha, B, result)
    end
end

# function gemm!{T}(alpha::T, B::Matrix{T}, M::SparseMatrix2{T}, beta::T, result::Matrix{T})
#     for j=1:size(result,2), i=1:size(result,1)
#         result[i,j] *= beta
#     end
#     @inbounds for i=1:length(M.values)
#         m = alpha*M.values[i]
#         i_l = M.index_l[i]
#         i_r = M.index_r[i]
#         for j=1:size(B,1)
#             result[i_l, j] += m * B[i_r, j]
#         end
#     end
# end

type SparseMatrix3{T}
    indices::Vector{Int}
    values::Vector{T}
end

function SparseMatrix3{T}(a::Matrix{T})
    indices = Int[]
    values = T[]
    for i=1:size(a,1), j=1:size(a,2)
        if abs2(a[i,j]) > 1e-15
            push!(indices, i)
            push!(indices, j)
            push!(values, a[i,j])
        end
    end

    return SparseMatrix3(indices, values)
end

function _f2r{T}(j::Int, indices::Vector{Int}, values::Vector{T}, alpha::T, B::Matrix{T}, result::Matrix{T})
    counter = 1
    @inbounds for i=1:length(values)
        result[indices[counter], j] += alpha * values[i] * B[indices[counter+1], j]
        counter += 2
    end
end

function gemm!{T}(alpha::T, M::SparseMatrix3{T}, B::Matrix{T}, beta::T, result::Matrix{T})
    for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(B,1)
        _f2r(j, M.indices, M.values, alpha, B, result)
    end
end


N = 2000
r = full(sprand(N, N, 0.1))
A = rand(N,N)

M1 = SparseMatrix(r)
M3 = SparseMatrix3(r)
M2 = SparseMatrix2(r)

println("version 1 reversed")
result = zeros(eltype(A), N, N)
@time gemm!(1., A, M1, 0., result)
result = zeros(eltype(A), N, N)
@time gemm!(1., A, M1, 0., result)
#println(norm(result-A*r))

println("version 2 reversed")
result = zeros(eltype(A), N, N)
@time gemm!(1., A, M2, 0., result)
result = zeros(eltype(A), N, N)
@time gemm!(1., A, M2, 0., result)
#println(norm(result-A*r))

println("version 1")
result = zeros(eltype(A), N, N)
@time gemm!(1., M1, A, 0., result)
result = zeros(eltype(A), N, N)
@time gemm!(1., M1, A, 0., result)
# println(norm(result-r*A))

println("version 2")
result = zeros(eltype(A), N, N)
@time gemm!(1., M2, A, 0., result)
result = zeros(eltype(A), N, N)
@time gemm!(1., M2, A, 0., result)
# println(norm(result-r*A))

println("version 3")
result = zeros(eltype(A), N, N)
@time gemm!(1., M3, A, 0., result)
result = zeros(eltype(A), N, N)
@time gemm!(1., M3, A, 0., result)
# println(norm(result-r*A))

end