using Base.LinAlg.BLAS

N = 4000
srand(0)
A = rand(N,N)
B = rand(N,N)
result = zeros(N,N)


function substep{T}(a::Matrix{T}, beta::T, b::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        for i=1:size(b,1)
            @inbounds result[i,j] = a[i,j] + beta*b[i,j]
        end
    end
end

function f0{T}(A::Matrix{T}, alpha::T, B::Matrix{T}, result::Matrix{T}, repeat=1)
    for i=1:repeat
        substep(A, alpha, B, result)
    end
    return nothing
end

@time f0(A, 0.1, B, result)
@time f0(A, 0.1, B, result)


A_vec = reshape(A, length(A))
result_vec = reshape(result, length(result))
N = length(A_vec)
@time BLAS.axpy!(N, 0.1, A, 1, B, 1)
@time BLAS.axpy!(N, 0.1, A, 1, B, 1)

# @time BLAS.blascopy!(N, A_vec, 1, result_vec, 1)
# @time BLAS.blascopy!(N, A_vec, 1, result_vec, 1)
# @time BLAS.scale!(0.1, A)
# @time BLAS.scale!(0.1, A)


