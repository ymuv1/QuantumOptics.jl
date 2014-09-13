using Base.LinAlg.BLAS

function scale{T}(alpha::T, a::Matrix{T}, result::Matrix{T})
    @inbounds for j=1:size(a,2)
        for i=1:size(a,1)
            result[i,j] = alpha*a[i,j]
        end
    end
    return nothing
end

N = 4000
srand(0)
A = rand(N,N)
result = zeros(N,N)

function f0{T}(alpha::T, A::Matrix{T}, result::Matrix{T}, repeat=1)
    for i=1:repeat
        scale(alpha, A, result)
    end
    return nothing
end

@time f1(0.1, A, result)
@time f1(0.1, A, result)


A_vec = reshape(A, length(A))
result_vec = reshape(result, length(result))
N = length(A_vec)
@time BLAS.blascopy!(N, A_vec, 1, result_vec, 1)
@time BLAS.blascopy!(N, A_vec, 1, result_vec, 1)
@time BLAS.scale!(0.1, A)
@time BLAS.scale!(0.1, A)
