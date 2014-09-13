function substep(a::Matrix{Complex128}, beta::Complex128, b::Matrix{Complex128}, result::Matrix{Complex128})
    for j=1:size(a,2), i=1:size(b,1)
        result[i,j] = a[i,j] + beta*b[i,j]
    end
end

function substep2{T}(a::Matrix{T}, beta::T, b::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        @simd for i=1:size(b,1)
            @inbounds result[i,j] = b[i,j]*beta + a[i,j]
        end
    end
end


n = 2^6

a = rand(Complex128,n,n)
b = rand(Complex128,n,n)
result = zeros(Complex128,n,n)
beta = Complex128(0,2)

function f1(N)
    for i=1:N
        substep(a,beta,b,result)
    end
end

function f2(N)
    for i=1:N
        substep2(a,beta,b,result)
    end
end


@time f1(100000)
@time f1(100000)

@time f2(100000)
@time f2(100000)



