module inplacearithmetic

using Base.LinAlg.BLAS

export mul!, add!, sub!, imul!, iadd!, isub!, set!, gemm!


gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::Matrix{T}, beta::T, result::Matrix{T}) = BLAS.gemm!('N', 'N', alpha, a, b, beta, result)

function mul!(a::Matrix, b::Matrix, result::Matrix)
    BLAS.gemm!('N', 'N', one(eltype(a)), a, b, zero(eltype(a)), result)
    return result
end

function mul!(a::Matrix, b::Number, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = b*a[i,j]
    end
    return result
end

mul!(a::Number, b::Matrix, result::Matrix) = mul!(b, a, result)
imul!(a::Matrix, b::Number) = mul!(a,b,a)
imul!(a::Number, b::Matrix) = mul!(b,a,b)


function add!(a::Matrix, b::Matrix, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = a[i,j] + b[i,j]
    end
    return result
end
iadd!(a::Matrix, b::Matrix) = add!(a,b,a)


function sub!(a::Matrix, b::Matrix, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = a[i,j] - b[i,j]
    end
    return a
end
isub!(a::Matrix, b::Matrix) = sub!(a,b,a)


function set!(a::Matrix, b::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        a[i,j] = b[i,j]
    end
end

end