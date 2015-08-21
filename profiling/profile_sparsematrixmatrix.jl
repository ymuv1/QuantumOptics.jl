#N = 8000
N = 500

srand(1)

A = complex(sprand(N, N, 0.1))

v1 = rand(Complex128, N, N)
result = rand(Complex128, N, N)


function mm1(alpha::Complex128, M::SparseMatrixCSC{Complex128}, v::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for colindex = 1:M.n::Int
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            for j=1:size(result,2)
                result[M.rowval[i], j] += M.nzval[i]*alpha*v[colindex, j]
            end
        end
    end
    nothing
end

function mm2(alpha::Complex128, M::SparseMatrixCSC{Complex128}, M2::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:M.m
        for colindex = 1:M.n
            m2 = alpha*M2[colindex, j]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i], j] += M.nzval[i]*m2
            end
        end
    end
    nothing
end

function mm1(alpha::Complex128, v::Matrix{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            mi = M.nzval[i]*alpha
            mrowvali = M.rowval[i]
            for j=1:size(result,1)
                result[j,colindex] += mi*v[j,mrowvali]
            end
        end
    end
    nothing
end

function mm2(alpha::Complex128, v::Matrix{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(result,1)
        for colindex = 1:M.n
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[j,colindex] += M.nzval[i]*alpha*v[j,M.rowval[i]]
            end
        end
    end
    nothing
end

@time mm1(Complex(1.), A, v1, Complex(0.), result)
@time mm1(Complex(1.), A, v1, Complex(0.), result)

println(norm(result-A*v1))

@time mm2(Complex(1.), A, v1, Complex(0.), result)
@time mm2(Complex(1.), A, v1, Complex(0.), result)

println(norm(result-A*v1))

@time mm1(Complex(1.), v1, A, Complex(0.), result)
@time mm1(Complex(1.), v1, A, Complex(0.), result)

println(norm(result-v1*A))

@time mm2(Complex(1.), v1, A, Complex(0.), result)
@time mm2(Complex(1.), v1, A, Complex(0.), result)

println(norm(result-v1*A))
