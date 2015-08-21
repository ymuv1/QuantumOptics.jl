
N = 8000
#N = 100

srand(1)

A = complex(sprand(N, N, 0.1))

v1 = rand(Complex128, N)
result = rand(Complex128, N)


function mv1(alpha::Complex128, M::SparseMatrixCSC{Complex128}, v::Vector{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n::Int
        vj = alpha*v[colindex]
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[M.rowval[i]] += M.nzval[i]*vj
        end
    end
    nothing
end

function mv1(alpha::Complex128, v::Vector{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n::Int
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[colindex] += M.nzval[i]*alpha*v[M.rowval[i]]
        end
    end
    nothing
end

@time mv1(Complex(1.), A, v1, Complex(0.), result)
@time mv1(Complex(1.), A, v1, Complex(0.), result)

println(norm(result-A*v1))

@time mv1(Complex(1.), v1, A, Complex(0.), result)
@time mv1(Complex(1.), v1, A, Complex(0.), result)

println(norm(transpose(result)-transpose(v1)*A))
