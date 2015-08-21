using quantumoptics.sparsematrix
using Base.LinAlg.BLAS

N = 600

x1 = complex(sprand(N, N, 0.1))
x2 = complex(sprand(N, N, 0.1))
y1 = quantumoptics.sparsematrix.SparseMatrix(full(x1))
y2 = quantumoptics.sparsematrix.SparseMatrix(full(x2))

x1*x2
v1 = ones(Complex128, N)
v2 = zeros(Complex128, N)

function f_mv(alpha::Complex128, M::SparseMatrixCSC{Complex128}, v::Vector{Complex128}, beta::Complex128, result::Vector{Complex128})
    result[:] = beta*result + alpha*(M*v)
end

function f_mv(alpha::Complex128, v::Vector{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Vector{Complex128})
    result[:] = beta*result + alpha*transpose(transpose(v)*M)
end

function f_mm(alpha::Complex128, M::SparseMatrixCSC{Complex128}, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    result[:] = beta*result + (alpha*M)*B
end

function f_mm(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Matrix{Complex128})
    result[:] = beta*result + B*(alpha*M)
end


function mv(alpha::Complex128, M::SparseMatrixCSC{Complex128}, v::Vector{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n
        vj = alpha*v[colindex]
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[M.rowval[i]] += M.nzval[i]*vj
        end
    end
    nothing
end

function mv(alpha::Complex128, v::Vector{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[colindex] += M.nzval[i]*alpha*v[M.rowval[i]]
        end
    end
    nothing
end


function mm(alpha::Complex128, M::SparseMatrixCSC{Complex128}, M2::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
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

function mm(alpha::Complex128, v::Matrix{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Matrix{Complex128})
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


M = full(x2)
result = full(x2)

println("spMatrix-Vector")
@time f_mv(Complex(1.), x1, v1, Complex(0.), v2)
@time f_mv(Complex(1.), x1, v1, Complex(0.), v2)

@time mv(Complex(1.), x1, v1, Complex(0.), v2)
@time mv(Complex(1.), x1, v1, Complex(0.), v2)

@time quantumoptics.sparsematrix.gemv!(Complex(1.), y1, v1, Complex(0.), v2)
@time quantumoptics.sparsematrix.gemv!(Complex(1.), y1, v1, Complex(0.), v2)


println("Vector-spMatrix")
@time f_mv(Complex(1.), v1, x1, Complex(0.), v2)
@time f_mv(Complex(1.), v1, x1, Complex(0.), v2)

@time mv(Complex(1.), v1, x1, Complex(0.), v2)
@time mv(Complex(1.), v1, x1, Complex(0.), v2)

# @time quantumoptics.sparsematrix.gemv!(Complex(1.), v1, y1, Complex(0.), v2)
# @time quantumoptics.sparsematrix.gemv!(Complex(1.), v1, y1, Complex(0.), v2)


println("spMatrix-Matrix")
@time f_mm(Complex(1.), x1, M, Complex(0.), result)
@time f_mm(Complex(1.), x1, M, Complex(0.), result)

@time mm(Complex(1.), x1, M, Complex(0.), result)
@time mm(Complex(1.), x1, M, Complex(0.), result)

@time quantumoptics.sparsematrix.gemm!(Complex(1.), y1, M, Complex(0.), result)
@time quantumoptics.sparsematrix.gemm!(Complex(1.), y1, M, Complex(0.), result)


println("Matrix-spMatrix")
@time f_mm(Complex(1.), M, x1, Complex(0.), result)
@time f_mm(Complex(1.), M, x1, Complex(0.), result)

@time mm(Complex(1.), M, x1, Complex(0.), result)
@time mm(Complex(1.), M, x1, Complex(0.), result)

@time quantumoptics.sparsematrix.gemm!(Complex(1.), M, y1, Complex(0.), result)
@time quantumoptics.sparsematrix.gemm!(Complex(1.), M, y1, Complex(0.), result)
