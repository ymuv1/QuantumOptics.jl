module metrics

using ..operators, ..operators_sparse

export tracedistance


function tracedistance(rho::DenseOperator, sigma::DenseOperator)
    delta = (rho - sigma).data
    @assert size(delta, 1) == size(delta, 2)
    for i=1:size(delta,1)
        delta[i,i] = real(delta[i,i])
    end
    s = eigvals(Hermitian(delta))
    return 0.5*sum(abs(s))
end

function tracedistance(rho::SparseOperator, sigma::SparseOperator)
    N = size(rho.data, 1)
    if N < 10
        return tracedistance(full(rho), full(sigma))
    end
    delta = (rho - sigma).data
    @assert size(delta, 1) == size(delta, 2)
    s, v, nconv, niter, nmult, resid = Base.eigs(delta; nev=N-2, ncv=N, sigma=1e-30)
    return 0.5*sum(abs(s))
end

end # module
