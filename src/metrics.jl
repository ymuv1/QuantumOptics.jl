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

function tracedistance{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance not implemented for $(T). Use dense operators instead."))
end

end # module
