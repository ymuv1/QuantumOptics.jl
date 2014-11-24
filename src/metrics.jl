module metrics

using ..operators

export tracedistance


function tracedistance(rho::Operator, sigma::Operator)
    s = eigvals(Hermitian((rho - sigma).data))
    return 0.5*sum(abs(s))
end

end