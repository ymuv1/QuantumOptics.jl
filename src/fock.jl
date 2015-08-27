module fock

import Base.==

using ..bases, ..states, ..operators

export FockBasis, coherentstate, number, destroy, create, qfunc


type FockBasis <: Basis
    shape::Vector{Int}
    N0::Int
    N1::Int
    FockBasis(N0::Int, N1::Int) = 0 < N0 <= N1 ? new([N1-N0+1], N0, N1) : throw(DimensionMismatch())
end

FockBasis(N::Int) = FockBasis(1,N)

==(b1::FockBasis, b2::FockBasis) = b1.N0==b2.N0 && b1.N1==b2.N1

number(b::Basis) = Operator(b, b, diagm(map(Complex, 0:(length(b)-1))))
destroy(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),1))
create(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),-1))

function coherentstate(b::FockBasis, alpha::Number)
    alpha = complex(alpha)
    x = zeros(Complex128, b.N1)
    x[1] = complex(exp(-abs2(alpha)/2))
    for n=2:b.N1
        x[n] = x[n-1]*alpha/sqrt(n-1)
    end
    return Ket(b, x)
end

function qfunc(rho::AbstractOperator, X::Vector{Float64}, Y::Vector{Float64})
    M = zeros(Float64, length(X), length(Y))
    @assert rho.basis_l == rho.basis_r
    for (i,x)=enumerate(X), (j,y)=enumerate(Y)
        z = complex(x,y)
        coh = coherent_state(rho.basis_l, z)
        M[i,j] = real(dagger(coh)*rho*coh)
    end
    return M
end

end # module
