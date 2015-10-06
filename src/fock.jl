module fock

import Base.==

using ..bases, ..states, ..operators

export FockBasis, number, destroy, create, fockstate, coherentstate, qfunc


"""
Basis for a Fock space.

Arguments:
* Nmin
    Minimal particle number incuded in this basis. By default it is zero, i.e.
    starting with the vacuum state.
* Nmax
    Maximal particle number incuded in this basis.
"""
type FockBasis <: Basis
    shape::Vector{Int}
    Nmin::Int
    Nmax::Int
    function FockBasis(Nmin::Int, Nmax::Int)
        if Nmin < 0 || Nmax <= Nmin
            throw(DimensionMismatch())
        end
        new([Nmax-Nmin+1], Nmin, Nmax)
    end
end

FockBasis(Nmax::Int) = FockBasis(0, Nmax)


==(b1::FockBasis, b2::FockBasis) = b1.Nmin==b2.Nmin && b1.Nmax==b2.Nmax

"""
Number operator for the given Fock space.
"""
number(b::FockBasis) = Operator(b, diagm([complex(x) for x=b.Nmin:b.Nmax]))

"""
Annihilation operator for the given Fock space.
"""
destroy(b::FockBasis) = Operator(b, diagm([complex(sqrt(x)) for x=b.Nmin+1:b.Nmax],1))

"""
Creation operator for the given Fock space.
"""
create(b::FockBasis) = Operator(b, diagm([complex(sqrt(x)) for x=b.Nmin+1:b.Nmax],-1))


"""
Fock state for the given particle number.
"""
function fockstate(b::FockBasis, n::Int)
    @assert b.Nmin <= n <= b.Nmax
    basis_ket(b, n+1-b.Nmin)
end

"""
Coherent state |alpha> for the given Fock basis.
"""
function coherentstate(b::FockBasis, alpha::Complex128)
    x = zeros(Complex128, b.Nmax - b.Nmin + 1)
    if b.Nmin == 0
        x[1] = exp(-abs2(alpha)/2)
    else
        x[1] = exp(-abs2(alpha)/2)*alpha^b.Nmin/sqrt(factorial(b.Nmin))
    end
    for n=1:(b.Nmax-b.Nmin)
        x[n+1] = x[n]*alpha/sqrt(b.Nmin + n)
    end
    return Ket(b, x)
end

coherentstate(b::FockBasis, alpha::Number) = coherentstate(b, complex(alpha))


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
