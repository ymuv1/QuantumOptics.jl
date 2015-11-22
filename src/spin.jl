module spin

import Base.==

using ..bases, ..states, ..operators

export SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown


"""
Basis for spin-n particles.

The basis can be created for arbitrary spin numbers by using a rational number,
e.g. SpinBasis(3//2). The Pauli operators are defined for all possible
spin numbers.
"""
type SpinBasis <: Basis
    shape::Vector{Int}
    spinnumber::Rational{Int}
    function SpinBasis(spinnumber::Rational{Int})
        @assert den(spinnumber) == 2 || den(spinnumber) == 1
        @assert den(spinnumber) > 0
        new([num(spinnumber*2 + 1)], spinnumber)
    end
end

==(b1::SpinBasis, b2::SpinBasis) = b1.spinnumber==b2.spinnumber


function sigmax(b::SpinBasis)
    d = [complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,1) + diagm(d,-1))
end

function sigmay(b::SpinBasis)
    d = [1im*complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,-1) - diagm(d,1))
end

function sigmaz(b::SpinBasis)
    Operator(b, diagm([complex(2*m) for m=b.spinnumber:-1:-b.spinnumber]))
end

function sigmap(b::SpinBasis)
    S = (b.spinnumber + 1)*b.spinnumber
    d = [complex(sqrt(float(S - m*(m+1)))) for m=b.spinnumber-1:-1:-b.spinnumber]
    Operator(b, diagm(d, 1))
end

function sigmam(b::SpinBasis)
    S = (b.spinnumber + 1)*b.spinnumber
    d = [complex(sqrt(float(S - m*(m-1)))) for m=b.spinnumber:-1:-b.spinnumber+1]
    Operator(b, diagm(d, -1))
end


spinup(b::SpinBasis) = basis_ket(b, 1)
spindown(b::SpinBasis) = basis_ket(b, b.shape[1])


end #module
