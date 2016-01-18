module spin

import Base.==

using ..bases, ..states, ..operators

export SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown


"""
Basis for spin-n particles.

The basis can be created for arbitrary spinnumbers by using a rational number,
e.g. SpinBasis(3//2). The Pauli operators are defined for all possible
spinnumbers.

Arguments
---------

spinnumber
    Rational number specifying the spin quantum number.
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

"""
Pauli sigma_x operator for the given SpinBasis.
"""
function sigmax(b::SpinBasis)
    d = [complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,1) + diagm(d,-1))
end

"""
Pauli sigma_y operator for the given SpinBasis.
"""
function sigmay(b::SpinBasis)
    d = [1im*complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,-1) - diagm(d,1))
end

"""
Pauli sigma_z operator for the given SpinBasis.
"""
function sigmaz(b::SpinBasis)
    Operator(b, diagm([complex(2*m) for m=b.spinnumber:-1:-b.spinnumber]))
end

"""
Raising operator :math:`\\sigma_+` for the given SpinBasis.
"""
function sigmap(b::SpinBasis)
    S = (b.spinnumber + 1)*b.spinnumber
    d = [complex(sqrt(float(S - m*(m+1)))) for m=b.spinnumber-1:-1:-b.spinnumber]
    Operator(b, diagm(d, 1))
end

"""
Lowering operator :math:`\\sigma_-` for the given SpinBasis.
"""
function sigmam(b::SpinBasis)
    S = (b.spinnumber + 1)*b.spinnumber
    d = [complex(sqrt(float(S - m*(m-1)))) for m=b.spinnumber:-1:-b.spinnumber+1]
    Operator(b, diagm(d, -1))
end


"""
Spin up state for the given SpinBasis.
"""
spinup(b::SpinBasis) = basis_ket(b, 1)

"""
Spin down state for the given SpinBasis.
"""
spindown(b::SpinBasis) = basis_ket(b, b.shape[1])


end #module
