module spins

import Base.==

using ..bases, ..states, ..operators

export SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam

type SpinBasis <: Basis
    shape::Vector{Int}
    spinnumber::Rational{Int}
    function SpinBasis(spinnumber::Rational{Int})
        @assert den(spinnumber) == 2 || den(spinnumber) == 1
        new([num(spinnumber*2 + 1)], spinnumber)
    end
end

==(b1::SpinBasis, b2::SpinBasis) = b1.spinnumber==b2.spinnumber

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

function sigmax(b::SpinBasis)
    d = [complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,1) + diagm(d,-1))
end

function sigmay(b::SpinBasis)
    d = [1im*complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:num(2*b.spinnumber)]
    Operator(b, diagm(d,-1) - diagm(d,1))
end

# sigmax(b::SpinBasis) = (sigmap(b) + sigmam(b))
# sigmay(b::SpinBasis) = (sigmap(b) - sigmam(b))/1im

end #module
