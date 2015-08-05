module fock

using ..bases, ..states, ..operators

export FockBasis, coherentstate


type FockBasis <: Basis
    shape::Vector{Int}
    N0::Int
    N1::Int
    FockBasis(N0::Int, N1::Int) = 0 < N0 <= N1 ? new([N1-N0+1], N0, N1) : throw(DimensionMismatch())
end

FockBasis(N::Int) = FockBasis(1,N)

==(b1::FockBasis, b2::FockBasis) = b1.N0==b2.N0 && b1.N1==b2.N1

function coherentstate(b::FockBasis, alpha::Number)
    alpha = complex(alpha)
    x = zeros(Complex128, b.N1)
    x[1] = complex(exp(-abs2(alpha)/2))
    for n=2:b.N1
        x[n] = x[n-1]*alpha/sqrt(n-1)
    end
    return Ket(b, x)
end

end # module
