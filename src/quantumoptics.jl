module quantumoptics

export bases, Basis, GenericBasis, CompositeBasis, FockBasis,
        states, StateVector, Bra, Ket, tensor, dagger, ⊗, basis_bra, basis_ket,
        operators, AbstractOperator, Operator, expect, identity, number, destroy, create,
                    sigmax, sigmay, sigmaz, sigmap, sigmam, spinbasis,
        operators_lazy,
        timeevolution_simple,
        timeevolution,
        inplacearithmetic

include("bases.jl")
include("states.jl")
include("operators.jl")
include("operators_lazy.jl")
include("timeevolution_simple.jl")
include("timeevolution.jl")

using .bases
using .states
using .operators

end

