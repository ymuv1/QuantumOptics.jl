module quantumoptics

export bases, Basis, GenericBasis, CompositeBasis, FockBasis, compose,
        states, StateVector, Bra, Ket, tensor, dagger, ⊗, basis_bra, basis_ket, coherentstate, basis,
        normalize, normalize!,
        operators, AbstractOperator, Operator, expect, identity, number, destroy, create,
                    sigmax, sigmay, sigmaz, sigmap, sigmam, spinbasis, ptrace, qfunc, embed,
        operators_lazy, LazyOperator,
        operators_sparse, SparseOperator, sparse_identity,
        sparse,
        metrics, tracedistance,
        timeevolution_simple,
        timeevolution

include("bases.jl")
include("states.jl")
include("operators.jl")
include("sparsematrix.jl")
include("operators_sparse.jl")
include("operators_lazy.jl")
include("spins.jl")
include("particle.jl")
include("fock.jl")
include("metrics.jl")
include("ode_dopri.jl")
include("timeevolution_simple.jl")
include("timeevolution.jl")
include("io.jl")

using .bases
using .states
using .sparse
using .operators
using .operators_sparse
using .spins
using .fock
using .particle
using .metrics


end # module
