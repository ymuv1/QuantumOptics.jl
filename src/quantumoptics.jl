module quantumoptics

export bases, Basis, GenericBasis, CompositeBasis, FockBasis, compose,
        states, StateVector, Bra, Ket, tensor, dagger, ⊗, basis_bra, basis_ket, coherent_state, basis,
        operators, AbstractOperator, Operator, expect, identity, number, destroy, create,
                    sigmax, sigmay, sigmaz, sigmap, sigmam, spinbasis, ptrace, qfunc, embed,
        operators_lazy,
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
using .metrics


end # module
