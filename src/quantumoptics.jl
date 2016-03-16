__precompile__()

module quantumoptics

export bases, Basis, GenericBasis, CompositeBasis,
        tensor, ⊗,
        states, StateVector, Bra, Ket, basis_bra, basis_ket,
                dagger, normalize, normalize!,
        operators, Operator, DenseOperator,
                expect, dense_identity, ptrace, embed,
        operators_lazy, LazyOperator, LazyTensor, LazySum, LazyProduct,
        operators_sparse, SparseOperator,
        super, DenseSuperOperator, SparseSuperOperator,
                spre, spost, liouvillian,
        fock, FockBasis, number, destroy, create,
                fockstate, coherentstate, qfunc,
        spin, SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown,
        particle, PositionBasis, MomentumBasis, gaussianstate,
                positionoperator, momentumoperator, laplace_x, laplace_p,
        metrics, tracedistance,
        spectralanalysis, operatorspectrum, operatorspectrum_hermitian, eigenbasis, eigenbasis_hermitian, groundstate,
        timeevolution_simple,
        timeevolution,
        steadystate,
        correlations


include("bases.jl")
include("states.jl")
include("operators.jl")
include("sparsematrix.jl")
include("operators_sparse.jl")
include("operators_lazy.jl")
include("superoperators.jl")
include("spin.jl")
include("fock.jl")
include("particle.jl")
include("metrics.jl")
include("ode_dopri.jl")
include("timeevolution_simple.jl")
include("timeevolution.jl")
include("steadystate.jl")
include("correlations.jl")
include("spectralanalysis.jl")
include("io.jl")


using .bases
using .states
using .operators
using .operators_sparse
using .operators_lazy
using .superoperators
using .spin
using .fock
using .particle
using .metrics
using .spectralanalysis


end # module
