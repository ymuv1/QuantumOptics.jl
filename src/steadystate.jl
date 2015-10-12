module steadystate

using ..states
using ..operators
using ..timeevolution
using ..metrics


type ConvergenceReached <: Exception end

function master(H::AbstractOperator, J::Vector;
                rho0::Operator=tensor(basis_ket(H.basis_l, 1), basis_bra(H.basis_r, 1)),
                eps::Float64=1e-3,
                Gamma::Union{Real, Vector, Matrix}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    t0 = 0.
    rho0 = deepcopy(rho0)
    function fout(t, rho)
        dt = t - t0
        drho = metrics.tracedistance(rho0, rho)
        t0 = t
        rho0.data[:] = rho.data
        if drho/dt < eps
            throw(ConvergenceReached())
        end
    end
    try
        timeevolution.master([0., Inf], rho0, H, J; Gamma=Gamma, Jdagger=Jdagger,
                            hmin=1e-7, hmax=Inf,
                            tmp=tmp,
                            display_initialvalue=false,
                            display_finalvalue=false,
                            display_intermediatesteps=true,
                            fout=fout, kwargs...)
    catch e
        if !isa(e, ConvergenceReached)
            rethrow(e)
        end
    end
    return rho0
end

end # module
