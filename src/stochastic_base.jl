using QuantumOpticsBase
using QuantumOpticsBase: check_samebases, check_multiplicable
using Random: AbstractRNG, randn!
import ..timeevolution: view_recast!, QO_CHECKS, pure_inference, as_vector

import DiffEqCallbacks, StochasticDiffEq, OrdinaryDiffEqCore, DiffEqNoiseProcess

"""
    integrate_stoch(tspan, df::Function, dg{Function}, x0{ComplexF64},
            state, dstate, fout::Function; kwargs...)

Integrate using StochasticDiffEq
"""
function integrate_stoch(tspan, df, dg, x0,
            state, dstate, fout, n; save_noise=false,
            save_everystep = false, callback=nothing, saveat=tspan,
            alg=StochasticDiffEq.EM(),
            noise_rate_prototype = nothing,
            noise_prototype_classical = nothing,
            noise=nothing,
            ncb=nothing,
            kwargs...)
    # TODO consider making a function that will generate the SDEProblem and Callback objects, and this will run that function's output

    function df_(dx, x, p, t)
        # recast!(state,x)
        # recast!(dstate,dx)
        # df(t, state, dstate)
        # recast!(dx,dstate)
        view_recast!(state,x)
        view_recast!(dstate,dx)
        df(t, state, dstate)
        return nothing
    end

    function dg_(dx, x, p, t)
        # recast!(state,x)
        view_recast!(state,x)
        dg(dx, t, state, dstate, n)
        return nothing
    end

    function fout_(x, t, integrator)
        # recast!(state,x)
        view_recast!(state,x)
        fout(t, state) 
    end

    nc = isa(noise_prototype_classical, Nothing) ? 0 : size(noise_prototype_classical)[2]
    if isa(noise, Nothing) && n > 0
        if n + nc == 1
            noise_ = StochasticDiffEq.RealWienerProcess(0.0, 0.0)
        else
            noise_ = StochasticDiffEq.RealWienerProcess!(0.0, zeros(n + nc))
        end
    else
        noise_ = noise
    end
    if isa(noise_rate_prototype, Nothing)
        if n > 1 || nc > 1 || (n > 0 && nc > 0)
            noise_rate_prototype = zeros(eltype(x0), length(x0), n + nc)
        end
    end

    out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(state)})

    out = DiffEqCallbacks.SavedValues(eltype(tspan),out_type)

    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=saveat,
                                         save_everystep=save_everystep,
                                         save_start = false,
                                         tdir = first(tspan)<last(tspan) ? one(eltype(tspan)) : -one(eltype(tspan)))

    full_cb = OrdinaryDiffEqCore.CallbackSet(callback, ncb, scb)

    prob = StochasticDiffEq.SDEProblem{true}(df_, dg_, x0,(tspan[1],tspan[end]),
                    noise=noise_,
                    noise_rate_prototype=noise_rate_prototype)

    sol = StochasticDiffEq.solve(
                prob,
                alg;
                reltol = 1.0e-3,
                abstol = 1.0e-3,
                save_everystep = false, save_start = false,
                save_end = false,
                callback=full_cb, save_noise=save_noise, kwargs...)
    if save_noise
        return out.t, out.saveval, sol.W
    end
    return out.t, out.saveval
end

"""
    integrate_stoch

Define fout if it was omitted.
"""
function integrate_stoch(tspan, df, dg, x0,
    state, dstate, ::Nothing, n; kwargs...)
    function fout(t, state)
        copy(state)
    end
    integrate_stoch(tspan, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function _check_const(op)
    if !QuantumOpticsBase.is_const(op)
        throw(
          ArgumentError("You are attempting to use a time-dependent dynamics generator " *
            "(a Hamiltonian or Lindbladian) with a solver that assumes constant " *
            "dynamics. To avoid errors, please use the _dynamic solvers instead, " *
            "e.g. schroedinger_dynamic instead of schroedinger")
        )
    end
    nothing
end

DiffEqNoiseProcess.wiener_randn!(rng::AbstractRNG,rand_vec::T) where {T<:Union{Bra,Ket,Operator}} = randn!(rng,rand_vec.data)
