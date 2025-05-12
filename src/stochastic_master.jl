import ...timeevolution: dmaster_h!, dmaster_nh!, dmaster_h_dynamic!, dmaster_nh_dynamic!, check_master, nh_hamiltonian, master_nh_dynamic_function, master_nh_dynamic_function, master_h_dynamic_function, master_stochastic_dynamics_function, view_recast!, as_vector
import QuantumOpticsBase:axpy!
"""
        stochastic.master(tspan, rho0, H, J, C; <keyword arguments>)

    Time-evolution according to a stochastic master equation.

    For dense arguments the `master` function calculates the
    non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

    # Arguments
    * `tspan`: Vector specifying the points of time for which output should
            be displayed.
    * `rho0`: Initial density operator. Can also be a state vector which is
            automatically converted into a density operator.
    * `H`: Deterministic part of the Hamiltonian.
    * `J`: Vector containing all deterministic
            jump operators which can be of any arbitrary operator type.
    * `C`: Vector containing the stochastic operators for a superoperator
            of the form `C[i]*rho + rho*Cdagger[i]`.
    * `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
            for the jump operators. If nothing is specified all rates are assumed
            to be 1.    RECOMMEND NOT TO USE THIS: IT IS APPLIED ONLY TO Js EVEN THOUGH Cs SHOULD BE FUNCTIONS OF Js, AND YOU CAN ALWAYS USE A BASIS WHERE RATE MATRIX IS DIAGONAL AND REDEFINE J[i]<-sqrt(rates[i]*J[i]).
    * `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
            operators. If they are not given they are calculated automatically.
    * `fout=nothing`: If given, this function `fout(t, rho)` is called every time
            an output should be displayed. ATTENTION: The given state rho is not
            permanent! It is still in use by the ode solver and therefore must not
            be changed.
    * `kwargs...`: Further arguments are passed on to the ode solver.
    """
function master(tspan, rho0::T, H::AbstractOperator{B,B},
                J, C;
                rates=nothing,
                Jdagger=dagger.(J), Cdagger=dagger.(C),
                fout=nothing, save_noise=false,
                kwargs...) where {B,T<:Operator{B,B}}
    _check_const(H)
    _check_const.(J)
    _check_const.(Jdagger)
    tmp = copy(rho0)

    n = length(C)

    dmaster_stoch_(dx, t, rho, drho, n) = dmaster_stochastic!(dx, rho, C, Cdagger, drho, n)

    isreducible = check_master(rho0, H, J, Jdagger, rates) && check_master_stoch(rho0, C, Cdagger)
    if !isreducible
        dmaster_h_determ_(t, rho, drho) =
            dmaster_h!(drho, H, J, Jdagger, rates, rho, tmp)
        integrate_master_stoch(tspan, dmaster_h_determ_, dmaster_stoch, rho0, fout, n; kwargs...)
    else
        Hnh = copy(H)
        if isa(rates, AbstractMatrix)
            for i=1:length(J), j=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im*rates[i,j])*Jdagger[i]*J[j]
            end
        elseif isa(rates, AbstractVector)
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im*rates[i])*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im)*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)

        dmaster_nh_determ(t, rho, drho) =
            dmaster_nh!(drho, Hnh, Hnhdagger, J, Jdagger, rates, rho, tmp)
        integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch, rho0, fout, n; save_noise=save_noise, kwargs...)
    end
end
master(tspan, psi0::Ket, args...; kwargs...) = master(tspan, dm(psi0), args...; kwargs...)

"""
    stochastic.master_h_dynamic(tspan, rho0, fdeterm, fstoch; <keyword arguments>)

Time-evolution according to a stochastic master equation with a
dynamic Hamiltonian and J. 

# Notes:
- This version uses a Hermitian Hamiltonian, which is often less efficient. Consider using stochastic.master_nh_dynamic instead.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `fdeterm`: Function `f(t, rho) -> (H, J, Jdagger)` or
        `f(t, rho) -> (H, J, Jdagger, rates)` giving the deterministic
        part of the master equation.
* `fstoch`: Function `f(t, rho) -> (C, Cdagger)` giving the stochastic
        superoperator of the form `C[i]*rho + rho*Cdagger[i]`.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.    RECOMMEND NOT TO USE THIS: IT IS APPLIED ONLY TO Js EVEN THOUGH Cs SHOULD BE FUNCTIONS OF Js, AND YOU CAN ALWAYS USE A BASIS WHERE RATE MATRIX IS DIAGONAL AND REDEFINE J[i]<-sqrt(rates[i]*J[i]).
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `noise_processes=0`: Number of distinct white-noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch` and all optional functions. If unset, the number
        is calculated automatically from the function outputs. NOTE: Set this
        number if you want to avoid an initial calculation of function outputs!
*  `save_noise`: whether to return the noise dW in the same format as in DifferentialEquations.jl
* `kwargs...`: Further arguments are passed on to the ode solver.

        stochastic.master_h_dynamic(tspan, rho0, H::AbstractTimeDependentOperator, J, C; <keyword arguments>)

In this version simply input operators for the time dependent Hamiltonian `H` and optionally time-dependent jump operators `J` and measurement (collapse) operators `C`).
"""
function master_h_dynamic(tspan, rho0::T, fdeterm, fstoch;
                rates=nothing,
                fout=nothing,
                noise_processes::Int=0,
                kwargs...) where {B,T<:Operator{B,B}}

    tmp = copy(rho0)

    if noise_processes == 0
        fs_out = fstoch(tspan[1], rho0)
        n = length(fs_out[1])
    else #why do we even need this as an argument? I see no reason that we won't simply want what we get in the `if` clause. TODO: Consider removing.
        n = noise_processes
    end

    dmaster_determ_(t, rho, drho) = dmaster_h_dynamic!(drho, fdeterm, rates, rho, tmp, t)
    dmaster_stoch_(dx, t, rho, drho, n) = dmaster_stoch_dynamic!(dx, t, rho, fstoch, drho, n)
    integrate_master_stoch(tspan, dmaster_determ_, dmaster_stoch_, rho0, fout, n; save_noise=save_noise, kwargs...)
end

function master_h_dynamic(tspan, rho0::T, H::AbstractTimeDependentOperator, J, C;
    rates=nothing,
    fout=nothing,
    save_noise=false,
    kwargs...) where {B,T<:Operator{B,B}}

    fdeterm_ = master_h_dynamic_function(H, J)
    fstoch_ = master_stochastic_dynamics_function(C)
    master_h_dynamic(tspan, rho0::T, fdeterm_, fstoch_; save_noise=false, fout=fout, kwargs...)
end

"""
        stochastic.master_nh_dynamic(tspan, rho0, fdeterm, fstoch; <keyword arguments>)

    Time-evolution according to a stochastic master equation with a
    dynamic non-Hermitian Hamiltonian, J and C. 

    # Arguments
    * `tspan`: Vector specifying the points of time for which output should be displayed.
    * `rho0`: Initial density operator. Can also be a state vector which is
            automatically converted into a density operator.
    * `fdeterm`: Function `f(t, rho) -> (Hnh, Hnh_dagger, J, Jdagger)` or
            `f(t, rho) -> (Hnh, Hnh_dagger, J, Jdagger, rates)` used to calculate the deterministic
            part of the master equation.
    * `fstoch`: Function `f(t, rho) -> (C, Cdagger)` used to calculate the stochastic
            part of the master equation `C[i]*rho + rho*Cdagger[i]`.
    * `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
            for the jump operators. If nothing is specified all rates are assumed
            to be 1.    RECOMMEND NOT TO USE THIS: IT IS APPLIED ONLY TO Js EVEN THOUGH Cs SHOULD BE FUNCTIONS OF Js, AND YOU CAN ALWAYS USE A BASIS WHERE RATE MATRIX IS DIAGONAL AND REDEFINE J[i]<-sqrt(rates[i]*J[i]).
    * `fout=nothing`: If given, this function `fout(t, rho)` is called every time
            an output should be displayed. ATTENTION: The given state rho is not
            permanent! It is still in use by the ode solver and therefore must not
            be changed.
    * `noise_processes=0`: Number of distinct white-noise processes in the equation.
            This number has to be equal to the total number of noise operators
            returned by `fstoch` and all optional functions. If unset, the number
            is calculated automatically from the function outputs. NOTE: Set this
            number if you want to avoid an initial calculation of function outputs!
    * `kwargs...`: Further arguments are passed on to the ode solver.        
    """
function master_nh_dynamic(tspan, rho0::T, fdeterm, fstoch;
    rates=nothing,
    fout=nothing,
    save_noise=false,
    noise_processes::Int=0,
    kwargs...) where {B,T<:Operator{B,B}}

    tmp = copy(rho0)

    if noise_processes == 0
    fs_out = fstoch(tspan[1], rho0)
    n = length(fs_out[1])
    else #why do we even need this as an argument? I see no reason that we won't simply want what we get in the `if` clause. TODO: Consider removing.
    n = noise_processes
    end

    dmaster_determ_(t, rho, drho) = dmaster_nh_dynamic!(drho, fdeterm, rates, rho, tmp, t)
    dmaster_stoch(dx, t, rho, drho, n) = dmaster_stoch_dynamic!(dx, t, rho, fstoch, drho, n)
    integrate_master_stoch(tspan, dmaster_determ_, dmaster_stoch, rho0, fout, n; save_noise=save_noise, kwargs...)
end

"""
Recommended: use Hnh=nh_hamiltonian(H, J)
set `save_noise=true` to obtain a vector (matrix) dW in addition to the regular output
"""
function master_nh_dynamic(tspan, rho0::T, Hnh::AbstractTimeDependentOperator, J, C;
    fout = nothing,  save_noise=false, kwargs...)  where {B,T<:Operator{B,B}}
    fdeterm = master_nh_dynamic_function(Hnh, J)
    fstoch = master_stochastic_dynamics_function(C)
    master_nh_dynamic(tspan, rho0, fdeterm, fstoch; fout=fout, save_noise=save_noise, kwargs...)
end


master_h_dynamic(tspan, psi0::Ket, args...; kwargs...) = master_h_dynamic(tspan, dm(psi0), args...; kwargs...)
master_nh_dynamic(tspan, psi0::Ket, args...; kwargs...) = master_nh_dynamic(tspan, dm(psi0), args...; kwargs...)


# Derivative functions
#YM: In DifferentialEquations.jl's the state u and its differential du must be vectors and not matrices like ρ.data. Since we need matrix forms
#    to calculate the differential (when deriving matrix products) we need to combine both methods.
#    originally, recast! was used to copy the data back and forth. We will use a more efficient version by using views, so there will be the same data
#    referenced by both a matrix object and a vector object.
# In DifferentialEquations.jl, SDEs are typically taken to be of the form `du = f(u,p,t)dt + g(u,p,t)dW`. The user defines f, g.
# To simulate a more general SDE `du = f(u,p,t)dt + ∑ᵢgᵢ(t,u,p)dWᵢ`, we must use a function g that returns a matrix where each column is
# gᵢ. This matrix is the dx here, and the columns are the dx_i.
#

# Since the calculation of each timestep requires the matrix form (for matrix multiplication) we can either

# TODO: Code is currently optimized for only dense matrices. Define several functions such that:
#   axpy! is used instead of .-= tr(dρ)*ρ iff both matrices are dense
#   second mul! instead of drho .+= drho' iff Cdagger is sparse (both mathematically and in its type) and dρ,ρ are dense
#   This is the conclusion of benchmarking for operators that are sparse, dense, pseudo-sparse (sparse-types with mathematically dense matrices) and pseudo-dense
function dmaster_stochastic!(dx::AbstractMatrix, rho, C, Cdagger, drho, n)
    for i=1:n
        dx_i = @view dx[:, i]
        view_recast!(drho, dx_i)
        QuantumOpticsBase.mul!(drho,C[i],rho)
        # QuantumOpticsBase.mul!(drho,rho,Cdagger[i],true,true)
        drho.data .+= drho.data' #might be slightly less efficient than commented line if C is sparse (both mathematically and in its type). But in the other cases this is MUCH more efficient.
        # drho.data .-= tr(drho)*rho.data #allocations!
        axpy!(-tr(drho.data), rho.data, drho.data) # much slower than drho.data .-= tr(drho)*rho.data  if any of the matrices is a sparse object
    end
    return nothing
end

# If dx is a vector, this means that there should be only one collapse operator (since the 'g function' of the SDE returns a vector and not a matrix).
# TODO: Code is currently optimized for only dense matrices. Define several functions such that:
#   axpy! is used instead of .-= tr(dρ)*ρ iff both matrices are dense
#   second mul! instead of drho .+= drho' iff Cdagger is sparse (both mathematically and in its type) and dρ,ρ are dense
function dmaster_stochastic!(dx::AbstractVector, rho, C, Cdagger, drho, n)
    view_recast!(drho, dx)
    QuantumOpticsBase.mul!(drho,C[1],rho)
    # QuantumOpticsBase.mul!(drho,rho,Cdagger[1],true,true)
    drho.data .+= drho.data'
    # drho.data .-= tr(drho)*rho.data #allocations!
    axpy!(-tr(drho), rho.data, drho.data) # much slower than drho.data .-= tr(drho)*rho.data  if any of the matrices is a sparse object
    return nothing
end


function dmaster_stoch_dynamic!(dx, t, rho, f, drho, n)
    result = f(t, rho)
    QO_CHECKS[] && @assert 2 == length(result)
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(rho, C, Cdagger)
    dmaster_stochastic!(dx, rho, C, Cdagger, drho, n)
end

function integrate_master_stoch(tspan, df, dg,
                        rho0, fout,
                        n; save_noise=false,
                        kwargs...)
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    state = copy(rho0)
    dstate = copy(rho0)
    x0 = as_vector(state)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; save_noise=save_noise, kwargs...)
end

function check_master_stoch(rho0::Operator{B,B}, C, Cdagger) where B
    # TODO: replace type checks by dispatch; make types of C known
    @assert length(C) == length(Cdagger)
    isreducible = true
    for c=C
        @assert isa(c, AbstractOperator{B,B})
        if !isa(c, DataOperator)
            isreducible = false
        end
    end
    for c=Cdagger
        @assert isa(c, AbstractOperator{B,B})
        if !isa(c, DataOperator)
            isreducible = false
        end
    end
    isreducible
end