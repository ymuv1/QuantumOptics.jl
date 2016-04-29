module timeevolution_schroedinger

export schroedinger

using ...bases
using ...operators
using ...states
using ...ode_dopri

"""
Integrate Schroedinger equation with given derivative function.
"""
function integrate_schroedinger{T<:StateVector}(dschroedinger::Function, tspan, psi0::T; fout=nothing, kwargs...)
    as_statevector(x::Vector{Complex128}) = T(psi0.basis, x)
    as_vector(psi::T) = psi.data
    if fout==nothing
        tout = Float64[]
        xout = T[]
        function fout_(t, psi::T)
            push!(tout, t)
            push!(xout, deepcopy(psi))
        end
        f = fout_
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_statevector(x))
    dschroedinger_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dschroedinger(t, as_statevector(x), as_statevector(dx))
    ode(dschroedinger_, float(tspan), as_vector(psi0), f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

"""
Evaluate Schroedinger equation for ket states.
"""
function dschroedinger_ket(psi::Ket, H::Operator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    return dpsi
end

"""
Evaluate Schroedinger equation for bra states.
"""
function dschroedinger_bra(psi::Bra, H::Operator, dpsi::Bra)
    operators.gemv!(complex(0,1.), psi, H, complex(0.), dpsi)
    return dpsi
end

"""
Integrate Schroedinger equation.

Arguments
---------
tspan
    Vector specifying the points of time for which output should be displayed.
psi0
    Initial state vector (can be a bra or a ket).
H
    DenseOperator specifying the Hamiltonian.

Keyword Arguments
-----------------

fout (optional)
    If given this function fout(t, psi) is called every time an output should
    be displayed. To limit copying to a minimum the given state psi
    is further used and therefore must not be changed.
"""
function schroedinger{T<:StateVector}(tspan, psi0::T, H::Operator;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    if T==Ket
        @assert psi0.basis==H.basis_l
        @assert multiplicable(H.basis_r, psi0.basis)
        dschroedinger_(t, psi::Ket, dpsi::Ket) = dschroedinger_ket(psi, H, dpsi)
    elseif T==Bra
        @assert psi0.basis==H.basis_r
        @assert multiplicable(psi0.basis, H.basis_l)
        dschroedinger_(t, psi::Bra, dpsi::Bra) = dschroedinger_bra(psi, H, dpsi)
    end
    return integrate_schroedinger(dschroedinger_, tspan, psi0; fout=fout, kwargs...)
end

end