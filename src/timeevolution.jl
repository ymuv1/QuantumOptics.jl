module timeevolution

using ..bases
using ..operators
using ..states
using ..ode_dopri

export master, master_nh, master_h


"""
Evaluate master equation for diagonal jump operators.
"""
function dmaster_h(rho::Operator, H::AbstractOperator,
                    Gamma::Vector{Complex128}, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for i=1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[i], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i], rho, Jdagger[i], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for nondiagonal jump operators.
"""
function dmaster_h(rho::Operator, H::AbstractOperator,
                    Gamma::Matrix{Complex128}, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[j], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i,j], rho, Jdagger[j], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for non-hermitian Hamiltonian and diagonal jump operators.
"""
function dmaster_nh(rho::Operator, Hnh::AbstractOperator, Hnh_dagger::AbstractOperator,
                    Gamma::Vector{Complex128}, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for i=1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for non-hermitian Hamiltonian and nondiagonal jump operators.
"""
function dmaster_nh(rho::Operator, Hnh::AbstractOperator, Hnh_dagger::AbstractOperator,
                    Gamma::Matrix{Complex128}, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)
    end
    return drho
end

"""
Integrate master equation with given derivative function.
"""
function integrate_master(dmaster::Function, tspan, rho0::Operator; fout=nothing, kwargs...)
    nl = prod(rho0.basis_l.shape)
    nr = prod(rho0.basis_r.shape)
    N = nl*nr
    as_operator(x::Vector{Complex128}) = Operator(rho0.basis_l, rho0.basis_r, reshape(x, nl, nr))
    as_vector(rho::Operator) = reshape(rho.data, N)
    f = (x->x)
    if fout==nothing
        tout = Float64[]
        xout = Operator[]
        function fout_(t, rho::Operator)
            push!(tout, t)
            push!(xout, deepcopy(rho))
        end
        f = fout_
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_operator(x))
    dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmaster(t, as_operator(x), as_operator(dx))
    ode(dmaster_, float(tspan), as_vector(rho0), f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function _check_input(rho0::Operator, H::AbstractOperator, J::Vector, Jdagger::Vector, Gamma::Union{Vector{Float64}, Matrix{Float64}})
    operators.check_samebases(rho0, H)
    for j=J
        @assert typeof(j) <: AbstractOperator
        operators.check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert typeof(j) <: AbstractOperator
        operators.check_samebases(rho0, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(Gamma) == Matrix{Float64}
        @assert size(Gamma, 1) == size(Gamma, 2) == length(J)
    elseif typeof(Gamma) == Vector{Float64}
        @assert length(Gamma) == length(J)
    else
        error()
    end
end

"""
Integrate master equation with dmaster_h as derivative function.
"""
function master_h(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master_h(tspan, psi0::Ket, H::AbstractOperator, J::Vector; kwargs...) = master_h(tspan, tensor(psi0, dagger(psi0)), H, J; kwargs...)


"""
Integrate master equation with master_nh as derivative function.
"""
function master_nh(tspan, rho0::Operator, Hnh::AbstractOperator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Hnhdagger::AbstractOperator=dagger(Hnh),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, Hnh, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector; kwargs...) = master_nh(tspan, tensor(psi0, dagger(psi0)), Hnh, J; kwargs...)


"""
Integrate master equation.

Hnh is first calculated from the given Hamiltonian and Jump operators and
then dmaster_nh is used for the time evolution.
"""
function master(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

function master(tspan, rho0::Operator, H::Operator, J::Vector{Operator};
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector{Operator}=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Hnh = deepcopy(H)
    if typeof(Gamma) == Matrix{Float64}
        for i=1:length(J), j=1:length(J)
            Hnh -= 0.5im*Gamma[i,j]*Jdagger[i]*J[j]
        end
    elseif typeof(Gamma) == Vector{Float64}
        for i=1:length(J)
            Hnh -= 0.5im*Gamma[i]*Jdagger[i]*J[i]
        end
    else
        error()
    end
    Hnhdagger = dagger(Hnh)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master(tspan, psi0::Ket, H::AbstractOperator, J::Vector; kwargs...) = master(tspan, tensor(psi0, dagger(psi0)), H, J; kwargs...)


"""
Integrate schroedinger equation with given derivative function.
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
Evaluate schroedinger equation for ket states.
"""
function dschroedinger_ket(psi::Ket, H::AbstractOperator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    return dpsi
end

"""
Evaluate schroedinger equation for bra states.
"""
function dschroedinger_bra(psi::Bra, H::AbstractOperator, dpsi::Bra)
    operators.gemv!(complex(0,1.), psi, H, complex(0.), dpsi)
    return dpsi
end

"""
Integrate schroedinger equation.
"""
function schroedinger{T<:StateVector}(tspan, psi0::T, H::AbstractOperator;
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

function integrate_mcwf(dmcwf::Function, jumpfun::Function, tspan, psi0::Ket, seed::UInt64;
                fout=nothing,
                kwargs...)
    tmp = deepcopy(psi0)
    as_ket(x::Vector{Complex128}) = Ket(psi0.basis, x)
    as_vector(psi::Ket) = psi.data
    rng = MersenneTwister(convert(UInt64, seed))
    jumpnorm = Float64[rand(rng)]
    djumpnorm(t, x::Vector{Complex128}) = norm(as_ket(x))^2 - (1-jumpnorm[1])
    function dojump(t, x::Vector{Complex128})
        jumpfun(rng, t, as_ket(x), tmp)
        for i=1:length(x)
            x[i] = tmp.data[i]
        end
        jumpnorm[1] = rand(rng)
        return ode_dopri.jump
    end
    if fout==nothing
        tout = Float64[]
        xout = Ket[]
        function fout_(t, x::Vector{Complex128})
            psi = deepcopy(as_ket(x))
            psi /= norm(psi)
            push!(tout, t)
            push!(xout, psi)
            nothing
        end
    else
        fout_(t, x::Vector{Complex128}) = fout(t, as_ket(x))
    end
    dmcwf_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmcwf(t, as_ket(x), as_ket(dx))
    ode_event(dmcwf_, float(tspan), as_vector(psi0), fout_,
        djumpnorm, dojump;
        kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket)
    if length(J)==1
        operators.gemv!(complex(1.), J[1], psi, complex(0.), psi_new)
        N = norm(psi_new)
        for i=1:length(psi_new.data)
            psi_new.data[i] /= N
        end
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            operators.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
            #probs[i] = norm(psi_new)^2
            probs[i] = dagger(psi_new)*psi_new
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        operators.gemv!(complex(1.)/sqrt(probs[i]), J[i], psi, complex(0.), psi_new)
    end
    return nothing
end

"""
Evaluate Schroedinger equation with nonhermitian Hamiltonian.

The nonhermitian Hamiltonian is given in two parts - the hermition part H and
the jump operators J.
"""
function dmcwf_h(psi::Ket, H::AbstractOperator,
                 J::Vector, Jdagger::Vector, dpsi::Ket, tmp::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    for i=1:length(J)
        operators.gemv!(complex(1.), J[i], psi, complex(0.), tmp)
        operators.gemv!(-complex(0.5,0.), Jdagger[i], tmp, complex(1.), dpsi)
    end
    return dpsi
end


"""
Evaluate Schroedinger equation with nonhermitian Hamiltonian.
"""
function dmcwf_nh(psi::Ket, Hnh::AbstractOperator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return dpsi
end

"""
Integrate master equation using MCWF method with mcwf_h as derivative function.
"""
function mcwf_h(tspan, psi0::Ket, H::AbstractOperator, J::Vector;
                seed=rand(UInt64), fout=nothing, Jdagger::Vector=map(dagger, J),
                tmp::Ket=deepcopy(psi0),
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    f(t, psi, dpsi) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...)
end

"""
Integrate master equation using MCWF method with mcwf_nh as derivative function.
"""
function mcwf_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector;
                seed=rand(UInt64), fout=nothing,
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...)
end

"""
Integrate master equation using MCWF method.

Hnh is first calculated from the given Hamiltonian and Jump operators and
then dmcwf_nh is used for the time evolution.
"""
function mcwf(tspan, psi0::Ket, H::AbstractOperator, J::Vector;
                seed=rand(UInt64), fout=nothing, Jdagger::Vector=map(dagger, J),
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    Hnh = deepcopy(H)
    for i=1:length(J)
        Hnh -= 0.5im*Jdagger[i]*J[i]
    end
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed;
                fout=fout,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...)
end

end # module
