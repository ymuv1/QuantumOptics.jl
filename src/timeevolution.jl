module timeevolution

using ..bases
using ..operators
using ..states
using ..ode_dopri

export master, master_nh, master_h


function dmaster_h(rho::Operator, H::AbstractOperator,
                    Gamma::Vector, J::Vector, Jdagger::Vector,
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

function dmaster_h(rho::Operator, H::AbstractOperator,
                    Gamma::Matrix, J::Vector, Jdagger::Vector,
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

function dmaster_nh(rho::Operator, Hnh::AbstractOperator, Hnh_dagger::AbstractOperator,
                    Gamma::Vector, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for i=1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

function dmaster_nh(rho::Operator, Hnh::AbstractOperator, Hnh_dagger::AbstractOperator,
                    Gamma::Matrix, J::Vector, Jdagger::Vector,
                    drho::Operator, tmp::Operator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)
    end
    return drho
end

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
    ode(dmaster_, float(tspan), as_vector(rho0), fout=f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function _check_input(rho0::Operator, H::AbstractOperator, J::Vector, Jdagger::Vector, Gamma)
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
    if typeof(Gamma)<:Real
        Gamma = ones(typeof(Gamma), length(J))*Gamma
    end
end

function master_h(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma::Union(Real, Vector, Matrix)=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union(Function,Nothing)=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

function master_nh(tspan, rho0::Operator, Hnh::AbstractOperator, J::Vector;
                Gamma::Union(Real, Vector, Matrix)=ones(Float64, length(J)),
                Hnhdagger::AbstractOperator=dagger(Hnh),
                Jdagger::Vector=map(dagger, J),
                fout::Union(Function,Nothing)=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, Hnh, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

function master(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma::Union(Real, Vector, Matrix)=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union(Function,Nothing)=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    Hnh = deepcopy(H)
    for i=1:length(J)
        Hnh -= 0.5im*Jdagger[i]*J[i]
    end
    Hnhdagger = dagger(Hnh)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

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
    ode(dschroedinger_, float(tspan), as_vector(psi0), fout=f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function dschroedinger_ket(psi::Ket, H::AbstractOperator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    return dpsi
end

function dschroedinger_bra(psi::Bra, H::AbstractOperator, dpsi::Bra)
    operators.gemv!(complex(0,1.), psi, H, complex(0.), dpsi)
    return dpsi
end

function schroedinger{T<:StateVector}(tspan, psi0::T, H::AbstractOperator;
                fout::Union(Function,Nothing)=nothing,
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

function integrate_mcwf(dmaster::Function, jumpfun::Function, tspan, psi0::Ket, seed::Uint64;
                fout=nothing,
                kwargs...)
    tmp = deepcopy(psi0)
    as_ket(x::Vector{Complex128}) = Ket(psi0.basis, x)
    as_vector(psi::Ket) = psi.data
    rng = MersenneTwister(convert(Uint, seed))
    jumpnorm = rand(rng)
    djumpnorm(t, x::Vector{Complex128}) = vecnorm(x) - jumpnorm
    function dojump(t, x::Vector{Complex128})
        jumpfun(rng, t, as_ket(x), tmp)
        for i=1:length(x)
            x[i] = tmp.data[i]
        end
        return "jump"
    end
    if fout==nothing
        tout = Float64[]
        xout = Ket[]
        function f(t, psi::Ket)
            push!(tout, t)
            push!(xout, deepcopy(psi))
            nothing
        end
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_ket(x))
    dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmaster(t, as_ket(x), as_ket(dx))
    ode(dmaster_, float(tspan), as_vector(psi0), fout=f_;
        event_locator=djumpnorm, event_callback=dojump,
        kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket)
    probs = zeros(Float64, length(J))
    for i=1:length(J)
        operators.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
        probs[i] = vecnorm(psi_new.data)
    end
    cumprobs = cumsum(probs./sum(probs))
    r = rand(rng)
    i = findfirst(cumprobs.>r)
    operators.gemv!(complex(1.)/probs[i], J[i], psi, complex(0.), psi_new)
    return nothing
end

function dmcwf_nh(psi::Ket, Hnh::AbstractOperator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return psi
end

function mcwf_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector;
                seed=rand(Uint64), fout=nothing, kwargs...)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout, kwargs...)
end

function mcwf(tspan, psi0::Ket, H::AbstractOperator, J::Vector;
                seed=rand(Uint64), fout=nothing, Jdagger::Vector=map(dagger, J), kwargs...)
    Hnh = deepcopy(H)
    for i=1:length(J)
        Hnh -= 0.5im*Jdagger[i]*J[i]
    end
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout, kwargs...)
end

end # module
