module timeevolution

using ..operators
using ..states
using ..ode_dopri2

export master

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

function integrate_master(dmaster::Function, tspan, rho0; fout=nothing, kwargs...)
    n = size(rho0.data, 1)
    N = n^2
    as_operator(x::Vector{Complex128}) = Operator(rho0.basis_l, rho0.basis_r, reshape(x, n, n))
    as_vector(rho::Operator) = reshape(rho.data, N)
    if fout==nothing
        tout = Float64[]
        xout = Operator[]
        function f(t, rho)
            push!(tout, t)
            push!(xout, rho)
        end
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_operator(x))
    dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmaster(t, as_operator(x), as_operator(dx))
    ode(dmaster_, float(tspan), as_vector(rho0), fout=f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function master_nh(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma=[Complex(1.) for i=1:length(J)],
                Hdagger::AbstractOperator=dagger(H),
                Jdagger::Vector=map(dagger, J),
                fout=nothing,
                tmp::Operator=deepcopy(rho0), kwargs...)
    Gamma = complex(Gamma)
    dmaster_(t, rho::Operator, drho::Operator) = dmaster_nh(rho, H, Hdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end




function integrate_dopri_mcwf(dmaster::Function, jumpfun::Function, tspan, psi0; fout=nothing, kwargs...)
    if fout==nothing
        tout = Float64[]
        xout = Ket[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Ket(psi0.basis, 1.*x))
            nothing
        end
        ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=f; kwargs...)
        return tout, xout
    else
        ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=fout; kwargs...)
        return nothing
    end
end

function integrate_dopri_mcwf2(dmaster::Function, jumpfun::Function, tspan, psi0; fout=nothing, kwargs...)
    if fout==nothing
        tout = Float64[]
        xout = Ket[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Ket(psi0.basis, 1.*x))
            nothing
        end
        ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=f; kwargs...)
        return tout, xout
    else
        ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=fout; kwargs...)
        return nothing
    end
end


function jump{T<:Complex}(rng, t::Float64, psi::Vector{T}, J::Vector, psi_new::Vector{T})
    probs = zeros(Float64, length(J))
    for i=1:length(J)
        operators.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
        probs[i] = vecnorm(psi_new)
    end
    cumprobs = cumsum(probs./sum(probs))
    r = rand(rng)
    i = findfirst(cumprobs.>r)
    operators.gemv!(complex(1.)/probs[i], J[i], psi, complex(0.), psi_new)
    return nothing
end

function dmcwf_nh{T<:Complex}(psi::Vector{T}, Hnh, dpsi::Vector{T})
    operators.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return psi
end

function mcwf_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector;
                fout=nothing, kwargs...)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_dopri_mcwf(f, j, tspan, psi0, fout=fout; kwargs...)
end

end