module timeevolution_simple

using ..states
using ..operators
using ODE

export master


function dmaster(rho::Operator, H::AbstractOperator, gamma::Vector, J::Vector, Jdagger::Vector)
    drho = -1im * (H*rho - rho*H)
    for n = 1:length(J)
        drho = drho + gamma[n]*(J[n]*rho*Jdagger[n] - Jdagger[n]*(J[n]*rho)/Complex(2) - rho*Jdagger[n]*J[n]/Complex(2))
    end
    return drho
end

function dmaster(rho::Operator, H::AbstractOperator, gamma::Matrix, J::Vector, Jdagger::Vector)
    drho = -1im * (H*rho - rho*H)
    for m=1:length(J), n=1:length(J)
       drho += gamma[m,n]*(J[m]*rho*Jdagger[n] - Jdagger[n]*(J[m]*rho)/Complex(2) - rho*Jdagger[n]*J[m]/Complex(2))
    end
    return drho
end

function master(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector;
                    Jdagger=map(dagger,J),
                    gamma::Union(Real, Vector, Matrix)=ones(Int, length(J)),
                    kwargs...)
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
    if typeof(gamma)<:Real
        gamma = ones(typeof(gamma), length(J))*gamma
    end
    f(t::Float64, rho::Operator) = dmaster(rho, H, gamma, J, Jdagger)
    tout, rho_t = ode45(f, rho0, T; kwargs...)
    return tout, rho_t
end

function dschroedinger(psi::Ket, H::AbstractOperator)
    return -1im*H*psi
end

function dschroedinger(psi::Bra, H::AbstractOperator)
    return 1im*psi*H
end

function schroedinger(T::Vector, psi0::StateVector, H::AbstractOperator; kwargs...)
    f(t::Float64, psi::StateVector) = dschroedinger(psi, H)
    tout, rho_t = ode45(f, psi0, T; kwargs...)
    return tout, rho_t
end

end  # module
