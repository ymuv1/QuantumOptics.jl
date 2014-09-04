module timeevolution_simple

using ..operators
using ODE

export master


function dmaster(rho::Operator, H::AbstractOperator, J::Vector, Jdagger::Vector)
    drho = -1im * (H*rho - rho*H)
    for n = 1:length(J)
        drho = drho + J[n]*rho*Jdagger[n] - Jdagger[n]*(J[n]*rho)/Complex(2) - rho*Jdagger[n]*J[n]/Complex(2)
    end
    return drho
end

function master(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector; Jdagger=map(dagger,J))
    f(t::Number,rho::AbstractOperator) = dmaster(rho, H, J, Jdagger)
    tout, rho_t = ode45(f, rho0, T)
    return tout, rho_t
end

function dmaster_nondiag(rho::Operator, H::AbstractOperator, gamma::Matrix, J::Vector, Jdagger::Vector)
	drho = -1im * (H*rho - rho*H)
	for m=1:length(J), n=1:length(J)
		drho += gamma[m,n]*(J[m]*rho*dagger(J[n]) - Jdagger[n]*(J[m]*rho)/Complex(2) - rho*Jdagger[n]*J[m]/Complex(2))
	end
	return drho
end

function master_nondiag(T::Vector, rho0::Operator, H::AbstractOperator, gamma::Matrix, J::Vector; Jdagger=map(dagger,J))
	f(t::Number,rho::AbstractOperator) = dmaster_nondiag(rho, H, gamma, J, Jdagger)
    tout, rho_t = ode45(f, rho0, T)
    return tout, rho_t
end

end