module timeevolution_simple

using ..operators
using ODE

export master


function dmaster(rho::Operator, H::AbstractOperator, J::Vector)
    drho = -1im * (H*rho - rho*H)
    for j = J
        drho = drho + j*rho*dagger(j) - dagger(j)*(j*rho)/Complex(2) - rho*dagger(j)*j/Complex(2)
    end
    return drho
end

function master(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector)
    f(t::Number,rho::AbstractOperator) = dmaster(rho, H, J)
    tout, rho_t = ode45(f, rho0, T)
    return tout, rho_t
end

end