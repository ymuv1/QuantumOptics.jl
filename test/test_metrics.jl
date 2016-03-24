using Base.Test
using QuantumOptics

b = SpinBasis(1//2)

psi1 = spinup(b)
psi2 = spindown(b)

rho = tensor(psi1, dagger(psi1))
sigma = tensor(psi2, dagger(psi2))

@test tracedistance(rho, sigma) == 1.
@test tracedistance(rho, rho) == 0.
@test tracedistance(sigma, sigma) == 0.
