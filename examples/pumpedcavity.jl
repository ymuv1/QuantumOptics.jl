require("../src/quantumoptics.jl")

using quantumoptics

basis = FockBasis(20)
Ψ₀ = basis_ket(basis, 1)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)
a = destroy(basis)
aᵀ = create(basis)
n = number(basis)

η = 0.9
κ = 1.

H = η*(a+aᵀ)
J = [sqrt(κ)*a]
T = [0,10]

tout, ρ = timeevolution_simple.master(T, ρ₀, H, J)

exp_n = expect(n, ρ)
exp_a = expect(a, ρ)

0