using quantumoptics

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

T = Float64[0.,1.]


fockbasis = FockBasis(10)
basis = compose(spinbasis, fockbasis)

Ha = embed(basis, 1, 0.5*ωa*sigmaz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sigmam ⊗ create(fockbasis) + sigmap ⊗ destroy(fockbasis)
H = SparseOperator(Ha + Hc + Hint)

Ja = embed(basis, 1, sqrt(γ)*sigmam)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [SparseOperator(Ja), SparseOperator(Jc)]

Ψ₀ = basis_ket(spinbasis, 2) ⊗ basis_ket(fockbasis, 1)
ρ₀ = Ψ₀⊗dagger(Ψ₀)


tout, ρt = timeevolution.master(T, ρ₀, H, J)


