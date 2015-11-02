using Base.Test
using quantumoptics

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

T = Float64[0.,1.]


fockbasis = FockBasis(10)
spinbasis = SpinBasis(1//2)
basis = compose(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint
Hsparse = SparseOperator(H)

Ja = embed(basis, 1, sqrt(γ)*sm)
Ja2 = embed(basis, 1, sqrt(0.5*γ)*sp)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jsparse = map(SparseOperator, J)

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 2)
ρ₀ = Ψ₀⊗dagger(Ψ₀)

tout, ρt = timeevolution.master([0,100], ρ₀, H, J; reltol=1e-7)

ρss = steadystate.master(H, J; eps=1e-4)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.eigenvector(H, J)
@test tracedistance(ρss, ρt[end]) < 1e-6

ρss = steadystate.eigenvector(Hsparse, Jsparse; rho0=ρ₀)
@test tracedistance(ρss, ρt[end]) < 1e-3
