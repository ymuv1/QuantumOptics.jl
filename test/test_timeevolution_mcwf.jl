using Base.Test
using quantumoptics

# Define parameters for spin coupled to electric field mode.
ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

Ntrajectories = 1000
T = Float64[0.:0.1:10.;]

# Define operators
fockbasis = FockBasis(10)
basis = compose(spinbasis, fockbasis)

# Hamiltonian
Ha = embed(basis, 1, 0.5*ωa*sigmaz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sigmam ⊗ create(fockbasis) + sigmap ⊗ destroy(fockbasis)
H = Ha + Hc + Hint
Hsparse = SparseOperator(H)

# Jump operators
Ja = embed(basis, 1, sqrt(γ)*sigmam)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jsparse = map(SparseOperator, J)

# Initial conditions
Ψ₀ = basis_ket(spinbasis, 2) ⊗ basis_ket(fockbasis, 6)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)


# Test mcwf with seed
tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt64(1), reltol=1e-7)
Ψ = Ψt[end]

tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hsparse, Jsparse; seed=UInt64(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hsparse, Jsparse; seed=UInt64(2), reltol=1e-6)
@test norm(Ψt[end]-Ψ) > 0.1


# Test convergence to master solution
tout_master, ρt_master = timeevolution.master(T, ρ₀, H, J)

ρ_average = Operator[ρ₀*0. for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=Uint64(i))
    for j=1:length(T)
        ρ_average[j] += (Ψt[j] ⊗ dagger(Ψt[j]))/Ntrajectories
    end
end
for i=1:length(T)
    err = quantumoptics.tracedistance(ρt_master[i], ρ_average[i])
    @test err < 0.1
end
