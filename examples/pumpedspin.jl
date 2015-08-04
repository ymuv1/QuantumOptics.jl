using quantumoptics

# System parameters for a single decaying spin
ω0 = 1.
γ = 1.
R = 0.5

# System operators
H = ω0 * sigmaz
J = [γ*sigmam, R*sigmap]

# Initial state
Ψ₀ = basis_ket(spinbasis, 1)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

# Integration options
Ntrajectories = 1000
T = [0:0.01:10;]


# Master time evolution
tout_master, ρt_master = timeevolution.master(T, ρ₀, H, J)

# Averaged MCWF time evolution
ρt_average = Operator[ρ₀*0. for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J)
    for j=1:length(T)
        ρt_average[j] += (Ψt[j] ⊗ dagger(Ψt[j]))/Ntrajectories
    end
end

# Example trajectories
T_example = [0.,100.]
tout_example, Ψt_example = timeevolution.mcwf(T_example, Ψ₀, H, J; seed=UInt(1),
                                        display_beforeevent=true,
                                        display_afterevent=true)


using PyCall
@pyimport matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(T, real(expect(sigmaz, ρt_master)))
plt.plot(T, real(expect(sigmaz, ρt_average)))
plt.xlabel("\$\\mathrm\{Time\} \$")
plt.ylabel("\$\\sigma^z\$")

plt.subplot(2, 1, 2)
plt.plot(tout_example, real(expect(sigmaz, Ψt_example)))
plt.xlabel("\$\\mathrm\{Time\} \$")
plt.ylabel("\$\\sigma^z\$")

plt.show()
