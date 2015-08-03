using quantumoptics

omega0 = 1.
gamma = 1.
Ntrajectories = 10000

T = [0:0.01:5;]
H = omega0 * sigmaz
J = [gamma*sigmam]
psi0 = basis_ket(spinbasis, 1)
rho0 = tensor(psi0, dagger(psi0))

tout, rho_t = timeevolution.master(T, rho0, H, J);
rho_average = Operator[rho0*0. for i=1:length(T)]
for i=1:Ntrajectories
    tout, psi_t = timeevolution.mcwf(T, psi0, H, J, display_beforeevent=false, display_afterevent=false);
    for j=1:length(T)
        #print(norm(psi_t[j]))
        rho_average[j] += tensor(psi_t[j], dagger(psi_t[j]))/Ntrajectories
    end
end

using PyCall
@pyimport matplotlib.pyplot as plt

plt.plot(T, expect(sigmaz, rho_t))
plt.plot(T, expect(sigmaz, rho_average))
plt.show()