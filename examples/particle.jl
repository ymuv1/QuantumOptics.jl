using quantumoptics

N = 500
xmin = -10.
xmax = 10.
omega = 2.

T = [0.:0.2:2.;]


basis_position = quantumoptics.particle.PositionBasis(xmin, xmax, N)
basis_momentum = quantumoptics.particle.MomentumBasis(basis_position)

p = quantumoptics.particle.momentumoperator(basis_position)
x = quantumoptics.particle.positionoperator(basis_position)
xsamplepoints = quantumoptics.particle.samplepoints(basis_position)

Tpx = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)

V = Operator(basis_position, diagm(omega^2*xsamplepoints.^2))
H = p*p/2 + V

psi0 = quantumoptics.particle.gaussianstate(basis_position, 2., 0., 1.)

tout, psi_t = timeevolution.schroedinger(T, psi0, H)

exp_x = expect(x, psi_t)
exp_p = expect(p, psi_t)
println(real(exp_x))
println(real(exp_p))

using PyCall
@pyimport matplotlib.pyplot as plt

for psi=psi_t
    psi_p = Tpx*psi
    plt.subplot(2,1,1)
    plt.plot(xsamplepoints, real(dagger(psi).data .* psi.data), "b")
    plt.subplot(2,1,2)
    plt.plot(xsamplepoints, real(dagger(psi_p).data .* psi_p.data), "b")
end
plt.show()
