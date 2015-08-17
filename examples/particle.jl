using quantumoptics

N = 500
xmin = -10.
xmax = 10.
omega = 2.

T = [0.:0.2:2.;]


basis_position = quantumoptics.particle.PositionBasis(xmin, xmax, N)
basis_momentum = quantumoptics.particle.MomentumBasis(basis_position)
xsamplepoints = quantumoptics.particle.samplepoints(basis_position)
psamplepoints = quantumoptics.particle.samplepoints(basis_momentum)

Tpx = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)
Txp = quantumoptics.particle.FFTOperator(basis_position, basis_momentum)

# Position Basis
p = quantumoptics.particle.momentumoperator(basis_position)
x = quantumoptics.particle.positionoperator(basis_position)
V = Operator(basis_position, diagm(omega^2*xsamplepoints.^2))
H = p*p/2 #+ V

psi0 = quantumoptics.particle.gaussianstate(basis_position, 0., 0., 1.)
tout, psi_x_t = timeevolution.schroedinger(T, psi0, H)


# Momentum Basis
p_ = quantumoptics.particle.momentumoperator(basis_momentum)
x_ = quantumoptics.particle.positionoperator(basis_momentum)
V_ = Tpx*V*Txp
H_ = p_*p_/2 #+ V_

#psi0 = quantumoptics.particle.gaussianstate(basis_momentum, 2., 0., 1.)
tout, psi_p_t = timeevolution.schroedinger(T, Tpx*psi0, H_)

for i=1:length(T)
    println(expect(p_, psi_p_t[i]), expect(p, psi_x_t[i]))
    println(expect(x_, psi_p_t[i]), expect(x, psi_x_t[i]))
    println(norm(Txp*psi_p_t[i] - psi_x_t[i]))
    println("================================================")
end
# using PyCall
# @pyimport matplotlib.pyplot as plt
#
# for i=1:length(T)
#     psi_x = psi_x_t[i]
#     psi_p = psi_p_t[i]
#     plt.subplot(2,1,1)
#     plt.plot(xsamplepoints, real(dagger(psi_x).data .* psi_x.data), "b")
#     plt.plot(xsamplepoints, real(dagger(Txp*psi_p).data .* (Txp*psi_).data), "g")
#     plt.subplot(2,1,2)
#     plt.plot(psamplepoints, real(dagger(Tpx*psi_x).data .* (Tpx*psi_x).data), "b")
#     plt.plot(psamplepoints, real(dagger(psi_p).data .* psi_.data), "g")
# end
# plt.show()
