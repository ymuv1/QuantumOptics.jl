using Base.Test
using quantumoptics

N = 500
xmin = -62.5
xmax = 70.1

basis_position = quantumoptics.particle.PositionBasis(xmin, xmax, N)
basis_momentum = quantumoptics.particle.MomentumBasis(basis_position)

x0 = 5.1
p0 = -3.2
sigma = 1.
sigma_x = sigma/sqrt(2)
sigma_p = 1./(sigma*sqrt(2))

psix0 = quantumoptics.particle.gaussianstate(basis_position, x0, p0, sigma)
psip0 = quantumoptics.particle.gaussianstate(basis_momentum, x0, p0, sigma)

@test_approx_eq 1.0 norm(psix0)
@test_approx_eq 1.0 norm(psip0)

opx_p = quantumoptics.particle.momentumoperator(basis_position)
opx_x = quantumoptics.particle.positionoperator(basis_position)

opp_p = quantumoptics.particle.momentumoperator(basis_momentum)
opp_x = quantumoptics.particle.positionoperator(basis_momentum)

@test_approx_eq x0 expect(opx_x, psix0)
@test_approx_eq_eps p0 expect(opx_p, psix0) 0.1
@test_approx_eq p0 expect(opp_p, psip0)
@test_approx_eq_eps x0 expect(opp_x, psip0) 0.1

psix0_fft = quantumoptics.particle.transformation(basis_position, basis_momentum, psix0)
psip0_fft = quantumoptics.particle.transformation(basis_momentum, basis_position, psip0)

@test_approx_eq_eps x0 expect(opp_x, psix0_fft) 0.1
@test_approx_eq p0 expect(opp_p, psix0_fft)
@test_approx_eq_eps p0 expect(opx_p, psip0_fft) 0.1
@test_approx_eq x0 expect(opx_x, psip0_fft)

println((angle(psix0_fft.data) - angle(psip0.data)))
println(norm(psix0_fft - psip0))

#println(opp_x.data)
#
# println("Norm(psix0) = ", norm(psix0))
# println("Norm(psip0_fft) = ", norm(psip0_fft))
# println("Norm(psip0) = ", norm(psip0))
# println("Norm(psix0_fft) = ", norm(psix0_fft))
#
# println("<p>x0 = ", expect(opx_p, psix0))
# println("<p>p0_fft = ", expect(opx_p, psip0_fft))
# println("Var(p)x0 = ", expect(opx_p*opx_p, psix0) - expect(opx_p, psix0)^2)
# println("Var(p)p0_fft = ", expect(opx_p*opx_p, psip0_fft) - expect(opx_p, psip0_fft)^2)
# println("<x>x0 = ", expect(opx_x, psix0))
# println("<x>p0_fft = ", expect(opx_x, psip0_fft))
#
# println("<p>fft = ", expect(opp_p, psix0_fft))
# println("<p>p0 = ", expect(opp_p, psip0))
# println("<x>fft = ", expect(opp_x, psix0_fft))
# println("<x>p0 = ", expect(opp_x, psip0))


# println(norm(psix0_fft))
# println(norm(psip0))
