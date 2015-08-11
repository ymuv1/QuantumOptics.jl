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

function transformation(b1::quantumoptics.particle.MomentumBasis, b2::quantumoptics.particle.PositionBasis, psi::Ket)
    Lp = (b1.pmax - b1.pmin)
    dx = quantumoptics.particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dx - Lp)/Lp > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(1im*b2.xmin*(quantumoptics.particle.samplepoints(b1)-b1.pmin)).*psi.data
    psi_fft = exp(1im*b1.pmin*quantumoptics.particle.samplepoints(b2)).*ifft(psi_shifted)*sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation(b1::quantumoptics.particle.PositionBasis, b2::quantumoptics.particle.MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = quantumoptics.particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(-1im*b2.pmin*(quantumoptics.particle.samplepoints(b1)-b1.xmin)).*psi.data
    psi_fft = exp(-1im*b1.xmin*quantumoptics.particle.samplepoints(b2)).*fft(psi_shifted)/sqrt(N)
    return Ket(b2, psi_fft)
end

psix0_fft = transformation(basis_position, basis_momentum, psix0)
psip0_fft = transformation(basis_momentum, basis_position, psip0)

@test_approx_eq_eps x0 expect(opp_x, psix0_fft) 0.1
@test_approx_eq p0 expect(opp_p, psix0_fft)
@test_approx_eq_eps p0 expect(opx_p, psip0_fft) 0.1
@test_approx_eq x0 expect(opx_x, psip0_fft)

@test_approx_eq_eps 0. norm(psix0_fft - psip0) 1e-12
@test_approx_eq_eps 0. norm(psip0_fft - psix0) 1e-12


Tpx = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)
Txp = quantumoptics.particle.FFTOperator(basis_position, basis_momentum)
psix0_fft = Tpx*psix0
psip0_fft = Txp*psip0

@test_approx_eq_eps x0 expect(opp_x, psix0_fft) 0.1
@test_approx_eq p0 expect(opp_p, psix0_fft)
@test_approx_eq_eps p0 expect(opx_p, psip0_fft) 0.1
@test_approx_eq x0 expect(opx_x, psip0_fft)

@test_approx_eq_eps 0. norm(psix0_fft - psip0) 1e-12
@test_approx_eq_eps 0. norm(psip0_fft - psix0) 1e-12

@test_approx_eq_eps 0. norm(dagger(Tpx*psix0) - dagger(psix0)*dagger(Tpx)) 1e-12
@test_approx_eq_eps 0. norm(dagger(Txp*psip0) - dagger(psip0)*dagger(Txp)) 1e-12

psi_ = deepcopy(psip0)
operators.gemv!(Complex(1.), Tpx, psix0, Complex(0.), psi_)
@test_approx_eq_eps 0. norm(psi_ - psip0) 1e-12
operators.gemv!(Complex(1.), Tpx, psix0, Complex(1.), psi_)
@test_approx_eq_eps 0. norm(psi_ - 2*psip0) 1e-12

psi_ = deepcopy(psix0)
operators.gemv!(Complex(1.), Txp, psip0, Complex(0.), psi_)
@test_approx_eq_eps 0. norm(psi_ - psix0) 1e-12
operators.gemv!(Complex(1.), Txp, psip0, Complex(1.), psi_)
@test_approx_eq_eps 0. norm(psi_ - 2*psix0) 1e-12

rhox0x0 = tensor(psix0, dagger(psix0))
rhop0x0 = tensor(psip0, dagger(psix0))
rhop0p0 = tensor(psip0, dagger(psip0))

rho_ = deepcopy(rhop0x0)
operators.gemm!(Complex(1.), Tpx, rhox0x0, Complex(0.), rho_)
@test_approx_eq_eps 0. tracedistance(rho_, rhop0x0) 1e-5

rho_ = deepcopy(rhop0p0)
operators.gemm!(Complex(1.), rhop0x0, Txp, Complex(0.), rho_)
@test_approx_eq_eps 0. tracedistance(rho_, rhop0p0) 1e-5
