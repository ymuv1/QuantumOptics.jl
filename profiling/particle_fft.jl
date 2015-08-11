using quantumoptics

N = 500000
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

@time psix0_fft = transformation(basis_position, basis_momentum, psix0)
@time psix0_fft = transformation(basis_position, basis_momentum, psix0)


function transformation!(b1::quantumoptics.particle.PositionBasis, b2::quantumoptics.particle.MomentumBasis, psi::Ket)
    N::Int = b1.N
    dp = quantumoptics.particle.spacing(b2)
    dx = quantumoptics.particle.spacing(b1)
    @inbounds for i=1:N
        psi.data[i] .*= exp(-1im*b2.pmin*(i-1)*dx)
    end
    fft!(psi.data)
    @inbounds for i=1:N
        psi.data[i] .*= exp(-1im*b1.xmin*(b2.pmin + (i-1)*dp))/sqrt(N)
    end
    return psi
end

psi_ = deepcopy(psix0)
@time transformation!(basis_position, basis_momentum, psi_)
psi_ = deepcopy(psix0)
@time transformation!(basis_position, basis_momentum, psi_)


function transformation!(b1::quantumoptics.particle.PositionBasis, b2::quantumoptics.particle.MomentumBasis, psi::Ket, fft_plan!, mul_before, mul_after)
    N::Int = b1.N
    @inbounds for i=1:N
        psi.data[i] *= mul_before[i]
    end
    fft_plan!(psi.data)
    @inbounds for i=1:N
        psi.data[i] *= mul_after[i]
    end
    return psi
end

mul_before = exp(-1im*basis_momentum.pmin*(quantumoptics.particle.samplepoints(basis_position)-basis_position.xmin))
mul_after = exp(-1im*basis_position.xmin*quantumoptics.particle.samplepoints(basis_momentum))/sqrt(N)

A = plan_fft!(psix0.data)
psi_ = deepcopy(psix0)
@time transformation!(basis_position, basis_momentum, psi_, A, mul_before, mul_after)
A = plan_fft!(psix0.data)
psi_ = deepcopy(psix0)
@time transformation!(basis_position, basis_momentum, psi_, A, mul_before, mul_after)

T = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)
@time T*psix0
@time T*psix0

psi_ = deepcopy(psix0)
@time quantumoptics.operators.gemv!(Complex(1.), T, psix0, Complex(0.), psi_)
psi_ = deepcopy(psix0)
@time quantumoptics.operators.gemv!(Complex(1.), T, psix0, Complex(0.), psi_)
