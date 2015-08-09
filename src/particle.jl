module particle

using ..bases, ..states, ..operators

type PositionBasis <: Basis
    shape::Vector{Int}
    xmin::Float64
    xmax::Float64
    N::Int
    PositionBasis(xmin::Float64, xmax::Float64, N::Int) = new([N], xmin, xmax, N)
end

type MomentumBasis <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    MomentumBasis(pmin::Float64, pmax::Float64, N::Int) = new([N], pmin, pmax, N)
end

PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, N))
MomentumBasis(b::MomentumBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


function gaussianstate(b::PositionBasis, x0::Float64, p0::Float64, sigma::Float64)
    psi = Ket(b)
    dx = (b.xmax - b.xmin)/b.N
    alpha = 1./(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*x - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end

function gaussianstate(b::MomentumBasis, x0::Float64, p0::Float64, sigma::Float64)
    psi = Ket(b)
    dp = (b.pmax - b.pmin)/b.N
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*x0*x - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end

spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N

samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

positionoperator(b::PositionBasis) = Operator(b, diagm(samplepoints(b)))

function positionoperator(b::MomentumBasis)
    p_op = Operator(b)
    u = -1im/(2*spacing(b))
    for i=1:b.N-1
        p_op.data[i+1,i] = u
        p_op.data[i,i+1] = -u
    end
    return p_op
end

momentumoperator(b::MomentumBasis) = Operator(b, diagm(samplepoints(b)))

function momentumoperator(b::PositionBasis)
    x_op = Operator(b)
    u = 1im/(2*spacing(b))
    for i=1:b.N-1
        x_op.data[i+1,i] = u
        x_op.data[i,i+1] = -u
    end
    return x_op
end

function laplace(b::PositionBasis)
    x_op = Operator(b)
    u = 1/spacing(b)^2
    for i=1:b.N-1
        x_op.data[i+1,i] = u
        x_op.data[i,i] = -2*u
        x_op.data[i,i+1] = u
    end
    x_op.data[b.N,b.N] = -2*u
    return x_op
end

laplace(b::MomentumBasis) = Operator(b, diagm(samplepoints(b).^2))


function transformation(b1::MomentumBasis, b2::PositionBasis, psi::Ket)
    Lp = (b1.pmax - b1.pmin)
    dx = spacing(b2)
    if b1.N != b2.N || abs(2*pi/dx - Lp)/Lp > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(1im*2*pi*b2.xmin/N*(samplepoints(b1)-b1.pmin)).*psi.data
    psi_fft = exp(1im*2*pi*b1.pmin/N*samplepoints(b2)).*ifft(psi_shifted)*sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation(b1::PositionBasis, b2::MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = spacing(b2)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(-1im*2*pi*b2.pmin/N*(samplepoints(b1)-b1.xmin)).*psi.data
    psi_fft = exp(-1im*2*pi*b1.xmin/N*samplepoints(b2)).*fft(psi_shifted)/sqrt(N)
    return Ket(b2, psi_fft)
end

end # module
