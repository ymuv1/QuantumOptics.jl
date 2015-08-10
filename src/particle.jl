module particle

using ..bases, ..states, ..operators, ..operators_lazy

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

PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


function gaussianstate(b::PositionBasis, x0::Float64, p0::Float64, sigma::Float64)
    psi = Ket(b)
    dx = spacing(b)
    alpha = 1./(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*(x-x0) - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end

function gaussianstate(b::MomentumBasis, x0::Float64, p0::Float64, sigma::Float64)
    psi = Ket(b)
    dp = spacing(b)
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(-1im*x0*p - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end

spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N

samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

positionoperator(b::PositionBasis) = Operator(b, diagm(samplepoints(b)))

# function positionoperator(b::MomentumBasis)
#     p_op = Operator(b)
#     u = 1im/(2*spacing(b))
#     for i=1:b.N-1
#         p_op.data[i+1,i] = -u
#         p_op.data[i,i+1] = u
#     end
#     return p_op
# end

function positionoperator(b::MomentumBasis)
    p_op = Operator(b)
    u = 1im/(12*spacing(b))
    for i=1:b.N-2
        p_op.data[i,i+2] = -u
        p_op.data[i,i+1] = 8*u
        p_op.data[i+1,i] = -8*u
        p_op.data[i+2,i] = u
    end
    return p_op
end

momentumoperator(b::MomentumBasis) = Operator(b, diagm(samplepoints(b)))

# function momentumoperator(b::PositionBasis)
#     x_op = Operator(b)
#     u = -1im/(2*spacing(b))
#     for i=1:b.N-1
#         x_op.data[i+1,i] = -u
#         x_op.data[i,i+1] = u
#     end
#     return x_op
# end

function momentumoperator(b::PositionBasis)
    p_op = Operator(b)
    u = -1im/(12*spacing(b))
    for i=1:b.N-2
        p_op.data[i,i+2] = -u
        p_op.data[i,i+1] = 8*u
        p_op.data[i+1,i] = -8*u
        p_op.data[i+2,i] = u
    end
    return p_op
end

function laplace_x(b::PositionBasis)
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

laplace_x(b::MomentumBasis) = Operator(b, diagm(samplepoints(b).^2))



function transformation(b1::MomentumBasis, b2::PositionBasis, psi::Ket)
    Lp = (b1.pmax - b1.pmin)
    dx = spacing(b2)
    if b1.N != b2.N || abs(2*pi/dx - Lp)/Lp > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(1im*b2.xmin*(samplepoints(b1)-b1.pmin)).*psi.data
    psi_fft = exp(1im*b1.pmin*samplepoints(b2)).*ifft(psi_shifted)*sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation(b1::PositionBasis, b2::MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = spacing(b2)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(-1im*b2.pmin*(samplepoints(b1)-b1.xmin)).*psi.data
    psi_fft = exp(-1im*b1.xmin*samplepoints(b2)).*fft(psi_shifted)/sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation!(b1::PositionBasis, b2::MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = spacing(b2)
    dx = spacing(b1)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    for i=1:N
        psi.data[i] *= exp(-1im*b2.pmin*i*dx)
    end
    fft!(psi.data)
    for i=1:N
        psi.data[i] *= exp(-1im*b1.xmin*(b2.pmin + i*dp))/sqrt(N)
    end
    psi.basis = b2
    return psi
end

type FFTOperator <: LazyOperator
    basis_l::Basis
    basis_r::Basis
end

*(op::FFTOperator, psi::Ket) = transformation(op.basis_r, op.basis_l, psi)

end # module
