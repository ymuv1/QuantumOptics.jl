module particle

using ..bases, ..states, ..operators

type PositionBasis <: Basis
    shape::Vector{Int}
    xmin::Float64
    xmax::Float64
    N::Int
    PositionBasis(xmin::Float64, xmax::Float64, N::Int) = new([N], xmin, xmax, N)
end
PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N


type MomentumBasis <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    MomentumBasis(pmin::Float64, pmax::Float64, N::Int) = new([N], pmin, pmax, N)
end
MomentumBasis(b::MomentumBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, N))

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

function samplepoints(b::PositionBasis)
    dx = (b.xmax - b.xmin)/b.N
    return Float64[b.xmin + i*dx for i=0:b.N-1]
end

function samplepoints(b::MomentumBasis)
    dp = (b.pmax - b.pmin)/b.N
    return Float64[b.pmin + i*dp for i=0:b.N-1]
end


function positionoperator(b::PositionBasis)
    return Operator(b, diagm(samplepoints(b)))
end

function positionoperator(b::MomentumBasis)
    p_op = Operator(b)
    dp = (b.pmax - b.pmin)/b.N
    u = -1im/(2*dp)
    for i=1:b.N-1
        p_op.data[i+1,i] = u
        p_op.data[i,i+1] = -u
    end
    return p_op
end

function momentumoperator(b::MomentumBasis)
    return Operator(b, diagm(samplepoints(b)))
end

function momentumoperator(b::PositionBasis)
    x_op = Operator(b)
    dx = (b.xmax - b.xmin)/b.N
    u = 1im/(2*dx)
    for i=1:b.N-1
        x_op.data[i+1,i] = u
        x_op.data[i,i+1] = -u
    end
    return x_op
end

end # module
