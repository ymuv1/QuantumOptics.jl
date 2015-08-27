module particle

using ..bases, ..states, ..operators, ..operators_lazy

importall ..operators

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

function laplace_p(b::MomentumBasis)
    p_op = Operator(b)
    u = 1/spacing(b)^2
    for i=1:b.N-1
        p_op.data[i+1,i] = u
        p_op.data[i,i] = -2*u
        p_op.data[i,i+1] = u
    end
    p_op.data[b.N,b.N] = -2*u
    return p_op
end

laplace_p(b::PositionBasis) = Operator(b, diagm(samplepoints(b).^2))


type FFTOperator <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    fft_l!
    fft_r!
    mul_before::Vector{Complex128}
    mul_after::Vector{Complex128}
end

function FFTOperator(basis_l::MomentumBasis, basis_r::PositionBasis)
    Lx = (basis_r.xmax - basis_r.xmin)
    dp = spacing(basis_l)
    dx = spacing(basis_r)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    x = zeros(Complex128, length(basis_r))
    mul_before = exp(-1im*basis_l.pmin*(samplepoints(basis_r)-basis_r.xmin))
    mul_after = exp(-1im*basis_r.xmin*samplepoints(basis_l))/sqrt(basis_r.N)
    FFTOperator(basis_l, basis_r, plan_bfft!(x), plan_fft!(x), mul_before, mul_after)
end

function FFTOperator(basis_l::PositionBasis, basis_r::MomentumBasis)
    Lx = (basis_l.xmax - basis_l.xmin)
    dp = spacing(basis_r)
    dx = spacing(basis_l)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    x = zeros(Complex128, length(basis_r))
    mul_before = exp(1im*basis_l.xmin*(samplepoints(basis_r)-basis_r.pmin))
    mul_after = exp(1im*basis_r.pmin*samplepoints(basis_l))/sqrt(basis_r.N)
    FFTOperator(basis_l, basis_r, plan_fft!(x), plan_bfft!(x), mul_before, mul_after)
end


dagger(op::FFTOperator) = FFTOperator(op.basis_r, op.basis_l)


function operators.gemv!{T<:Complex}(alpha::T, M::FFTOperator, b::Ket, beta::T, result::Ket)
    N::Int = M.basis_r.N
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = M.mul_before[i] * b.data[i]
        end
        M.fft_r! * result.data
        @inbounds for i=1:N
            result.data[i] *= M.mul_after[i] * alpha
        end
    else
        psi_ = Ket(M.basis_l, deepcopy(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= M.mul_before[i]
        end
        M.fft_r! * psi_.data
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * psi_.data[i] * M.mul_after[i]
        end
    end
    nothing
end

function operators.gemv!{T<:Complex}(alpha::T, b::Bra, M::FFTOperator, beta::T, result::Bra)
    N::Int = M.basis_l.N
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = conj(M.mul_after[i]) * conj(b.data[i])
        end
        M.fft_l! * result.data
        @inbounds for i=1:N
            result.data[i] = conj(result.data[i]) * M.mul_before[i] * alpha
        end
    else
        psi_ = Bra(M.basis_r, conj(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= conj(M.mul_after[i])
        end
        M.fft_l! * psi_.data
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * conj(psi_.data[i]) * M.mul_before[i]
        end
    end
    nothing
end


end # module
