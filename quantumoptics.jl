include("quantumbases.jl")

module quantumoptics

using quantumbases

export Basis, GenericBasis, FockBasis, CompositeBasis, Operator, compose, multiplicable, create, destroy, basis_ket, tensor, dagger, mul!, check_multiplicable, add!

import Iterators
#import ODE
import ode
import Base.LinAlg.BLAS

abstract AbstractStateVector
abstract StateVector <: AbstractStateVector

type Bra <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
end

type Ket <: StateVector
    basis::Basis
    data::Vector{Complex{Float64}}
end

Bra(b::Basis) = Bra(b, zeros(Complex, length(b)))
Ket(b::Basis) = Ket(b, zeros(Complex, length(b)))

abstract AbstractOperator

type Operator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex{Float64}}
end

Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(Complex, length(b1), length(b2)))

Base.norm(op::Operator, p) = norm(op.data, p)
Base.trace(op::Operator) = trace(op.data)
expect(op::AbstractOperator, state::AbstractOperator) = trace(op*state)
expect{T<:AbstractOperator}(op::Operator, states::Vector{T}) = [expect(op, state) for state=states]
identity(b::Basis) = Operator(b, b, eye(Complex, length(b)))
number(b::Basis) = Operator(b, b, diagm(map(Complex, 0:(length(b)-1))))
destroy(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),1))
create(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),-1))

function basis_vector(shape::Vector{Int}, index::Vector{Int})
    x = zeros(Complex, shape...)
    x[index] = Complex(1.)
    reshape(x, prod(shape))
end

basis_bra(b::Basis, index::Array{Int}) = Bra(b, basis_vector(b.shape, index))
basis_bra(b::Basis, index::Int) = basis_bra(b, [index])
basis_ket(b::Basis, index::Array{Int}) = Ket(b, basis_vector(b.shape, index))
basis_ket(b::Basis, index::Int) = basis_ket(b, [index])

dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))
dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')

type IncompatibleBases <: Exception end

check_multiplicable(b1::Basis, b2::Basis) = (multiplicable(b1, b2) ? true : throw(IncompatibleBases()))
check = check_multiplicable

*(a::Operator, b::Ket) = (check(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::Operator) = (check(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::Operator, b::Operator) = (check(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))


# function mul!(a::Matrix, b::Matrix, result::Matrix)
#     N1, N2 = size(a)
#     N3 = size(b)[2]
#     fill!(result, zero(eltype(a)))
#     for i=1:N1, j=1:N3
#         tmp = 0.im
#         for k=1:N2
#             tmp += a[i,k]*b[k,j]
#         end
#         result[i,j] = tmp
#     end
#     return result
# end


function mul!(a::Matrix, b::Matrix, result::Matrix)
    Base.LinAlg.BLAS.gemm!('N', 'N', one(eltype(a)), a, b, zero(eltype(a)), result)
    return result
end
function mul!(a::Matrix, b::Number, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = b*a[i,j]
    end
    return result
end
mul!(a::Number, b::Matrix, result::Matrix) = mul!(b, a, result)
mul!(a::Matrix, b::Number) = mul!(a,b,a)
mul!(a::Number, b::Matrix) = mul!(b,a,b)

mul!(a::Operator, b::Operator, result::Operator) = (check(a.basis_r, b.basis_l); a.basis_l==result.basis_l && b.basis_r==result.basis_r ? mul!(a.data, b.data, result.data) : throw(IncompatibleBases()); result)
mul!(a::Operator, b::Number, result::Operator) = (a.basis_l==result.basis_l && a.basis_r==result.basis_r ? mul!(a.data, complex(b), result.data): throw(IncompatibleBases()); result)
mul!(a::Number, b::Operator, result::Operator) = (b.basis_l==result.basis_l && b.basis_r==result.basis_r ? mul!(complex(a), b.data, result.data): throw(IncompatibleBases()); result)
mul!(a::Operator, b::Number) = mul!(a, complex(b), a)
mul!(a::Number, b::Operator) = mul!(complex(a), b, b)


function add!(a::Matrix, b::Matrix, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = a[i,j] + b[i,j]
    end
    return result
end
add!(a::Matrix, b::Matrix) = add!(a,b,a)
add!(a::Operator, b::Operator, result::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) && (a.basis_l==result.basis_l) && (a.basis_r==result.basis_r)? add!(a.data, b.data, result.data) : throw(IncompatibleBases()); result)
add!(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? add!(a.data, b.data) : throw(IncompatibleBases()); a)

function sub!(a::Matrix, b::Matrix, result::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        result[i,j] = a[i,j] - b[i,j]
    end
    return a
end
sub!(a::Matrix, b::Matrix) = sub!(a,b,a)
sub!(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? sub!(a.data, b.data) : throw(IncompatibleBases()); a)
sub!(a::Operator, b::Operator, result::Operator) = (a.basis_l==b.basis_l && a.basis_l==result.basis_l && a.basis_r==b.basis_r && a.basis_r==result.basis_r ? sub!(a.data, b.data, result.data) : throw(IncompatibleBases()); result)

function set!(a::Matrix, b::Matrix)
    shape = size(a)
    for j=1:shape[2], i=1:shape[1]
        a[i,j] = b[i,j]
    end
end
set!(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? set!(a.data, b.data) : throw(IncompatibleBases()); a)

zero!(a::Operator) = fill!(a.data, zero(eltype(a.data)))



*(a::Bra, b::Ket) = (check(a.basis, b.basis); sum(a.data.*b.data))
*{T<:StateVector}(a::Number, b::T) = T(b.basis, complex(a)*b.data)
*{T<:StateVector}(a::T, b::Number) = T(a.basis, complex(b)*a.data)
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, complex(a)*b.data)

/{T<:StateVector}(a::T, b::Number) = T(a.basis, a.data/complex(b))
/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data/complex(b))

+{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data+b.data) : throw(IncompatibleBases()))
+(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? Operator(a.basis_l, a.basis_r, a.data+b.data) : throw(IncompatibleBases()))

-{T<:StateVector}(a::T, b::T) = (a.basis==b.basis ? T(a.basis, a.data-b.data) : throw(IncompatibleBases()))
-(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? Operator(a.basis_l, a.basis_r, a.data-b.data) : throw(IncompatibleBases()))




tensor{T<:StateVector}(a::T, b::T) = T(compose(a.basis, b.basis), kron(a.data, b.data))
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(a.data, b.data), prod(a.basis.shape), prod(b.basis.shape)))
tensor(a::Operator, b::Operator) = Operator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))

abstract LazyOperator <: AbstractOperator

type EmbeddedOperator <: AbstractOperator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    indices_l::Array{Int}
    indices_r::Array{Int}
    operator::AbstractOperator

    function EmbeddedOperator(basis_l::CompositeBasis, basis_r::CompositeBasis, indices_l::Array{Int}, indices_r::Array{Int}, operator::AbstractOperator)
        if length(indices_l)==1
            @assert(operator.basis_l==basis_l.bases[indices_l[1]])
        else
            @assert(operator.basis_l.bases==basis_l.bases[indices_l])
        end
        if length(indices_r)==1
            @assert(operator.basis_r==basis_r.bases[indices_r[1]])
        else
            @assert(operator.basis_r.bases==basis_r.bases[indices_r])
        end
        compl_indices_l = complementary_indices(length(basis_l.shape), indices_l)
        compl_indices_r = complementary_indices(length(basis_r.shape), indices_r)
        @assert(basis_l.bases[compl_indices_l]==basis_r.bases[compl_indices_r])
        new(basis_l, basis_r, indices_l, indices_r, operator)
    end
end

function complementary_indices(N::Int, indices::Vector{Int})
    dual_indices = zeros(Int, N-length(indices))
    count = 1
    for i=1:N
        if !(i in indices)
            dual_indices[count] = i
            count += 1
        end
    end
    return dual_indices
end

function *(a::EmbeddedOperator, b::Ket)
    check_multiplicable(a.basis_r, b.basis)
    uninvolved_indices_l = complementary_indices(length(a.basis_l.shape), a.indices_l)
    uninvolved_indices_r = complementary_indices(length(a.basis_r.shape), a.indices_r)
    uninvolved_shape = a.basis_l.shape[uninvolved_indices_l]
    x = Ket(a.basis_l)
    data_x = reshape(x.data, x.basis.shape...)
    broad_index_l = {(1:n) for n=a.basis_l.shape}
    broad_index_r = {(1:n) for n=a.basis_r.shape}
    #broad_index_l = zeros(Int, length(a.basis_l.shape))
    #broad_index_r = zeros(Int, length(a.basis_r.shape))
    for uninvolved_index in Iterators.product([1:n for n = uninvolved_shape]...)
        # println("Uninvolved Index: ", uninvolved_index)
        # println("Uninvolved Indices L: ", uninvolved_indices_l)
        # println("Uninvolved Indices R: ", uninvolved_indices_r)
        broad_index_l[uninvolved_indices_l] = [uninvolved_index...]
        broad_index_r[uninvolved_indices_r] = [uninvolved_index...]
        # println("Broad Index L: ", broad_index_l)
        # println("Broad Index R: ", broad_index_r)
        subdata_b = reshape(reshape(b.data, b.basis.shape...)[broad_index_r...], prod(a.operator.basis_r.shape))
        sub_b = Ket(a.operator.basis_r, subdata_b)
        sub_x = a.operator*sub_b
        data_x[broad_index_l...] = reshape(sub_x.data, sub_x.basis.shape...)
        # println("subdata_x: ", data_x[broad_index_l])
        # println("x.data", x.data)
    end
    return x
end

type LazySum <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    operands::Array{AbstractOperator}
end


function oderkf(F, x0, tspan, p, a, bs, bp; reltol = 1.0e-5, abstol = 1.0e-8)
    # see p.91 in the Ascher & Petzold reference for more infomation.
    pow = 1/p   # use the higher order to estimate the next step size
    c = float(sum(a, 2))   # consistency condition

    # Initialization
    t = tspan[1]
    tfinal = tspan[end]
    tdir = sign(tfinal - t)
    hmax = abs(tfinal - t)/2.5
    hmin = abs(tfinal - t)/1e9
    h = tdir*abs(tfinal - t)/100  # initial guess at a step size
    # x = x0
    x = Operator(x0.basis_l, x0.basis_r)
    add!(x, x0)
    tout = t            # first output time
    xout = Array(typeof(x0), 1)
    xout[1] = x         # first output solution

    #k = Array(typeof(x0), length(c))
    #k[1] = F(t,x) # first stage
    k = [Operator(x0.basis_l, x0.basis_r) for i=1:length(c)]
    set!(k[1], F(t,x))

    xs = Operator(x0.basis_l, x0.basis_r)
    xp = Operator(x0.basis_l, x0.basis_r)
    dx = Operator(x0.basis_l, x0.basis_r)
    tmp = Operator(x0.basis_l, x0.basis_r)

    while abs(t) != abs(tfinal) && abs(h) >= hmin
        if abs(h) > abs(tfinal-t)
            h = tfinal - t
        end

        #(p-1)th and pth order estimates
        #xs = x + h*bs[1]*k[1]
        mul!(k[1], h*bs[1], xs)
        add!(xs, x)
        #xp = x + h*bp[1]*k[1]
        mul!(k[1], h*bp[1], xp)
        add!(xp, x)

        for j = 2:length(c)
            #dx = a[j,1]*k[1]
            mul!(a[j,1], k[1], dx)
            for i = 2:j-1
                #dx += a[j,i]*k[i]
                mul!(a[j,i], k[i], tmp)
                add!(dx, tmp)
            end
            #k[j] = F(t + h*c[j], x + h*dx)
            mul!(h, dx, tmp)
            add!(tmp, x)
            set!(k[j], F(t + h*c[j], tmp))

            # compute the (p-1)th order estimate
            #xs = xs + h*bs[j]*k[j]
            mul!(h*bs[j], k[j], tmp)
            add!(xs, tmp)
            # compute the pth order estimate
            #xp = xp + h*bp[j]*k[j]
            mul!(h*bp[j], k[j], tmp)
            add!(xp, tmp)
        end

        # estimate the local truncation error
        #gamma1 = xs - xp
        sub!(xs, xp, tmp)
        gamma1 = tmp

        # Estimate the error and the acceptable error
        delta = norm(gamma1, Inf)              # actual error
        tau   = max(reltol*norm(x,Inf),abstol) # allowable error

        # Update the solution only if the error is acceptable
        if delta <= tau
            t = t + h

            #x = xp    # <-- using the higher order estimate is called 'local extrapolation'
            x, xp = xp, x

            tout = [tout; t]
            #push!(xout, x)

            # Compute the slopes by computing the k[:,j+1]'th column based on the previous k[:,1:j] columns
            # notes: k needs to end up as an Nxs, a is 7x6, which is s by (s-1),
            #        s is the number of intermediate RK stages on [t (t+h)] (Dormand-Prince has s=7 stages)
            if c[end] == 1
                # Assign the last stage for x(k) as the first stage for computing x[k+1].
                # This is part of the Dormand-Prince pair caveat.
                # k[:,7] has already been computed, so use it instead of recomputing it
                # again as k[:,1] during the next step.
                k[1], k[end] = k[end], k[1]
            else
                #k[1] = F(t,x) # first stage
                set!(k[1], F(t,x))
            end
        end

        # Update the step size
        h = min(hmax, 0.8*h*(tau/delta)^pow)
    end # while (t < tfinal) & (h >= hmin)

    if abs(t) < abs(tfinal)
      println("Step size grew too small. t=", t, ", h=", abs(h), ", x=", x)
    end

    return tout, xout
end

# Dormand-Prince coefficients
const dp_coefficients = (5,
                         [    0           0          0         0         0        0
                              1/5         0          0         0         0        0
                              3/40        9/40       0         0         0        0
                             44/45      -56/15      32/9       0         0        0
                          19372/6561 -25360/2187 64448/6561 -212/729     0        0
                           9017/3168   -355/33   46732/5247   49/176 -5103/18656  0
                             35/384       0        500/1113  125/192 -2187/6784  11/84],
                         # 4th order b-coefficients
                         [5179/57600 0 7571/16695 393/640 -92097/339200 187/2100 1/40],
                         # 5th order b-coefficients
                         [35/384 0 500/1113 125/192 -2187/6784 11/84 0],
                         )
ode45(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, dp_coefficients...; kwargs...)



function dmaster(rho::AbstractOperator, H::AbstractOperator, J::Vector)
    drho = -1im * (H*rho - rho*H)
    for j = J
        drho = drho + j*rho*dagger(j) - dagger(j)*j*rho/Complex(2) - rho*dagger(j)*j/Complex(2)
    end
    return drho
end

function dmaster2(rho::AbstractOperator, H::AbstractOperator, J::Vector, Jdagger::Vector, drho::Operator, tmp1::Operator, tmp2::Operator)
    zero!(drho)
    mul!(H, rho, tmp1)
    mul!(rho, H, tmp2)
    add!(drho, sub!(tmp1, tmp2))
    mul!(drho, Complex(0, -1.))
    for i = 1:length(J)
        mul!(J[i], rho, tmp1)
        mul!(tmp1, Jdagger[i], tmp2)
        add!(drho, tmp2)

        mul!(J[i], rho, tmp1)
        mul!(Jdagger[i], tmp1, tmp2)
        mul!(tmp2, Complex(-0.5))
        add!(drho, tmp2)

        mul!(rho, Jdagger[i], tmp1)
        mul!(tmp1, J[i], tmp2)
        mul!(tmp2, Complex(-0.5))
        add!(drho, tmp2)
    end
    return drho
end

function master(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector)
    f(t::Number,rho::AbstractOperator) = dmaster(rho, H, J)
    tout, rho_t = ode.ode45(f, rho0, T)
    return tout, rho_t
end

function master2(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector)
    Jdagger = [dagger(j) for j=J]
    tmp1 = Operator(rho0.basis_l, rho0.basis_r)
    tmp2 = Operator(rho0.basis_l, rho0.basis_r)
    drho = Operator(rho0.basis_l, rho0.basis_l)
    f(t::Float64,rho::Operator) = dmaster2(rho, H, J, Jdagger, drho, tmp1, tmp2)
    tout, rho_t = ode45(f, rho0, float(T))
    return tout, rho_t
end


end