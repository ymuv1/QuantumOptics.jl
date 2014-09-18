module timeevolution

using ..operators
using ..states
using ..ode_dopri

export master

function substep{T}(a::Matrix{T}, beta::T, b::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        @simd for i=1:size(b,1)
            @inbounds result[i,j] = a[i,j] + beta*b[i,j]
        end
    end
end

function scale{T}(alpha::T, a::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        @simd for i=1:size(a,1)
            @inbounds result[i,j] = alpha*a[i,j]
        end
    end
end

function oderk_step{T}(F::Function, t::T, h::T, x::Matrix{T}, a::Matrix{T},
                bs::Vector{T}, bp::Vector{T}, c::Vector{T},
                xs::Matrix{T}, xp::Matrix{T}, dx::Matrix{T}, tmp::Matrix{T},
                k::Vector{Matrix{T}})
    substep(x, h*bs[1], k[1], xs) #xs = x + h*bs[1]*k[1]
    substep(x, h*bp[1], k[1], xp) #xp = x + h*bp[1]*k[1]
    for j = 2:length(c)
        scale(a[j,1], k[1], dx) #dx = a[j,1]*k[1]
        for i = 2:j-1
            substep(dx, a[j,i], k[i], dx) #dx += a[j,i]*k[i]
        end
        #k[j] = F(t + h*c[j], x + h*dx)
        substep(x, h, dx, tmp)
        F(t + h*c[j], tmp, k[j])
        # compute the (p-1)th order estimate
        substep(xs, h*bs[j], k[j], xs) #xs = xs + h*bs[j]*k[j]
        # compute the pth order estimate
        substep(xp, h*bp[j], k[j], xp) #xp = xp + h*bp[j]*k[j]
    end
    # Estimate error
    substep(xs, complex(-1.), xp, tmp) #gamma1 = xs - xp
    return norm(tmp, Inf)
end

function oderk{T}(F, tspan, x0::Matrix{T}, p, a, bs, bp; reltol = 1.0e-5, abstol = 1.0e-8,
                    tmps=Vector{Matrix{T}}, fout=(t,x)->nothing)
    # see p.91 in the Ascher & Petzold reference for more infomation.
    pow = 1/p   # use the higher order to estimate the next step size
    c = vec(complex(sum(a, 2)))   # consistency condition
    a = complex(a)
    bs = vec(complex(bs))
    bp = vec(complex(bp))
    #println(c)

    # Initialization
    t = tspan[1]
    tfinal = tspan[end]
    tdir = sign(tfinal - t)
    hmax = abs(tfinal - t)/2.5
    hmin = abs(tfinal - t)/1e9
    h = tdir*abs(tfinal - t)/100  # initial guess at a step size
    x = 1*x0
    fout(t, x)

    xs, xp, dx, tmp = tmps[1:4]
    k = tmps[5:4+length(c)]
    F(t,x,k[1]) #k[1] = F(t,x)

    while abs(t) != abs(tfinal) && abs(h) >= hmin
        if abs(h) > abs(tfinal-t)
            h = tfinal - t
        end
        delta = oderk_step(F, complex(float(t)), complex(h), x, a, bs, bp, c, xs, xp, dx, tmp, k)
        tau = max(reltol*norm(x,Inf),abstol) # allowable error

        # Update the solution only if the error is acceptable
        if delta <= tau
            t = t + h
            x, xp = xp, x
            fout(t, x)
            if abs(1-c[end])<10*eps(1.)
                k[1], k[end] = k[end], k[1]
            else
                F(t,x,k[1]) # k[1] = F(t,x)
            end
        end

        # Update the step size
        h = min(hmax, 0.8*h*(tau/delta)^pow)
    end # while (t < tfinal) & (h >= hmin)

    if abs(t) < abs(tfinal)
      println("Step size grew too small. t=", t, ", h=", abs(h), ", x=", x)
    end
    return nothing
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
ode45(F, x0, tspan; kwargs...) = oderk(F, x0, tspan, dp_coefficients...; kwargs...)


function dmaster_nh{T<:Complex}(rho::Matrix{T}, Hnh, Hnh_dagger,
                    J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for i = 1:length(J)
        operators.gemm!(complex(1.), J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

function dmaster_nh{T<:Complex}(rho::Matrix{T}, Hnh, Hnh_dagger,
                    Gamma::Vector, J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for i = 1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

function dmaster_nh{T<:Complex}(rho::Matrix{T}, Hnh, Hnh_dagger,
                    Gamma::Matrix, J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for j=1:length(J), i = 1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)
    end
    return drho
end

function dmaster_h{T<:Complex}(rho::Matrix{T}, H,
                    Gamma::Vector, J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for i = 1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[i], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i], rho, Jdagger[i], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

function dmaster_h{T<:Complex}(rho::Matrix{T}, H,
                    Gamma::Matrix, J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for j=1:length(J), i = 1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[j], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i,j], rho, Jdagger[j], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

function integrate(dmaster::Function, tspan, rho0::Operator; fout=nothing, tmps_ode=nothing)
    if tmps_ode==nothing
        tmps_ode = Matrix{Complex128}[(rho0*0).data for i=1:11]
    end
    if fout==nothing
        tout = Float64[]
        xout = Operator[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Operator(rho0.basis_l, rho0.basis_r, 1.*x))
            nothing
        end
        ode45(dmaster, tspan, rho0.data, fout=f, tmps=tmps_ode)
        return tout, xout
    else
        ode45(dmaster, tspan, rho0.data, fout=fout, tmps=tmps_ode)
        return nothing
    end
end

function integrate_dopri(dmaster::Function, tspan, rho0; fout=nothing, kwargs...)
    n = size(rho0.data, 1)
    if fout==nothing
        tout = Float64[]
        xout = Operator[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Operator(rho0.basis_l, reshape(1.*x,n,n)))
            nothing
        end
        ode_dopri.ode(dmaster, float(tspan), reshape(rho0.data,n*n), fout=f, kwargs...)
        return tout, xout
    else
        ode_dopri.ode(dmaster, float(tspan), reshape(rho0.data,n*n), fout=fout, kwargs...)
        return nothing
    end
end

function master(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma=[Complex(1.) for i=1:length(J)],
                Jdagger::Vector=map(dagger, J),
                fout=nothing, tmp::Operator=rho0*0, kwargs...)
    Gamma = complex(Gamma)
    f(t, rho, drho) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp.data)
    return integrate(f, tspan, rho0; fout=fout, kwargs...)
end

function master_nh(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma=[Complex(1.) for i=1:length(J)],
                Hdagger::AbstractOperator=dagger(H),
                Jdagger::Vector=map(dagger, J),
                fout=nothing, tmp::Operator=rho0*0, kwargs...)
    Gamma = complex(Gamma)
    f(t, rho, drho) = dmaster_nh(rho, H, Hdagger, Gamma, J, Jdagger, drho, tmp.data)
    return integrate(f, tspan, rho0; fout=fout, kwargs...)
end

function master_nh_dopri(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Gamma=[Complex(1.) for i=1:length(J)],
                Hdagger::AbstractOperator=dagger(H),
                Jdagger::Vector=map(dagger, J),
                fout=nothing, tmp::Operator=rho0*0, kwargs...)
    Gamma = complex(Gamma)
    n = size(rho0.data, 1)
    N = n^2
    f(t, rho, drho) = dmaster_nh(reshape(rho, n, n), H, Hdagger, Gamma, J, Jdagger, reshape(drho,n,n), tmp.data)
    return integrate_dopri(f, tspan, rho0; fout=fout, kwargs...)
end



function integrate_dopri_mcwf(dmaster::Function, jumpfun::Function, tspan, psi0; fout=nothing, kwargs...)
    if fout==nothing
        tout = Float64[]
        xout = Ket[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Ket(psi0.basis, 1.*x))
            nothing
        end
        ode_dopri.ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=f; kwargs...)
        return tout, xout
    else
        ode_dopri.ode_mcwf(dmaster, jumpfun, float(tspan), psi0.data, fout=fout; kwargs...)
        return nothing
    end
end

function jump{T<:Complex}(rng, t::Float64, psi::Vector{T}, J::Vector, psi_new::Vector{T})
    probs = zeros(Float64, length(J))
    for i=1:length(J)
        operators.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
        probs[i] = vecnorm(psi_new)
    end
    cumprobs = cumsum(probs./sum(probs))
    r = rand(rng)
    i = findfirst(cumprobs.>r)
    operators.gemv!(complex(1.)/probs[i], J[i], psi, complex(0.), psi_new)
    return nothing
end

function dmcwf_nh{T<:Complex}(psi::Vector{T}, Hnh, dpsi::Vector{T})
    operators.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return psi
end

function mcwf_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector;
                fout=nothing, kwargs...)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_dopri_mcwf(f, j, tspan, psi0, fout=fout; kwargs...)
end

end