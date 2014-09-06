module timeevolution

using ..operators
using ..inplacearithmetic

export master


function oderkf(F, x0, tspan, p, a, bs, bp; reltol = 1.0e-5, abstol = 1.0e-8, tmps=[])
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
    iadd!(x, x0)
    tout = t            # first output time
    xout = Array(typeof(x0), 1)
    #xout[1] = 1*x      # first output solution

    for i=length(tmps):(4+length(c))
        push!(tmps, Operator(x0.basis_l, x0.basis_r))
    end
    xs, xp, dx, tmp = tmps[1:4]

    #k = Array(typeof(x0), length(c))
    #k[1] = F(t,x) # first stage
    #k = [Operator(x0.basis_l, x0.basis_r) for i=1:length(c)]
    k = tmps[5:5+length(c)]
    set!(k[1], F(t,x))


    # xs = Operator(x0.basis_l, x0.basis_r)
    # xp = Operator(x0.basis_l, x0.basis_r)
    # dx = Operator(x0.basis_l, x0.basis_r)
    # tmp = Operator(x0.basis_l, x0.basis_r)

    while abs(t) != abs(tfinal) && abs(h) >= hmin
        if abs(h) > abs(tfinal-t)
            h = tfinal - t
        end

        #(p-1)th and pth order estimates
        #xs = x + h*bs[1]*k[1]
        mul!(k[1], h*bs[1], xs)
        iadd!(xs, x)
        #xp = x + h*bp[1]*k[1]
        mul!(k[1], h*bp[1], xp)
        iadd!(xp, x)

        for j = 2:length(c)
            #dx = a[j,1]*k[1]
            mul!(a[j,1], k[1], dx)
            for i = 2:j-1
                #dx += a[j,i]*k[i]
                mul!(a[j,i], k[i], tmp)
                iadd!(dx, tmp)
            end
            #k[j] = F(t + h*c[j], x + h*dx)
            mul!(h, dx, tmp)
            iadd!(tmp, x)
            set!(k[j], F(t + h*c[j], tmp))

            # compute the (p-1)th order estimate
            #xs = xs + h*bs[j]*k[j]
            mul!(h*bs[j], k[j], tmp)
            iadd!(xs, tmp)
            # compute the pth order estimate
            #xp = xp + h*bp[j]*k[j]
            mul!(h*bp[j], k[j], tmp)
            iadd!(xp, tmp)
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
            #push!(xout, x*1)

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

function substep{T}(a::Matrix{T}, beta::T, b::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2), i=1:size(b,1)
        result[i,j] = a[i,j] + beta*b[i,j]
    end
end

function scale{T}(alpha::T, a::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2), i=1:size(a,1)
        result[i,j] = alpha*a[i,j]
    end
end

function oderkf2{T}(F, tspan, x0::Matrix{T}, p, a, bs, bp; reltol = 1.0e-5, abstol = 1.0e-8,
                    tmps=Vector{Matrix{T}}, fout=(t,x)->nothing)
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
    x = 1*x0
    fout(t, x)


    xs, xp, dx, tmp = tmps[1:4]

    #k[1] = F(t,x) # first stage
    k = tmps[5:5+length(c)]
    F(t,x,k[1])

    while abs(t) != abs(tfinal) && abs(h) >= hmin
        if abs(h) > abs(tfinal-t)
            h = tfinal - t
        end

        #(p-1)th and pth order estimates
        #xs = x + h*bs[1]*k[1]
        substep(x, complex(h*bs[1]), k[1], xs)
        #xp = x + h*bp[1]*k[1]
        substep(x, complex(h*bp[1]), k[1], xp)

        for j = 2:length(c)
            #dx = a[j,1]*k[1]
            scale(complex(a[j,1]), k[1], dx)
            for i = 2:j-1
                #dx += a[j,i]*k[i]
                substep(dx, complex(a[j,i]), k[i], dx)
            end
            #k[j] = F(t + h*c[j], x + h*dx)
            substep(x, complex(h), dx, tmp)
            F(t + h*c[j], tmp, k[j])

            # compute the (p-1)th order estimate
            #xs = xs + h*bs[j]*k[j]
            substep(xs, complex(h*bs[j]), k[j], xs)

            # compute the pth order estimate
            #xp = xp + h*bp[j]*k[j]
            substep(xp, complex(h*bp[j]), k[j], xp)
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
            fout(t, x)

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
                F(t,x,k[1])
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
ode45(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, dp_coefficients...; kwargs...)

ode45_ver2(F, x0, tspan; kwargs...) = oderkf2(F, x0, tspan, dp_coefficients...; kwargs...)

function dmaster(rho::Operator, H::AbstractOperator, J::Vector, Jdagger::Vector, tmps::Vector{Operator})
    drho, tmp1, tmp2 = tmps
    zero!(drho)
    mul!(H, rho, tmp1)
    mul!(rho, H, tmp2)
    iadd!(drho, isub!(tmp1, tmp2))
    imul!(drho, Complex(0, -1.))
    for i = 1:length(J)
        mul!(J[i], rho, tmp1)
        mul!(tmp1, Jdagger[i], tmp2)
        iadd!(drho, tmp2)

        mul!(J[i], rho, tmp1)
        mul!(Jdagger[i], tmp1, tmp2)
        imul!(tmp2, Complex(-0.5))
        iadd!(drho, tmp2)

        mul!(rho, Jdagger[i], tmp1)
        mul!(tmp1, J[i], tmp2)
        imul!(tmp2, Complex(-0.5))
        iadd!(drho, tmp2)
    end
    return drho
end

function master5(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector,
                Jdagger, tmps::Vector{Operator})
    for i=length(tmps):3
        push!(tmps, Operator(rho0.basis_l, rho0.basis_r))
    end
    f(t::Float64,rho::Operator) = dmaster(rho, H, J, Jdagger, tmps[1:3])#drho, tmp1, tmp2)
    tout, rho_t = ode45(f, rho0, float(T), tmps=tmps[4:end])
    return tout, rho_t
end

function dmaster_nh{T<:Complex}(rho::Matrix{T}, Heff, Heff_dagger,
                    J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    inplacearithmetic.gemm!(complex(1.), Heff, rho, complex(0.), drho)
    inplacearithmetic.gemm!(complex(1.), rho, Heff_dagger, complex(1.), drho)
    for i = 1:length(J)
        inplacearithmetic.gemm!(complex(1.), J[i], rho, complex(0.), tmp)
        inplacearithmetic.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

function dmaster_h{T<:Complex}(rho::Matrix{T}, H,
                    J::Vector, Jdagger::Vector,
                    drho::Matrix{T}, tmp::Matrix{T})
    inplacearithmetic.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    inplacearithmetic.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for i = 1:length(J)
        inplacearithmetic.gemm!(complex(1.), J[i], rho, complex(0.), tmp)
        inplacearithmetic.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)

        inplacearithmetic.gemm!(complex(-0.5), Jdagger[i], tmp, complex(1.), drho)

        inplacearithmetic.gemm!(complex(1.), rho, Jdagger[i], complex(0.), tmp)
        inplacearithmetic.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

function master2(T::Vector, rho0::Operator, Heff::Operator, Heff_dagger::Operator, J::Vector{Operator},
                Jdagger::Vector{Operator}, tmp::Matrix{Complex128}, tmps::Vector{Matrix{Complex128}})
    # for i=length(tmps):3
    #     push!(tmps, Operator(rho0.basis_l, rho0.basis_r))
    # end
    f(t::Float64, rho, drho) = dmaster_Heff(rho, Heff, Heff_dagger, J, Jdagger, drho, tmp)
    x = ode45_ver2(f, rho0.data, T, tmps=tmps)
    return x
    #return tout, rho_t
end

function master(dmaster::Function, tspan, rho0::Operator; fout=nothing, tmps_ode=Matrix{Complex128}[])
    for i=length(tmps_ode):11
        push!(tmps_ode, Operator(rho0.basis_l, rho0.basis_r).data)
    end
    if fout==nothing
        tout = Float64[]
        xout = Operator[]
        function f(t, x)
            push!(tout, t)
            push!(xout, Operator(rho0.basis_l, rho0.basis_r, 1.*x))
            nothing
        end
        ode45_ver2(dmaster, tspan, rho0.data, fout=f, tmps=tmps_ode)
        return tout, xout
    else
        ode45_ver2(dmaster, tspan, rho0.data, fout=fout, tmps=tmps_ode)
        return nothing
    end
end

function master_h(tspan, rho0::Operator, H::AbstractOperator, J::Vector{AbstractOperator};
                Jdagger::Vector{AbstractOperator}=map(dagger, J),
                fout=nothing, tmp::Operator=rho0*0, kwargs...)
    f(t, rho, drho) = dmaster_h(rho, H, J, Jdagger, drho, tmp.data)
    return master(f, tspan, rho0; fout=fout, kwargs...)
end

function master_nh(tspan, rho0::Operator, H::AbstractOperator, J::Vector;
                Hdagger::AbstractOperator=dagger(H),
                Jdagger::Vector=map(dagger, J),
                fout=nothing, tmp::Operator=rho*0, kwargs...)
    f(t, rho, drho) = dmaster_nh(rho, H, Hdagger, J, Jdagger, drho, tmp.data)
    return master(f, tspan, rho0; fout=fout, kwargs...)
end

function dmaster_nondiag(rho::Operator, H::AbstractOperator, gamma::Matrix,
                         J::Vector, Jdagger::Vector,
                         tmps::Vector{Operator})
    drho, tmp1, tmp2 = tmps
    zero!(drho)
    mul!(H, rho, tmp1)
    mul!(rho, H, tmp2)
    iadd!(drho, isub!(tmp1, tmp2))
    imul!(drho, Complex(0, -1.))
    for m = 1:length(J), n=1:length(J)
        mul!(J[m], rho, tmp1)
        mul!(tmp1, Jdagger[n], tmp2)
        imul!(tmp2, gamma[m,n])
        iadd!(drho, tmp2)

        mul!(J[m], rho, tmp1)
        mul!(Jdagger[n], tmp1, tmp2)
        imul!(tmp2, Complex(-0.5)*gamma[m,n])
        iadd!(drho, tmp2)

        mul!(rho, Jdagger[n], tmp1)
        mul!(tmp1, J[m], tmp2)
        imul!(tmp2, Complex(-0.5)*gamma[m,n])
        iadd!(drho, tmp2)
    end
    return drho
end


function master_nondiag(T::Vector, rho0::Operator, H::AbstractOperator, gamma::Matrix,
                        J::Vector;
                        Jdagger::Vector=map(dagger,J), tmps::Vector{Operator}=[])
    for i=length(tmps):3
        push!(tmps, Operator(rho0.basis_l, rho0.basis_r))
    end
    f(t::Float64,rho::Operator) = dmaster_nondiag(rho, H, gamma, J, Jdagger, tmps[1:3])#drho, tmp1, tmp2)
    tout, rho_t = ode45(f, rho0, float(T), tmps=tmps[4:end])
    return tout, rho_t
end

end