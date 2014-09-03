module timeevolution

using ..operators

export master


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
    iadd!(x, x0)
    tout = t            # first output time
    xout = Array(typeof(x0), 1)
    xout[1] = 1*x      # first output solution

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
            push!(xout, x*1)

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

function dmaster(rho::Operator, H::AbstractOperator, J::Vector, Jdagger::Vector, drho::Operator, tmp1::Operator, tmp2::Operator)
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


function master(T::Vector, rho0::Operator, H::AbstractOperator, J::Vector)
    Jdagger = [dagger(j) for j=J]
    tmp1 = Operator(rho0.basis_l, rho0.basis_r)
    tmp2 = Operator(rho0.basis_l, rho0.basis_r)
    drho = Operator(rho0.basis_l, rho0.basis_l)
    f(t::Float64,rho::Operator) = dmaster(rho, H, J, Jdagger, drho, tmp1, tmp2)
    tout, rho_t = ode45(f, rho0, float(T))
    return tout, rho_t
end

end