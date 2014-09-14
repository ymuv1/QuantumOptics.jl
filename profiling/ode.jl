
const order = 5
const a2 = Float64[1/5]
const a3 = Float64[3/40, 9/40]
const a4 = Float64[44/45, -56/15, 32/9]
const a5 = Float64[19372/6561, -25360/2187, 64448/6561, -212/729]
const a6 = Float64[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
const a7 = Float64[35/384, 0., 500/1113, 125/192, -2187/6784, 11/84]
const a = {Float64[] a2 a3 a4 a5 a6 a7}
const bs = Float64[5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
const c = Float64[0., 1/5, 3/10, 4/5, 8/9, 1., 1.]


function substep{T}(x::Vector{T}, x0::Vector{T}, h::Float64, coeffs::Vector{Float64}, k::Vector{Vector{T}})
    @inbounds for m=1:length(x0)
        dx::T = 0.
        @inbounds for i=1:length(coeffs)
            dx += coeffs[i]::Float64*k[i][m]
        end
        x[m] = x0[m] + h*dx
    end
    return nothing
end

function step{T}(F::Function, t::Float64, h::Float64,
                x0::Vector{T}, xp::Vector{T}, xs::Vector{T}, k::Vector{Vector{T}})
    for i=2:length(c)
        substep(xp, x0, h, a[i], k)
        F(t + h*c[i], xp, k[i])
    end
    substep(xs, x0, h, bs, k)
    return nothing
end

function allocate_memory{T}(x::Vector{T})
    xp = zeros(T, length(x))
    xs = zeros(T, length(x))
    k = Vector{Vector{T}}[]
    for i=1:7
        push!(k, zeros(T, length(x))
    return xp, xs, k
end

function error_estimate(xp, xs, abstol, reltol)
    err::Float64 = 0.
    for i=1:length(xp)
        sc_i = abstol + reltol*max(abs(xp[i]), abs(xs[i]))
        err += abs2(xp[i]-xs[i])/sc_i^2
    return sqrt(err/length(xp))
end

function initial_stepsize(F, x, k, tmp1, tmp2)
    d0 = norm(x,2)
    d1 = norm(k[1],2)
    h0 = ((d0<1e-5 | d1<1e-5) ? 1e-6 : 0.01*d0/d1)
    substep(tmp1, x, h0, [1.], k)
    F(t+h0, tmp1, tmp2)
    for i=1:length(tmp2)
        tmp2[i] -= k[1][i]
    end
    d2 = norm(tmp2, 2)
    if max(d1,d2)<1e-15
        h1 = max(1e-6, h0*1e-3)
    else
        h1 = (0.01/max(d1,d2))^(1./order)
    return min(100*h0, h1)
end

function ode{T}(F, tspan::Vector{Float64}, x0::Vector{T};
                    reltol = 1.0e-5,
                    abstol = 1.0e-8,
                    h0 = 0,
                    hmin = (tspan[end]-tspan[1])/1e9,
                    hmax = (tspan[end]-tspan[1]),
                    fout=(t,x)->nothing,)
    t, tend = tspan[1], tspan[end]

    fout(t, x)
    x = 1*x0
    xp, xs, k = allocate_memory(x0)

    F(t,x,k[1])
    h = (h0==0 ? initial_stepsize(F, x, k, k[2], k[3]) : h0)
    was_rejected = false
    while t < tfinal
        step(F, t, h, x, xp, xs, k)
        err = error_estimate(xp, xs, abstol, reltol)
        facmin = (was_rejected ? 1. : 5.)
        hnew = h*min(facmin, max(0.2, 0.9*(1./err)^(1./order)))
        hnew = min(hmax, hnew)
        if hnew<hmin
            error("Stepsize below hmin.")
        end
        if err>1
            was_rejected = true
            continue
        end
        was_rejected = false
        xp, x = x, xp
        fout(t, x)
    end
    return nothing
end



