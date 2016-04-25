module cumulantexpansion

using ..ode_dopri
using ..operators_lazy

function dmaster(rho0::LazyTensor, H::LazySum,
                J::Vector{LazySum}, Jdagger::Vector{LazySum}, JdaggerJ::Vector{LazySum},
                drho::LazyTensor, tmp::LazyTensor)
    for drho_alpha in values(drho.operators)
        fill!(drho_alpha.data, 0.)
    end
    for (k, h_k) in H.operators
        factors = Dict{Int,Complex128}()
        for (gamma, h_k_gamma) in h_k.operators
            if haskey(rho0.operators, gamma)
                operators.gemm!(complex(1.,0.), h_k_gamma, rho0.operators[gamma], complex(0.), tmp.operators[gamma])
                factors[gamma] = trace(tmp.operators[gamma])
            else
                factors[gamma] = trace(h_k_gamma)
            end
        end
        for (alpha, h_k_alpha) in h_k.operators
            if !haskey(rho.operators, alpha)
                continue
            end
            factor = 1.
            for gamma in keys(h_k.operators)
                if alpha!=gamma
                    factor *= factors[gamma]
                end
            end
            operators.gemm!(factor*complex(0,-1.), h_k_alpha, rho.operators[alpha], complex(1.), drho.operators[alpha])
            operators.gemm!(factor*complex(0,1.), rho.operators[alpha], h_k_alpha, complex(1.), drho.operators[alpha])
        end
    end
    for k=1:length(J.operators)
        factors = Dict{Int,Complex128}()
        for gamma in keys(J.operators[k].operators)
            if haskey(rho0.operators, gamma)
                operators.gemm!(complex(1.,0.), JdaggerJ.operators[k].operators[gamma], rho0.operators[gamma], complex(0.), tmp.operators[gamma])
                factors[gamma] = trace(tmp.operators[gamma])
            else
                factors[gamma] = trace(JdaggerJ.operators[k].operators[gamma])
            end
        end
        for alpha in keys(J.operators[k].operators)
            if !haskey(rho.operators, alpha)
                continue
            end
            factor = 1.
            for gamma in keys(J.operators[k].operators)
                if alpha!=gamma
                    factor *= factors[gamma]
                end
            end
            operators.gemm!(complex(2*factor), J.operators[k].operators[alpha], rho.operators[alpha], complex(0.), tmp.operators[alpha])
            operators.gemm!(complex(1.), tmp.operators[alpha], J.operators[k].operators[alpha], complex(1.), drho.operators[alpha])
            operators.gemm!(complex(-factor), JdaggerJ.operators[k].operators[alpha], rho.operators[alpha], complex(1.), drho.operators[alpha])
            operators.gemm!(complex(-factor), rho.operators[alpha], JdaggerJ.operators[k].operators[alpha], complex(1.), drho.operators[alpha])
        end
    end
end

function productstatedimension(rho::LazyTensor)
    N = 0
    for (alpha, rho_alpha) in rho.operators
        N += length(rho_alpha.basis_l)*length(rho_alpha.basis_r)
    end
end

function as_vector(rho::LazyTensor, x::Vector{Complex128})
    N = productstatedimension(rho)
    @assert length(x) == N
    i = 1
    for alpha in sort(alphas)
        rho_alpha = rho.operators[alpha]
        N = length(rho_alpha.basis_l)*length(rho_alpha.basis_r)
        x[i:i+N] = reshape(rho_alpha.data, N)
        i += N
    end
    x
end

function as_operator(x::Vector{Complex128}, tmp::LazyTensor)
    N = 0
    alphas = Int[]
    for (alpha, rho_alpha) in tmp.operators
        N += length(rho_alpha.basis_l)*length(rho_alpha.basis_r)
        push!(alphas, alpha)
    end
    i = 1
    for alpha in sort(alphas)
        rho_alpha = tmp.operators[alpha]
        nl = length(rho_alpha.basis_l)
        nr = length(rho_alpha.basis_r)
        N = nl*nr
        rho_alpha.data = reshape(x[i:i+N], nl, nr)
        i += N
    end
    tmp
end

function master(tspan, rho0::LazyTensor, H::LazySum, J::Vector{LazySum};
                fout::Union{Function,Void}=nothing,
                kwargs...)
    N = productstatedimension(rho)
    f = (x->x)
    if fout==nothing
        tout = Float64[]
        xout = LazyTensor[]
        function fout_(t, rho::LazyTensor)
            push!(tout, t)
            push!(xout, deepcopy(rho))
        end
        f = fout_
    else
        f = fout
    end
    Jdagger = LazySum[dagger(j) for j=J]
    JdaggerJ = LazySum[dagger(j)*j for j=J]
    rho = deepcopy(rho0)
    drho = deepcopy(rho0)
    tmp = deepcopy(rho0)

    f_(t, x::Vector{Complex128}) = f(t, as_operator(x))
    function dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        dmaster(as_operator(x, rho), H, J, Jdagger, JdaggerJ, drho, tmp)
        as_vector(drho, dx)
    end
    ode(dmaster_, float(tspan), as_vector(rho0, zeros(Complex128, N)), f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end



end # module