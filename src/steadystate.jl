module steadystate

using ..states
using ..operators
using ..superoperators
using ..timeevolution
using ..metrics


type ConvergenceReached <: Exception end

function master(H::AbstractOperator, J::Vector;
                rho0::Operator=tensor(basis_ket(H.basis_l, 1), basis_bra(H.basis_r, 1)),
                eps::Float64=1e-3, hmin=1e-7,
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::Operator=deepcopy(rho0),
                kwargs...)
    t0 = 0.
    rho0 = deepcopy(rho0)
    function fout_steady(t, rho)
        if fout!=nothing
            fout(t, rho)
        end
        dt = t - t0
        drho = metrics.tracedistance(rho0, rho)
        t0 = t
        rho0.data[:] = rho.data
        if drho/dt < eps
            throw(ConvergenceReached())
        end
    end
    try
        timeevolution.master([0., Inf], rho0, H, J; Gamma=Gamma, Jdagger=Jdagger,
                            hmin=hmin, hmax=Inf,
                            tmp=tmp,
                            display_initialvalue=false,
                            display_finalvalue=false,
                            display_intermediatesteps=true,
                            fout=fout_steady, kwargs...)
    catch e
        if !isa(e, ConvergenceReached)
            rethrow(e)
        end
    end
    return rho0
end


function eigenvector(L::SparseSuperOperator;
              rho0::Operator=tensor(basis_ket(L.basis_r[1], 1), basis_bra(L.basis_r[2], 1)),)
    d, v, nconv, niter, nmult, resid = Base.eigs(L.data; nev=1, sigma=1e-30)
    data = reshape(v[:,1], length(L.basis_r[1]), length(L.basis_r[2]))
    op = Operator(L.basis_r[1], L.basis_r[2], data)
    return op/trace(op)
end

function eigenvector(L::SuperOperator;
              rho0::Operator=tensor(basis_ket(L.basis_r[1], 1), basis_bra(L.basis_r[2], 1)),)
    d, v = Base.eig(L.data)
    data = reshape(v[:,1], length(L.basis_r[1]), length(L.basis_r[2]))
    return Operator(L.basis_r[1], L.basis_r[2], data)
end

eigenvector(H::AbstractOperator, J::Vector; kwargs...) = eigenvector(liouvillian(H, J); kwargs...)


end # module
