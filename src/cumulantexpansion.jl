module cumulantexpansion

using ..bases
using ..ode_dopri
using ..operators_lazy
using ..operators

import ..operators


type ProductDensityOperator <: Operator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    operators::Vector{DenseOperator}
    function ProductDensityOperator(operators::DenseOperator...)
        basis_l = tensor([op.basis_l for op in operators]...)
        basis_r = tensor([op.basis_r for op in operators]...)
        new(basis_l, basis_r, operators)
    end
end

dims(bl::CompositeBasis, br::CompositeBasis) = [length(bl_i)*length(br_i) for (bl_i, br_i) in zip(bl.bases, br.bases)]
dims(x::ProductDensityOperator) = dims(x.basis_l, x.basis_r)

traces(x::ProductDensityOperator) = [trace(op) for op in x.operators]

function fill!(x::ProductDensityOperator, alpha::Number)
    for op in x.operators
        fill!(op.data, complex(alpha))
    end
end

# Ignores the factor in the LazyTensor
function operators.gemm!(alpha, a::LazyTensor, b::ProductDensityOperator, beta, result::ProductDensityOperator)
    @assert abs(beta)==0.
    for (k, a_k) in a.operators
        operators.gemm!(1., a_k, b.operators[k], 0., result.operators[k])
    end
end

function dmaster(rho0::ProductDensityOperator, H::LazySum,
                 J::Vector{LazySum}, Jdagger::Vector{LazySum}, JdaggerJ::Vector{LazySum},
                 drho::ProductDensityOperator, tmp::ProductDensityOperator)
    fill!(drho.data, 0.)
    for (k, h_k) in H.operators
        operators.gemm!(1., h_k, rho0, 0., tmp)
        subtraces = traces(tmp)
        for (alpha, h_k_alpha) in h_k.operators
            factor = h_k.factor
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
        operators.gemm!(1., JdaggerJ.operators[k], rho0, 0., tmp)
        subtraces = traces(tmp)
        subindices = keys(J.operators[k].operators)
        for alpha in subindices
            factor = JdaggerJ.operators[k].factor
            for gamma in subindices
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

dims(rho::ProductDensityOperator) = [length(op.basis_l)*length(op.basis_r) for op in rho.operators]

function as_vector(rho::ProductDensityOperator, x::Vector{Complex128})
    i = 1
    for op in rho.operators
        N = length(op.basis_l)*length(op.basis_r)
        x[i:i+N] = reshape(op.data, N)
        i += N
    end
    x
end

function as_operator(x::Vector{Complex128}, rho::ProductDensityOperator)
    i = 1
    for op in rho.operators
        N = length(op.basis_l)*length(op.basis_r)
        reshape(op.data, N)[:] = x[i:i+N]
        i += N
    end
end

function master(tspan, rho0::ProductDensityOperator, H::LazySum, J::Vector{LazySum};
                fout::Union{Function,Void}=nothing,
                kwargs...)
    x0 = as_vector(rho0, zeros(Complex128, prod(dims(rho0))))
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
    ode(dmaster_, float(tspan), x0, f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end



end # module