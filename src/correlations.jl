module correlations

using ..operators
using ..timeevolution
using ..metrics
using ..steadystate


function correlation(tspan::Vector{Float64}, rho0::Operator, H::AbstractOperator, J::Vector,
                     op1::AbstractOperator, op2::AbstractOperator;
                     Gamma::Union{Real, Vector, Matrix}=ones(Float64, length(J)),
                     Jdagger::Vector=map(dagger, J),
                     tmp::Operator=deepcopy(rho0),
                     kwargs...)
    exp_values = Complex128[]
    function fout(t, rho)
        push!(exp_values, expect(op1, rho))
    end
    timeevolution.master(tspan, op2*rho0, H, J; Gamma=Gamma, Jdagger=Jdagger,
                        tmp=tmp, fout=fout, kwargs...)
    return exp_values
end


function correlation(rho0::Operator, H::AbstractOperator, J::Vector,
                     op1::AbstractOperator, op2::AbstractOperator;
                     eps::Float64=1e-3,
                     Gamma::Union{Real, Vector, Matrix}=ones(Float64, length(J)),
                     Jdagger::Vector=map(dagger, J),
                     tmp::Operator=deepcopy(rho0),
                     kwargs...)
    op2rho0 = op2*rho0
    tout = Float64[0.]
    exp_values = Complex128[expect(op1, op2rho0)]
    function fout(t, rho)
        push!(tout, t)
        push!(exp_values, expect(op1, rho))
    end
    steadystate.master(H, J; rho0=op2rho0, eps=eps, fout=fout,
                       Gamma=Gamma, Jdagger=Jdagger, tmp=tmp, kwargs...)
    return tout, exp_values
end


end # module
