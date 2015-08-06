module operators

using ..bases, ..states
using Base.LinAlg.BLAS
using Base.Cartesian

importall ..states
importall ..bases

export AbstractOperator, Operator,
       tensor, dagger, expect, ptrace, embed, normalize, normalize!,
       identity, number, destroy, create,
       qfunc


abstract AbstractOperator


type Operator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex{Float64}}
    Operator(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

Operator(b::Basis, data) = Operator(b, b, data)
Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(Complex, length(b1), length(b2)))
Operator(b::Basis) = Operator(b, b)


check_samebases(a::AbstractOperator, b::AbstractOperator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)

*(a::Operator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::Operator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::Operator, b::Operator) = (check_multiplicable(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::Operator, b::Operator) = (check_samebases(a,b); Operator(a.basis_l, a.basis_r, a.data+b.data))
-(a::Operator, b::Operator) = (check_samebases(a,b); Operator(a.basis_l, a.basis_r, a.data-b.data))


tensor(a::Operator, b::Operator) = Operator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))
tensor(ops::Operator...) = reduce(tensor, ops)
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))


dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')
Base.full(x::Operator) = x

Base.norm(op::Operator, p) = norm(op.data, p)
Base.trace(op::Operator) = trace(op.data)
normalize(op::Operator) = op/trace(op)
function normalize!(op::Operator)
    u = 1./trace(op)
    for j=1:size(op.data,2), i=1:size(op.data,1)
        op.data[i,j] *= u
    end
    return op
end

basis{T<:AbstractOperator}(op::T) = (check_equal(op.basis_l, op.basis_r); op.basis_l)

expect(op::AbstractOperator, state::Operator) = trace(op*state)
expect(op::AbstractOperator, states::Vector{Operator}) = [expect(op, state) for state=states]
expect(op::AbstractOperator, state::Ket) = dagger(state)*(op*state)
expect(op::AbstractOperator, states::Vector{Ket}) = [expect(op, state) for state=states]

Base.identity(b::Basis) = Operator(b, b, eye(Complex, length(b)))
Base.identity(b1::Basis, b2::Basis) = Operator(b1, b2, eye(Complex, length(b1), length(b2)))
number(b::Basis) = Operator(b, b, diagm(map(Complex, 0:(length(b)-1))))
destroy(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),1))
create(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),-1))

check_equal_bases(a::AbstractOperator, b::AbstractOperator) = (check_equal(a.basis_l,b.basis_l); check_equal(a.basis_r,b.basis_r))

Base.zero{T<:AbstractOperator}(a::T) = T(a.basis_l, a.basis_r)
Base.one{T<:AbstractOperator}(a::T) = identity(a.basis_l, a.basis_r)
set!(a::Operator, b::Operator) = (check_equal_bases(a, b); set!(a.data, b.data); a)
zero!(a::Operator) = fill!(a.data, zero(eltype(a.data)))

gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::Matrix{T}, beta::T, result::Matrix{T}) = BLAS.gemm!('N', 'N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Matrix{T}, b::Vector{T}, beta::T, result::Vector{T}) = BLAS.gemv!('N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Vector{T}, b::Matrix{T}, beta::T, result::Vector{T}) = BLAS.gemv!('T', alpha, b, a, beta, result)

gemm!(alpha, a::Operator, b::Operator, beta, result::Operator) = gemm!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::Operator, b::Ket, beta, result::Ket) = gemv!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::Bra, b::Operator, beta, result::Bra) = gemv!(alpha, a.data, b.data, beta, result.data)

Base.prod{B<:Basis, T<:AbstractArray}(basis::B, operators::T) = (length(operators)==0 ? identity(basis) : prod(operators))

embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector) = tensor([prod(basis.bases[i], operators[find(indices.==i)]) for i=1:length(basis.bases)]...)
embed{T<:AbstractOperator}(basis::CompositeBasis, index::Int, op::T) = embed(basis, Int[index], T[op])


function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[N] = 1
    for m=N-1:-1:1
        S[m] = S[m+1]*shape[m+1]
    end
    return S
end

@generated function _ptrace{RANK}(rank::Array{Int,RANK}, a::Matrix{Complex128},
                                  shape_l::Vector{Int}, shape_r::Vector{Int},
                                  indices::Vector{Int})
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = deepcopy(shape_l)
        result_shape_l[indices] = 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = deepcopy(shape_r)
        result_shape_r[indices] = 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(Complex128, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                #println("Jl_0: ", Jl_0, "; Jr_0: ", Jr_0, "; Il_0: ", Il_0, "; Ir_0: ", Ir_0)
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

function ptrace(a::Operator, indices::Vector{Int})
    rank = zeros(Int, [0 for i=1:length(a.basis_l.shape)]...)
    result = _ptrace(rank, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

ptrace(a::Operator, indices::Int) = ptrace(a, Int[indices])
ptrace(a::Ket, indices) = ptrace(tensor(a, dagger(a)), indices)
ptrace(a::Bra, indices) = ptrace(tensor(dagger(a), a), indices)


function qfunc(rho::AbstractOperator, X::Vector{Float64}, Y::Vector{Float64})
    M = zeros(Float64, length(X), length(Y))
    @assert rho.basis_l == rho.basis_r
    for (i,x)=enumerate(X), (j,y)=enumerate(Y)
        z = complex(x,y)
        coh = coherent_state(rho.basis_l, z)
        M[i,j] = real(dagger(coh)*rho*coh)
    end
    return M
end

end # module
