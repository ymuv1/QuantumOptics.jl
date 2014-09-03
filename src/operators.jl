module operators

using ..bases, ..states

importall ..states, ..inplacearithmetic

export AbstractOperator, Operator,
	   tensor, dagger, expect,
	   identity, number, destroy, create,
	   sigmax, sigmay, sigmaz, sigmap, sigmam, spinbasis,
	   mul!, add!, sub!, imul!, iadd!, isub!, set!, zero!


abstract AbstractOperator

type Operator <: AbstractOperator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex{Float64}}
end

Operator(b::Basis, data) = Operator(b, b, data)
Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(Complex, length(b1), length(b2)))
Operator(b::Basis) = Operator(b, b)


*(a::Operator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::Operator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::Operator, b::Operator) = (check_multiplicable(a.basis_r, b.basis_l); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? Operator(a.basis_l, a.basis_r, a.data+b.data) : throw(IncompatibleBases()))

-(a::Operator, b::Operator) = ((a.basis_l==b.basis_l) && (a.basis_r==b.basis_r) ? Operator(a.basis_l, a.basis_r, a.data-b.data) : throw(IncompatibleBases()))


tensor(a::Operator, b::Operator) = Operator(compose(a.basis_l, b.basis_l), compose(a.basis_r, b.basis_r), kron(a.data, b.data))
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(a.data, b.data), prod(a.basis.shape), prod(b.basis.shape)))

dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')

Base.norm(op::Operator, p) = norm(op.data, p)
Base.trace(op::Operator) = trace(op.data)

expect(op::AbstractOperator, state::Operator) = trace(op*state)
expect(op::AbstractOperator, states::Vector{Operator}) = [expect(op, state) for state=states]
identity(b::Basis) = Operator(b, b, eye(Complex, length(b)))
number(b::Basis) = Operator(b, b, diagm(map(Complex, 0:(length(b)-1))))
destroy(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),1))
create(b::Basis) = Operator(b, b, diagm(map(Complex, sqrt(1:(length(b)-1))),-1))

const spinbasis = GenericBasis([2])
const sigmax = Operator(spinbasis, [0 1;1 0])
const sigmay = Operator(spinbasis, [0 -1im;1im 0])
const sigmaz = Operator(spinbasis, [1 0;0 -1])
const sigmap = Operator(spinbasis, [0 0;1 0])
const sigmam = Operator(spinbasis, [0 1;0 0])

check_equal_bases(a::AbstractOperator, b::AbstractOperator) = (check_equal(a.basis_l,b.basis_l); check_equal(a.basis_r,b.basis_r))

function mul!(a::Operator, b::Operator, result::Operator)
	check_multiplicable(a.basis_r, b.basis_l)
	check_equal(a.basis_l, result.basis_l)
	check_equal(b.basis_r, result.basis_r)
	mul!(a.data, b.data, result.data)
	result
end
mul!(a::Operator, b::Number, result::Operator) = (check_equal_bases(a, result); mul!(a.data, complex(b), result.data); result)
mul!(a::Number, b::Operator, result::Operator) = mul!(b,a,result)
imul!(a::Operator, b::Number) = imul!(a.data, complex(b))

add!(a::Operator, b::Operator, result::Operator) = (check_equal_bases(a, b); check_equal_bases(a, result); add!(a.data, b.data, result.data); result)
iadd!(a::Operator, b::Operator) = (check_equal_bases(a, b); iadd!(a.data, b.data); a)

sub!(a::Operator, b::Operator, result::Operator) = (check_equal_bases(a, b); check_equal_bases(a, result); sub!(a.data, b.data, result.data); result)
isub!(a::Operator, b::Operator) = (check_equal_bases(a, b); isub!(a.data, b.data); a)

set!(a::Operator, b::Operator) = (check_equal_bases(a, b); set!(a.data, b.data); a)
zero!(a::Operator) = fill!(a.data, zero(eltype(a.data)))

end