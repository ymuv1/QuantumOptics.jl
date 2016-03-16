using Base.Test
using Quantumoptics

fockbasis = FockBasis(40)
spinbasis = SpinBasis(1//2)

alpha = 0.5
a = full(destroy(fockbasis))
at = full(create(fockbasis))
n = full(number(fockbasis))

sx = full(sigmax(spinbasis))
sy = full(sigmay(spinbasis))
sz = full(sigmaz(spinbasis))
sp = full(sigmap(spinbasis))
sm = full(sigmam(spinbasis))

xket = coherentstate(fockbasis, alpha)
xbra = dagger(xket)
op1 = DenseOperator(spinbasis, GenericBasis([3]), [1 1 1; 1 1 1])
op2 = DenseOperator(GenericBasis([3]), spinbasis, [1 1; 1 1; 1 1])
I = dense_identity(fockbasis)


# Test creation
@test_throws DimensionMismatch DenseOperator(spinbasis, [1 1 1; 1 1 1])
@test_throws DimensionMismatch DenseOperator(spinbasis, FockBasis(3), [1 1; 1 1; 1 1])
@test_approx_eq 0. maximum(abs((dagger(op1)-op2).data))

# Test addition
@test_approx_eq 0. tracedistance(sx, sp + sm)
@test_throws bases.IncompatibleBases op1+op2

# Test substraction
@test_approx_eq 0. tracedistance(sy, -1im*(sp - sm))
@test_throws bases.IncompatibleBases op1-op2

# Test multiplication
@test_approx_eq 0. norm(I*xket - xket)
@test_approx_eq 0. norm(xbra*I - xbra)
@test_approx_eq alpha norm(a*xket)
@test_approx_eq alpha norm(xbra*at)
@test_approx_eq_eps 0. tracedistance(n, at*a) 1e-13
@test_approx_eq_eps 0. norm((5. * at)*xket - 5 * (at*xket)) 1e-13
@test_approx_eq_eps 0. norm((at * 5.)*xket - (at*xket) * 5) 1e-13
@test_throws bases.IncompatibleBases a*op1

# Test division
@test_approx_eq_eps 0. norm((at/5.)*xket - (at*xket)/5) 1e-13

# Test trace and normalize
op = DenseOperator(GenericBasis([3]), [1 3 2;5 2 2;-1 2 5])
@test_approx_eq 8. trace(op)
op_normalized = normalize(op)
@test_approx_eq 8. trace(op)
@test_approx_eq 1. trace(op_normalized)
op_ = normalize!(op)
@test op_ === op
@test_approx_eq 1. trace(op)

# Test identity function
@test full(I) == dense_identity(a)

# Test gemv implementation
result_ket = deepcopy(xket)
operators.gemv!(complex(1.0), at, xket, complex(0.), result_ket)
@test_approx_eq 0. norm(result_ket-at*xket)

result_bra = deepcopy(xbra)
operators.gemv!(complex(1.0), xbra, at, complex(0.), result_bra)
@test_approx_eq 0. norm(result_bra-xbra*at)
