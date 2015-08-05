using Base.Test
using quantumoptics

fockbasis = FockBasis(20)

alpha = 0.5
a = destroy(fockbasis)
at = create(fockbasis)
n = number(fockbasis)
xket = coherentstate(fockbasis, alpha)
xbra = dagger(xket)
op1 = Operator(spinbasis, GenericBasis([3]), [1 1 1; 1 1 1])
op2 = Operator(GenericBasis([3]), spinbasis, [1 1; 1 1; 1 1])
I = identity(fockbasis)


# Test creation
@test_throws DimensionMismatch Operator(spinbasis, [1 1 1; 1 1 1])
@test_throws DimensionMismatch Operator(spinbasis, FockBasis(3), [1 1; 1 1; 1 1])
@test_approx_eq 0. maximum(abs((dagger(op1)-op2).data))

# Test addition
@test_approx_eq 0. tracedistance(sigmax, sigmap + sigmam)
@test_throws bases.IncompatibleBases op1+op2

# Test substraction
@test_approx_eq 0. tracedistance(sigmay, -1im*(sigmap - sigmam))
@test_throws bases.IncompatibleBases op1-op2

# Test multiplication
@test_approx_eq 0. norm(I*xket - xket)
@test_approx_eq 0. norm(xbra*I - xbra)
@test_approx_eq alpha norm(a*xket)
@test_approx_eq alpha norm(xbra*at)
@test_approx_eq_eps 0. tracedistance(n, at*a) 1e-14
@test_approx_eq_eps 0. norm((5. * at)*xket - 5 * (at*xket)) 1e-14
@test_approx_eq_eps 0. norm((at * 5.)*xket - (at*xket) * 5) 1e-14
@test_throws bases.IncompatibleBases a*op1

# Test division
@test_approx_eq_eps 0. norm((at/5.)*xket - (at*xket)/5) 1e-14

# Test gemv implementation
result_ket = deepcopy(xket)
operators.gemv!(complex(1.0), at, xket, complex(0.), result_ket)
@test_approx_eq 0. norm(result_ket-at*xket)

result_bra = deepcopy(xbra)
operators.gemv!(complex(1.0), xbra, at, complex(0.), result_bra)
@test_approx_eq 0. norm(result_bra-xbra*at)
