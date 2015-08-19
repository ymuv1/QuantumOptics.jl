using Base.Test
using quantumoptics

alpha = 0.5

fockbasis = FockBasis(20)
basis = compose(fockbasis, spinbasis)

a = destroy(fockbasis)
at = create(fockbasis)
n = number(fockbasis)

psi_ket = tensor(coherentstate(fockbasis, alpha), basis_ket(spinbasis, 1))
psi_bra = dagger(psi_ket)


# LazyTensor
op = LazyTensor(basis, basis, 1, a) + LazyTensor(basis, basis, 1, at)


# LazySum
psi_ket = coherentstate(fockbasis, alpha)
psi_bra = dagger(psi_ket)
op = LazySum(a, at)
@test typeof(op) == LazySum
@test_approx_eq_eps 0. norm(op*psi_ket - (a+at)*psi_ket) 1e-12

psi_ket2 = Ket(fockbasis)
operators.gemv!(Complex(1.), LazySum(a, at, at*a), psi_ket, Complex(0.), psi_ket2)
@test_approx_eq_eps 0. norm((a+at+at*a)*psi_ket - psi_ket2) 1e-12

psi_bra2 = Bra(fockbasis)
operators.gemv!(Complex(1.), psi_bra, LazySum(a, at, at*a), Complex(0.), psi_bra2)
@test_approx_eq_eps 0. norm(psi_bra*(a+at+at*a) - psi_bra2) 1e-12


# LazyProduct
n_lazy = LazyProduct(at, a)
@test_approx_eq_eps 0. norm(n_lazy*psi_ket - n*psi_ket) 1e-12
@test_approx_eq_eps 0. norm(psi_bra*n_lazy - psi_bra*n) 1e-12

zero_op = n_lazy - n
@test typeof(zero_op) == LazySum
@test_approx_eq_eps 0. norm(zero_op*psi_ket) 1e-12
@test_approx_eq_eps 0. norm(psi_bra*zero_op) 1e-12

psi_ket_ = deepcopy(psi_ket)
psi_ket2 = Ket(fockbasis)
operators.gemv!(Complex(1.), LazyProduct(a, at, at*a), psi_ket_, Complex(0.), psi_ket2)
@test_approx_eq_eps 0. norm(a*at*at*a*psi_ket - psi_ket2) 1e-12

psi_bra_ = deepcopy(psi_bra)
psi_bra2 = Bra(fockbasis)
operators.gemv!(Complex(1.), psi_bra_, LazyProduct(a, at, at*a), Complex(0.), psi_bra2)
@test_approx_eq_eps 0. norm(psi_bra*a*at*at*a - psi_bra2) 1e-12
