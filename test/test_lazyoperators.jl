using Base.Test
using quantumoptics

alpha = 0.5

fockbasis = FockBasis(20)
spinbasis = SpinBasis(1//2)

a = destroy(fockbasis)
at = create(fockbasis)
n = number(fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)

# LazyTensor
basis = compose(spinbasis, fockbasis, spinbasis)
psi_ket = tensor(basis_ket(spinbasis, 2), coherentstate(fockbasis, alpha), basis_ket(spinbasis, 1))
psi_bra = dagger(psi_ket)
op1 = LazyTensor(basis, basis, 2, a)
op2 = LazyTensor(basis, basis, 3, sy)
op3 = LazyTensor(basis, basis, [2,3], [a,sy])
op4 = LazyTensor(basis, basis, [1,2,3], [sx,a,sy])
result_ket = deepcopy(psi_ket)
result_bra = deepcopy(psi_bra)

for op=[op1, op2, op3, op4]
    operators.gemv!(Complex(1.), op, psi_ket, Complex(0.), result_ket)
    @test_approx_eq_eps 0. norm(full(op)*psi_ket - result_ket) 1e-12
    operators.gemv!(Complex(1.), psi_bra, op, Complex(0.), result_bra)
    @test_approx_eq_eps 0. norm(psi_bra*full(op) - result_bra) 1e-12
end


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
