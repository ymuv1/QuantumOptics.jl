using Base.Test
using quantumoptics

N = 3
basis = FockBasis(N)
bra = Bra(basis)
ket = Ket(basis)

@test_throws DimensionMismatch Bra(basis, [1, 2])
@test_approx_eq 0. norm(bra-Bra(basis, zeros(Int, N)))
@test_approx_eq 0. norm(ket-Ket(basis, zeros(Int, N)))
@test_approx_eq 0. bra*ket
@test_throws bases.IncompatibleBases bra*Ket(FockBasis(N+1))


bra = Bra(basis, [1im, 0, 1])
ket = Ket(basis, [0, -1im, 1])

@test_approx_eq 0. norm(5*bra - Bra(basis, [5im, 0, 5]))
@test_approx_eq 0. norm(5*ket - Ket(basis, [0, -5im, 5]))
@test_approx_eq 0. norm(5*ket - ket/0.2)
@test_throws bases.IncompatibleBases bra + Bra(FockBasis(N+1))
@test_throws bases.IncompatibleBases ket + Ket(FockBasis(N+1))


basis = FockBasis(2)
a = normalized(Bra(basis, [1im, 0]))
b = normalized(Bra(basis, [1, 2]))
c = normalized(Bra(CompositeBasis(basis, basis), [1im, 2im, 0, 0]))
@test_approx_eq 0. norm(tensor(a, b) - c)
@test_approx_eq_eps 0. tracedistance(operators.ptrace(c, 1), tensor(dagger(b), b)) 1e-15
@test_approx_eq_eps 0. tracedistance(operators.ptrace(c, 2), tensor(dagger(a), a)) 1e-15
