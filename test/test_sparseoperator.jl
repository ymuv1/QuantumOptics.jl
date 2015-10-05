using Base.Test
using quantumoptics


# Set up operators
spinbasis = SpinBasis(1//2)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)

sx_sp = SparseOperator(sx)
sy_sp = SparseOperator(sy)

b = FockBasis(3)
I = identity(b)
I_dense = operators.identity(b)

s = tensor(sx, sy)
s_sp = tensor(sx_sp, sy_sp)

@test typeof(I) == SparseOperator
@test typeof(I_dense) == Operator
@test_approx_eq 0. norm((I_dense-full(I)).data)
@test_approx_eq 0. norm((s - full(s_sp)).data)
