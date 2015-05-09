using Base.Test
using quantumoptics


# Set up operators
sx = operators.sigmax
sy = operators.sigmay

sx_sp = SparseOperator(sx)
sy_sp = SparseOperator(sy)

b = FockBasis(3)
I = quantumoptics.operators_sparse.sparse_identity(b)


s = tensor(sx, sy)
s_sp = tensor(sx_sp, sy_sp)

@test_approx_eq 0. norm((Operator(I)-identity(b)).data)
@test_approx_eq 0. norm((s - Operator(s_sp)).data)
