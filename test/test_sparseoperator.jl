using quantumoptics
using quantumoptics.operators_sparse

sx = operators.sigmax
sy = operators.sigmay

sx_sp = SparseOperator(sx)
sy_sp = SparseOperator(sy)

s = tensor(sx, sy)
s_sp = tensor(sx_sp, sy_sp) 

b = FockBasis(3)
I = sparse_identity(b)
@assert norm((Operator(I)-identity(b)).data)<1e-10
@assert norm((s - Operator(s_sp)).data)<1e-10
