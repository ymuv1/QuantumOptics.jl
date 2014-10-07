using quantumoptics
using quantumoptics.operators_sparse

sx = operators.sigmax
sy = operators.sigmay

sx_sp = SparseOperator(sx)
sy_sp = SparseOperator(sy)

s = tensor(sx, sy)
s_sp = tensor(sx_sp, sy_sp) 

@assert norm((s - Operator(s_sp)).data)<1e-10
