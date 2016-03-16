using Base.Test
using quantumoptics


# Set up operators
spinbasis = SpinBasis(1//2)

sx_sp = sigmax(spinbasis)
sy_sp = sigmay(spinbasis)

sx = full(sx_sp)
sy = full(sy_sp)

b = FockBasis(3)
I = identity(b)
I_dense = dense_identity(b)


s = tensor(sx, sy)
s_sp = tensor(sx_sp, sy_sp)

@test typeof(I) == SparseOperator
@test typeof(I_dense) == DenseOperator
@test_approx_eq 0. norm((I_dense-full(I)).data)
@test_approx_eq 0. norm((s - full(s_sp)).data)

@test I == identity(destroy(b))
