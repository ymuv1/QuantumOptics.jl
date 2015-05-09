using Base.Test
using quantumoptics.sparsematrix

SparseMatrix = quantumoptics.sparsematrix.SparseMatrix

# Set up test matrices
A = rand(Complex128, 5, 5)
A_sp = quantumoptics.sparsematrix.SparseMatrix(A)

B = eye(Complex128, 5)
B_sp = quantumoptics.sparsematrix.sparse_eye(Complex128, 5)

C = rand(Complex128, 3, 3)
C[2,:] = 0
C_sp = quantumoptics.sparsematrix.SparseMatrix(C)

R_sp = A_sp + B_sp
R = A + B


# Test arithmetic
@test_approx_eq 0. norm(full(R_sp) - R)
@test_approx_eq 0. norm(full(Complex128(0.5,0)*A_sp) - 0.5*A)
@test_approx_eq 0. norm(full(A_sp/2) - A/2)
@test_approx_eq 0. norm(full(A_sp*B_sp) - A*B)

# Test kronecker product
@test_approx_eq 0. norm(full(kron(A_sp, C_sp)) - kron(A, C))
@test_approx_eq 0. norm(full(kron(A_sp, B_sp)) - kron(A, B))

