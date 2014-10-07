using quantumoptics.sparse

SparseMatrix = quantumoptics.sparse.SparseMatrix

A = rand(Complex128,3,3)
A_sp = SparseMatrix(A)

B = rand(Complex128,3,3)
B[2,:] = 0
B_sp = SparseMatrix(B)

R_sp = A_sp + B_sp
R = A + B

@assert norm(full(R_sp) - R) < 1e-10
@assert norm(full(Complex128(0.5,0)*R_sp) - 0.5*R) < 1e^10
@assert norm(full(R_sp/2) - R/2) < 1e^10
@assert norm(full(A_sp*B_sp) -A*B) < 1e-10

@assert norm(full(kron(A_sp, B_sp)) - kron(A, B)) < 1e-10

