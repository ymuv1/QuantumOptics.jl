require("../src/quantumoptics.jl")

using quantumoptics

N1 = 2
N2 = 2
N3 = 2^9
N = N1*N2*N3

b1 = GenericBasis([N1])
b2 = GenericBasis([N2])
b3 = GenericBasis([N3])

basis_total = CompositeBasis(b1, b2, b3)

op1 = Operator(b1, rand(N1,N1))
op2 = Operator(b2, rand(N2,N2))

psi = Ket(basis_total, rand(N))
psi2 = Ket(basis_total, rand(N))

l1 = operators_lazy.LazyTensor(basis_total, basis_total, [1], [op1])
l2 = operators_lazy.LazyTensor(basis_total, basis_total, [2], [op2])
l3 = operators_lazy.LazyTensor(basis_total, basis_total, [1,2], [op1,op2])

#println(full(l).data)
#println(tensor(a, identity(basis)).data)

# println((l*rho).data)
# println((full(l)*rho).data)
# println(vecnorm((l1*psi-full(l1)*psi).data))
# println(vecnorm((l2*psi-full(l2)*psi).data))
# println(vecnorm((l3*psi-full(l3)*psi).data))

rho = tensor(psi, dagger(psi2))
# println(norm((l3*rho-full(l3)*rho).data))

#full_l1 = full(l1)
#full_l3 = full(l3)

#full_l3*rho
#@time full_l3*rho

#l3*rho
#@time l3*rho

#println(l3+l2)
#println(vecnorm(((l3+l2)*psi-(l3*psi + l2*psi)).data))

s = (l3+l2)

#@time l3*rho
#@time l2*rho

result = s*rho
@time s*rho
operators_lazy.mul!(s,rho,result)
@time operators_lazy.mul!(s,rho,result)
#@time l3*psi

result1 = l3*rho
@time operators_lazy.mul!(l3, rho, result1)

result2 = l2*rho
@time operators_lazy.mul!(l2, rho, result1)

@time (operators_lazy.mul!(l3, rho, result1); operators_lazy.mul!(l2, rho, result2); operators.iadd!(result1,result2))

#@profile operators_lazy.mul!(l3, rho, result)

#full_l1*psi
#@time full_l1*psi