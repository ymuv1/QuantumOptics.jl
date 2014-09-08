require("../src/quantumoptics.jl")

using quantumoptics

srand(0)

N1 = 2
N2 = 1
N3 = 2^1
N = N1*N2*N3

b1 = GenericBasis([N1])
b2 = GenericBasis([N2])
b3 = GenericBasis([N3])

basis_total = CompositeBasis(b1, b2, b3)

op1 = Operator(b1, rand(N1,N1))
op2 = Operator(b2, rand(N2,N2))

psi = Ket(basis_total, rand(N))

psi2 = Ket(basis_total, rand(N))
psi = tensor(psi, dagger(psi))
psi2 = tensor(psi2, dagger(psi2))

l1 = operators_lazy.LazyTensor(basis_total, basis_total, [1], [op1])
l2 = operators_lazy.LazyTensor(basis_total, basis_total, [2], [op2])
l3 = operators_lazy.LazyTensor(basis_total, basis_total, [1,2], [op1,op2])

#println(full(l).data)
#println(tensor(a, identity(basis)).data)
# operators_lazy.gemm!(Complex(1.), l3, psi.data, Complex(0.), psi2.data)
# println(vecnorm(vec((psi2-full(l3)*psi).data)))
# println(vecnorm(vec((l3*psi-full(l3)*psi).data)))

operators_lazy.gemm!(Complex(1.), l1, psi.data, Complex(0.), psi2.data)
#@time operators_lazy.gemm!(Complex(1.), psi.data, l3, Complex(0.), psi2.data)
println(vecnorm(vec((psi2-full(l1)*psi).data)))
#println(vecnorm(vec((psi*l1-psi*full(l1)).data)))

# operators_lazy.gemm!(Complex(1.), (l2+l3), psi.data, Complex(0.), psi2.data)
# println(vecnorm(vec(((l3+l2)*psi - psi2).data)))
# println(vecnorm(vec(((l3+l2)*psi - l3*psi - l2*psi).data)))

# 0.0 + 0.0im
# a_data_before
# Complex{Float64}[0.04140270967258792 + 0.0im
#                  0.008607397383288657 + 0.0im]

# Complex{Float64}[0.013891193130373177 + 0.0im
#                  0.07362359555699248 + 0.0im]
# index: 1
# a: Complex{Float64}[0.04140270967258792 + 0.0im
#                  0.008607397383288657 + 0.0im]

# Complex{Float64}[0.013891193130373177 + 0.0im
#                  0.07362359555699248 + 0.0im]
# b: Complex{Float64}[0.8236475079774124 + 0.0im 0.16456579813368521 + 0.0im
#                  0.9103565379264364 + 0.0im 0.17732884646626457 + 0.0im]
# result beforeComplex{Float64}[0.0 + 0.0im
#                  0.0 + 0.0im]

# Complex{Float64}[0.0 + 0.0im
#                  0.0 + 0.0im]
# result afterComplex{Float64}[0.046747177131173365 + 0.0im
#                  0.07411318296587685 + 0.0im]

# Complex{Float64}[0.009276779216015853 + 0.0im
#                  0.014472070493054968 + 0.0im]
# tmp1_after
# Complex{Float64}[0.046747177131173365 + 0.0im
#                  0.07411318296587685 + 0.0im]

# Complex{Float64}[0.009276779216015853 + 0.0im
#                  0.014472070493054968 + 0.0im]


# operators_lazy.gemm!(Complex(1.), psi.data, (l2+l3), Complex(0.), psi2.data)
# println(vecnorm(vec((psi*(l3+l2) - psi2).data)))
# println(vecnorm(vec((psi*(l3+l2) - psi*l3 - psi*l2).data)))

error()
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

# s = (l3+l2)

# #@time l3*rho
# #@time l2*rho

# result = s*rho
# @time s*rho
# operators_lazy.mul!(s,rho,result)
# @time operators_lazy.mul!(s,rho,result)
# #@time l3*psi

# result1 = l3*rho
# @time operators_lazy.mul!(l3, rho, result1)

# result2 = l2*rho
# @time operators_lazy.mul!(l2, rho, result1)

# @time (operators_lazy.mul!(l3, rho, result1); operators_lazy.mul!(l2, rho, result2); operators.iadd!(result1,result2))

# #@profile operators_lazy.mul!(l3, rho, result)

#full_l1*psi
#@time full_l1*psi
