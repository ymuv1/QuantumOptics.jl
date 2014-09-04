require("../src/quantumoptics.jl")

using quantumoptics

basis = FockBasis(2)
a = destroy(basis)

basis_total = bases.compose(basis, basis)

l = operators_lazy.LazyTensor(basis_total, basis_total, [1], [a])

println(full(l).data)
println(tensor(a, identity(basis)).data)

rho = identity(basis_total)
println((l*rho).data)
println((full(l)*rho).data)
println(norm(l*rho-full(l)*rho, 2))