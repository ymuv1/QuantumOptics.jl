using Base.Test
using Quantumoptics

particlenumber = 12
b_spin = SpinBasis(1//2)

b = FermionicNParticleBasis(particlenumber, 16)
# println(length(b.occupations))
# println(length(b))