using Base.Test
using QuantumOptics

particlenumber = 3
modenumber = 3
b_spin = SpinBasis(1//2)

m = complex(randn(modenumber, modenumber))
b = BosonicNParticleBasis(particlenumber, modenumber)
op = DenseOperator(GenericBasis([modenumber]), m)
op2 = SparseOperator(GenericBasis([modenumber]), sparse(m))
op_ = nparticleoperator_1(b, op)
op2_ = nparticleoperator_1(b, op2)
tic()
op_ = nparticleoperator_1(b, op)
toc()
tic()
op2_ = nparticleoperator_1(b, op2)
toc()

@test tracedistance(op_, full(op2_)) < 1e-12


m = complex(randn(modenumber^2, modenumber^2))
b = BosonicNParticleBasis(particlenumber, modenumber)
op = DenseOperator(GenericBasis([modenumber^2]), m)
op2 = SparseOperator(GenericBasis([modenumber^2]), sparse(m))
op_ = nparticleoperator_2(b, op)
op2_ = nparticleoperator_2(b, op2)
tic()
op_ = nparticleoperator_2(b, op)
toc()
tic()
op2_ = nparticleoperator_2(b, op2)
toc()

@test tracedistance(op_, full(op2_)) < 1e-12

# print(op_)
# print(op2_)

# println(length(b.occupations))
# println(length(b))