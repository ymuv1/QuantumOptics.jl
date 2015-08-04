using Base.Test
using quantumoptics

shape1 = [5]
shape2 = [2, 3]

b1 = GenericBasis(shape1)
b2 = GenericBasis(shape2)

@test b1.shape == shape1
@test b2.shape == shape2
@test b1 != b2
@test b1 == b1

comp_b1 = compose(b1, b2)
comp_b2 = compose(b1, b1, b2)
@test comp_b1.shape == [prod(shape1), prod(shape2)]
@test comp_b2.shape == [prod(shape1), prod(shape1), prod(shape2)]

comp_b1_b2 = compose(comp_b1, comp_b2)
@test comp_b1_b2.shape == [prod(shape1), prod(shape2), prod(shape1), prod(shape1), prod(shape2)]
@test comp_b1_b2 == CompositeBasis(b1, b2, b1, b1, b2)

@test comp_b2.shape == compose(b1, comp_b1).shape
@test comp_b2 == compose(b1, comp_b1)

@test ptrace(comp_b2, [1]) == ptrace(comp_b2, [2]) == comp_b1
@test ptrace(comp_b2, [1, 2]) == ptrace(comp_b1, [1])
@test ptrace(comp_b2, [2, 3]) == ptrace(comp_b1, [2])
