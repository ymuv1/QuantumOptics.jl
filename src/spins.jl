module spins

using ..bases, ..states, ..operators

export sigmax, sigmay, sigmaz, sigmap, sigmam, spinbasis


const spinbasis = GenericBasis([2])
const sigmax = Operator(spinbasis, [0 1;1 0])
const sigmay = Operator(spinbasis, [0 -1im;1im 0])
const sigmaz = Operator(spinbasis, [1 0;0 -1])
const sigmap = Operator(spinbasis, [0 1;0 0])
const sigmam = Operator(spinbasis, [0 0;1 0])


end #module
