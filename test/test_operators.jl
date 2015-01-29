using quantumoptics

fockbasis = FockBasis(5)

a = destroy(fockbasis)
at = create(fockbasis)
xket = coherent_state(fockbasis, 0.1)
xbra = dagger(coherent_state(fockbasis, 0.1))
result_ket = deepcopy(xket)
result_bra = deepcopy(xbra)

operators.gemv!(complex(1.0), at, xket, complex(0.), result_ket)
@assert norm(result_ket-at*xket, 2)<1e-14

operators.gemv!(complex(1.0), xbra, at, complex(0.), result_bra)
@assert norm(result_bra-xbra*at, 2)<1e-14