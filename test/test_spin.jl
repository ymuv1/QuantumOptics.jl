using Base.Test
using quantumoptics

I = identity(spinbasis)
Zero = Operator(spinbasis)
sx = quantumoptics.sigmax
sy = quantumoptics.sigmay
sz = quantumoptics.sigmaz
sp = quantumoptics.sigmap
sm = quantumoptics.sigmam


# Test traces
@test_approx_eq 0. trace(sx)
@test_approx_eq 0. trace(sy)
@test_approx_eq 0. trace(sz)


# Test if involutory
@test_approx_eq 0. tracedistance(sx*sx, I)
@test_approx_eq 0. tracedistance(sy*sy, I)
@test_approx_eq 0. tracedistance(sz*sz, I)
@test_approx_eq 0. tracedistance(-1im*sx*sy*sz, I)


# Test kommutation relations
kommutator(x, y) = x*y - y*x

@test_approx_eq 0. tracedistance(kommutator(sx, sx), Zero)
@test_approx_eq 0. tracedistance(kommutator(sx, sy), 2im*sz)
@test_approx_eq 0. tracedistance(kommutator(sx, sz), -2im*sy)
@test_approx_eq 0. tracedistance(kommutator(sy, sx), -2im*sz)
@test_approx_eq 0. tracedistance(kommutator(sy, sy), Zero)
@test_approx_eq 0. tracedistance(kommutator(sy, sz), 2im*sx)
@test_approx_eq 0. tracedistance(kommutator(sz, sx), 2im*sy)
@test_approx_eq 0. tracedistance(kommutator(sz, sy), -2im*sx)
@test_approx_eq 0. tracedistance(kommutator(sz, sz), Zero)


# Test antikommutator
antikommutator(x, y) = x*y + y*x

@test_approx_eq 0. tracedistance(antikommutator(sx, sx), 2*I)
@test_approx_eq 0. tracedistance(antikommutator(sx, sy), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sx, sz), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sy, sx), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sy, sy), 2*I)
@test_approx_eq 0. tracedistance(antikommutator(sy, sz), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sz, sx), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sz, sy), Zero)
@test_approx_eq 0. tracedistance(antikommutator(sz, sz), 2*I)


# Test creation and anihilation operators
@test_approx_eq 0. tracedistance(sp, 0.5*(sx + 1im*sy))
@test_approx_eq 0. tracedistance(sm, 0.5*(sx - 1im*sy))
@test_approx_eq 0. tracedistance(sx, (sp + sm))
@test_approx_eq 0. tracedistance(sy, -1im*(sp - sm))


# Test commutation relations with creation and anihilation operators
@test_approx_eq 0. tracedistance(kommutator(sp, sm), sz)
@test_approx_eq 0. tracedistance(kommutator(sz, sp), 2*sp)
@test_approx_eq 0. tracedistance(kommutator(sz, sm), -2*sm)


# Test v x (v x u) relation: [sa, [sa, sb]] = 4*(1-delta_{ab})*sb
@test_approx_eq 0. tracedistance(kommutator(sx, kommutator(sx, sx)), Zero)
@test_approx_eq 0. tracedistance(kommutator(sx, kommutator(sx, sy)), 4*sy)
@test_approx_eq 0. tracedistance(kommutator(sx, kommutator(sx, sz)), 4*sz)
@test_approx_eq 0. tracedistance(kommutator(sy, kommutator(sy, sx)), 4*sx)
@test_approx_eq 0. tracedistance(kommutator(sy, kommutator(sy, sy)), Zero)
@test_approx_eq 0. tracedistance(kommutator(sy, kommutator(sy, sz)), 4*sz)
@test_approx_eq 0. tracedistance(kommutator(sz, kommutator(sz, sx)), 4*sx)
@test_approx_eq 0. tracedistance(kommutator(sz, kommutator(sz, sy)), 4*sy)
@test_approx_eq 0. tracedistance(kommutator(sz, kommutator(sz, sz)), Zero)
