using Base.Test
using Quantumoptics

for spinnumber=1//2:1//2:5//2
    spinbasis = SpinBasis(spinnumber)
    I = operators.identity(spinbasis)
    Zero = SparseOperator(spinbasis)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)


    # Test traces
    @test_approx_eq 0. trace(sx)
    @test_approx_eq 0. trace(sy)
    @test_approx_eq 0. trace(sz)


    # Test kommutation relations
    kommutator(x, y) = x*y - y*x

    @test_approx_eq_eps 0. tracedistance(kommutator(sx, sx), Zero) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sx, sy), 2im*sz) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sx, sz), -2im*sy) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, sx), -2im*sz) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, sy), Zero) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, sz), 2im*sx) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, sx), 2im*sy) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, sy), -2im*sx) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, sz), Zero) 1e-12


    # Test creation and anihilation operators
    @test_approx_eq 0. tracedistance(sp, 0.5*(sx + 1im*sy))
    @test_approx_eq 0. tracedistance(sm, 0.5*(sx - 1im*sy))
    @test_approx_eq 0. tracedistance(sx, (sp + sm))
    @test_approx_eq 0. tracedistance(sy, -1im*(sp - sm))


    # Test commutation relations with creation and anihilation operators
    @test_approx_eq_eps 0. tracedistance(kommutator(sp, sm), sz) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, sp), 2*sp) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, sm), -2*sm) 1e-12


    # Test v x (v x u) relation: [sa, [sa, sb]] = 4*(1-delta_{ab})*sb
    @test_approx_eq_eps 0. tracedistance(kommutator(sx, kommutator(sx, sx)), Zero) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sx, kommutator(sx, sy)), 4*sy) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sx, kommutator(sx, sz)), 4*sz) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, kommutator(sy, sx)), 4*sx) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, kommutator(sy, sy)), Zero) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sy, kommutator(sy, sz)), 4*sz) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, kommutator(sz, sx)), 4*sx) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, kommutator(sz, sy)), 4*sy) 1e-12
    @test_approx_eq_eps 0. tracedistance(kommutator(sz, kommutator(sz, sz)), Zero) 1e-12


    # Test spinup and spindown states
    @test_approx_eq_eps 1. norm(spinup(spinbasis)) 1e-11
    @test_approx_eq_eps 1. norm(spindown(spinbasis)) 1e-11
    @test_approx_eq_eps 0. norm(sp*spinup(spinbasis)) 1e-11
    @test_approx_eq_eps 0. norm(sm*spindown(spinbasis)) 1e-11
end


# Test special relations for spin 1/2

spinbasis = SpinBasis(1//2)
I = identity(spinbasis)
Zero = SparseOperator(spinbasis)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)


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


# Test if involutory for spin 1/2
@test_approx_eq 0. tracedistance(sx*sx, I)
@test_approx_eq 0. tracedistance(sy*sy, I)
@test_approx_eq 0. tracedistance(sz*sz, I)
@test_approx_eq 0. tracedistance(-1im*sx*sy*sz, I)


# Test consistency of spin up and down with sigmap and sigmam
@test_approx_eq_eps 0. norm(sm*spinup(spinbasis) - spindown(spinbasis)) 1e-11
@test_approx_eq_eps 0. norm(sp*spindown(spinbasis) - spinup(spinbasis)) 1e-11
