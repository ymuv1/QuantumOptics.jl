using Base.Test
using quantumoptics

ωc = 1.2
ωa = 0.9
g = 1.0
Ncutoff = 10

T = Float64[0.,1.]


fockbasis = FockBasis(Ncutoff)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = SparseOperator(sigmax(spinbasis))
sy = SparseOperator(sigmay(spinbasis))
sz = SparseOperator(sigmaz(spinbasis))
sp = SparseOperator(sigmap(spinbasis))
sm = SparseOperator(sigmam(spinbasis))

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = g/2*(sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis))

H = Ha + Hc + Hint
Hsparse = SparseOperator(H)

Omega_n(n::Int) = sqrt((ωa-ωc)^2 + g^2*(n+1))
Alpha_n(n::Int) = atan2(g*sqrt(n+1),(ωa-ωc))

Em(n::Int) = ωc*(n+0.5) - 0.5*Omega_n(n)
Ep(n::Int) = ωc*(n+0.5) + 0.5*Omega_n(n)

psi_e(n::Int) = spinup(spinbasis) ⊗ fockstate(fockbasis, n)
psi_g(n::Int) = spindown(spinbasis) ⊗ fockstate(fockbasis, n+1)

psi_p(n::Int) = cos(Alpha_n(n)/2)*psi_e(n) + sin(Alpha_n(n)/2)*psi_g(n)
psi_m(n::Int) = -sin(Alpha_n(n)/2)*psi_e(n) + cos(Alpha_n(n)/2)*psi_g(n)
basisstates_p = Ket[psi_p(n) for n=0:Ncutoff-1]
basisstates_m = Ket[psi_m(n) for n=0:Ncutoff-1]

Emvec = Float64[Em(n) for n=0:Ncutoff-1]
Epvec = Float64[Ep(n) for n=-1:Ncutoff-1]

P = sortperm([Emvec; Epvec])
basisstates = Ket[basisstates_m; psi_g(-1); basisstates_p][P][1:(Ncutoff+1)]
E = [Emvec; Epvec][P][1:(Ncutoff+1)]

@test norm(operatorspectrum(H)[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum_hermitian(H)[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum(Hsparse)[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum_hermitian(Hsparse)[1:(Ncutoff+1)] - E) < 1e-12

b = eigenbasis(H)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenbasis_hermitian(H)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenbasis(Hsparse)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenbasis_hermitian(Hsparse)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end
