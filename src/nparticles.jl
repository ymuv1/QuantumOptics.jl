module nparticles

export NParticleBasis, BosonicNParticleBasis, FermionicNParticleBasis, nparticloperator

import Base.==

using ..bases, ..operators


function distribute_bosons(particlenumber::Int, singleparticledimension::Int, index::Int=1, occupations::Vector{Int}=zeros(Int,singleparticledimension), results::Vector{Vector{Int}}=Vector{Int}[])
    if index==singleparticledimension
        occupations[index] = particlenumber
        push!(results, copy(occupations))
    else
        for n=particlenumber:-1:0
            occupations[index] = n
            distribute_bosons(particlenumber-n, singleparticledimension, index+1, occupations, results)
        end
    end
    return results
end

function distribute_fermions(particlenumber::Int, singleparticledimension::Int, index::Int=1, occupations::Vector{Int}=zeros(Int,singleparticledimension), results::Vector{Vector{Int}}=Vector{Int}[])
    if (singleparticledimension-index)+1<particlenumber
        return results
    end
    if index==singleparticledimension
        occupations[index] = particlenumber
        push!(results, copy(occupations))
    else
        for n=min(1,particlenumber):-1:0
            occupations[index] = n
            distribute_fermions(particlenumber-n, singleparticledimension, index+1, occupations, results)
        end
    end
    return results
end


abstract NParticleBasis <: Basis

type BosonicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    particlenumber::Int
    particlebasis::Basis
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function BosonicNParticleBasis(particlenumber, particlebasis, occupations)
        particlenumber < 0 && throw(ArgumentError("Can't have less than zero particles."))
        for occupation=occupations
            length(particlebasis) != length(occupation) && throw(ArgumentError("Dimension of single particle basis has to be equal to the dimension of the N-particle basis vector."))
            any(occupation.<0) && throw(ArgumentError("Occupation numbers smaller than zero not possible."))
            sum(occupation) != particlenumber && throw(ArgumentError("Total occupation has to be equal to the particle number."))
        end
        new([length(occupations)], particlenumber, particlebasis, occupations, hash(occupations))
    end
end

type FermionicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    particlenumber::Int
    particlebasis::Basis
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function FermionicNParticleBasis(particlenumber::Int, particlebasis::Basis, occupations::Vector{Vector{Int}})
        particlenumber < 0 && throw(ArgumentError("Can't have less than zero particles."))
        for occupation=occupations
            length(particlebasis) != length(occupation) && throw(ArgumentError("Dimension of single particle basis has to be equal to the dimension of the N-particle basis vector."))
            any(occupation.<0) && throw(ArgumentError("Occupation numbers smaller than zero not possible."))
            sum(occupation) != particlenumber && throw(ArgumentError("Total occupation has to be equal to the particle number."))
            any(occupation.>1) && throw(ArgumentError("Occupation numbers greater than zero not possible for Fermions."))
        end
        new([length(occupations)], particlenumber, particlebasis, occupations, hash(occupations))
    end
end

BosonicNParticleBasis(particlenumber::Int, particlebasisdimension::Int) = BosonicNParticleBasis(particlenumber, GenericBasis([particlebasisdimension]), distribute_bosons(particlenumber, particlebasisdimension))
FermionicNParticleBasis(particlenumber::Int, particlebasisdimension::Int) = FermionicNParticleBasis(particlenumber, GenericBasis([particlebasisdimension]), distribute_fermions(particlenumber, particlebasisdimension))

BosonicNParticleBasis(particlenumber::Int, particlebasis::Basis) = BosonicNParticleBasis(particlenumber, particlebasis, distribute_bosons(particlenumber, length(particlebasis)))
FermionicNParticleBasis(particlenumber::Int, particlebasis::Basis) = FermionicNParticleBasis(particlenumber, particlebasis, distribute_fermions(particlenumber, length(particlebasis)))


=={T<:NParticleBasis}(b1::T, b2::T) = (b1.particlenumber==b2.particlenumber && b1.particlebasis==b2.particlebasis && b1.occupations_hash==b2.occupations_hash)

function nparticleoperator(nparticlebasis::NParticleBasis, op::DenseOperator; rank::Int=1)
    rank<1 && throw(ArgumentError("Rank has to be greater than zero."))
    rank<2 && throw(ArgumentError("Not yet implemented for ranks greater than one."))
    op = Operator(nparticlebasis)
    N = length(nparticlebasis)
    for m=1:N, n=1:N
        if m==n
            for i=1:length(nparticlebasis.particlebasis)
                op.data[m,m] += op.data[i,i]*basis.occupations[m][i]
            end
        end
        indices = indices_adaggeri_aj(basis.occupations[m], basis.occupations[n])
        if !isnull(indices)
            i, j = get(indices)
            Ni = basis.occupations[n][i]
            Nj = basis.occupations[n][j]+1
            op.data[m,n] += op.data[i,j]*sqrt(Ni*Nj)
        end
    end
    return op
end

"""
Find indices i and j where <{m}|a^t_i a_j |{n}> is not zero.

Returns null if this condition is never fullfilled for the given states.
"""
function indices_adaggeri_aj(occ_m::Vector{Int}, occ_n::Vector{Int})
    null = Nullable{Tuple{Int,Int}}()
    idx_create = 0
    idx_destroy = 0
    for i=1:length(occ_m)
        if occ_m[i]==occ_n[i]
            continue
        end
        delta = occ_n[i] - occ_m[i]
        if delta==-1
            if idx_destroy != 0
                return null
            end
            idx_destroy = i
            N_destroy = occ_n[i]
        elseif delta==1
            if idx_create != 0
                return null
            end
            idx_create = i
            N_create = occ_n[i]+1
        else
            return null
        end
    end
    return Nullable((idx_create, idx_destroy))
end


"""
Find indices i and j where <{m}|a^t_i a_j |{n}> is not zero.

Returns null if this condition is never fullfilled for the given states.
"""
function indices_adaggeri_adaggerj_ak_al(occ_m::Vector{Int}, occ_n::Vector{Int})
    null = Nullable{Tuple{Int,Int,Int,Int}}()
    i = 0
    j = 0
    k = 0
    l = 0
    for i=1:length(occ_m)
        if occ_m[i]==occ_n[i]
            continue
        end
        delta = occ_n[i] - occ_m[i]
        if delta==-1
            if idx_destroy != 0
                return null
            end
            idx_destroy = i
            N_destroy = occ_n[i]
        elseif delta==1
            if idx_create != 0
                return null
            end
            idx_create = i
            N_create = occ_n[i]+1
        else
            return null
        end
    end
    return Nullable((idx_create, idx_destroy))
end

end