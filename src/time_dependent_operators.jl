# Convert storage of heterogeneous stuff to tuples for maximal compilation
# and to avoid runtime dispatch.
function _tuplify(o::TimeDependentSum)
    if isconcretetype(eltype(o.coefficients)) && isconcretetype(eltype(o.static_op.operators))
        # No need to tuplify is types are concrete.
        # We will save on compile time this way.
        return o
    end
    return TimeDependentSum(Tuple, o)
end
function _tuplify(o::LazySum)
    if isconcretetype(eltype(o.factors)) && isconcretetype(eltype(o.operators))
        return o
    end
    return LazySum(eltype(o.factors), o.factors, (o.operators...,))
end
_tuplify(o::AbstractVector{T}) where T = isconcretetype(T) ? o : (o...,)
_tuplify(o::Tuple) = o
_tuplify(o::AbstractOperator) = o


"""
    function optimized_TDS(H::TimeDependentSum)
Returns a version of the TimeDependentSum (TDS) that is mathematically equivalent, but may be more efficient: if there are several constant terms (this
is often the case when non-Hermitian Hamiltonians are formed), they are summed together so that they don't have to be calculated separately and summed
every time we call the TDS (which is done at each timestep of a simulation).
"""
function optimized_TDS(H::TimeDependentSum)
    indices_of_consts = findall(x -> isa(x, Number), H.coefficients)
    const_op = sum(H.coefficients[indices_of_consts] .* H.static_op.operators[indices_of_consts])

    non_const_indices = setdiff(1:length(H.coefficients), indices_of_consts)
    non_const_coefs = H.coefficients[non_const_indices]
    non_const_ops = H.static_op.operators[non_const_indices]
    
    return TimeDependentSum(vcat([one(eltype(const_op))], non_const_coefs), vcat(const_op, non_const_ops))
end


"""
    schroedinger_dynamic_function(H::AbstractTimeDependentOperator)

Creates a function of the form `f(t, state) -> H(t)`. The `state` argument is ignored.

This is the function expected by [`timeevolution.schroedinger_dynamic()`](@ref).
"""
function schroedinger_dynamic_function(H::AbstractTimeDependentOperator)
    H = optimized_TDS(H)
    _getfunc(op) = (@inline _tdop_schroedinger_wrapper(t, _) = set_time!(op, t))
    Htup = _tuplify(H)
    return _getfunc(Htup)
end

_tdopdagger(o) = dagger(o)
function _tdopdagger(o::TimeDependentSum)
    # This is a kind-of-hacky, more efficient TimeDependentSum dagger operation
    # that requires that the original operator sticks around and is always
    # updated first (though this is checked).
    # Copies and conjugates the coefficients from the original op.
    # TODO: Make an Adjoint wrapper for TimeDependentSum instead?
    o_ls = QuantumOpticsBase.static_operator(o)
    facs = o_ls.factors
    c1 = (t)->(@assert current_time(o) == t; conj(facs[1]))
    crest = (((_)->conj(facs[i])) for i in 2:length(facs))
    odag = TimeDependentSum((c1, crest...), dagger(o_ls), current_time(o))
    return odag
end

"""
    master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)

Returns a function of the form `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is ignored.

This is the function expected by [`timeevolution.master_h_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
function master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)
    H = optimized_TDS(H)
    Htup = _tuplify(H)
    Js_tup = _tuplify(map(_tuplify, Js))
    Jdags_tup = map(_tdopdagger, Js_tup)

    return let Hop = Htup, Jops = Js_tup, Jdops = Jdags_tup
        function _tdop_master_wrapper_1(t, _)
            f = Base.Fix2(set_time!, t)
            foreach(f, Jops)
            foreach(f, Jdops)
            set_time!(Hop, t)
            return Hop, Jops, Jdops
        end
    end
end

"""
    master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)

Returns a function of the form `f(t, state) -> Hnh(t), Hnh(t)', Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.master_nh_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the
(time independent) jump operators.
"""
function master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)
    Hnh = optimized_TDS(Hnh)
    Hnhtup = _tuplify(Hnh)
    Js_tup = _tuplify(map(_tuplify, Js))

    Jdags_tup = map(_tdopdagger, Js_tup)
    Htdagup = _tdopdagger(Hnhtup)

    return let Hop = Hnhtup, Hdop = Htdagup, Jops = Js_tup, Jdops = Jdags_tup
        function _tdop_master_wrapper_2(t, _)
            f = Base.Fix2(set_time!, t)
            foreach(f, Jops)
            foreach(f, Jdops)
            set_time!(Hop, t)
            set_time!(Hdop, t)
            return Hop, Hdop, Jops, Jdops
        end
    end
end

"""
Returns a function of the form (t, ρ) -> (Cs, Cs').
Untested.
"""
function master_stochastic_dynamics_function(Cs)
    Cs_tup = _tuplify(map(_tuplify, Cs))
    Cdags_tup = map(_tdopdagger, Cs_tup)

    all_constant = !any(isa.(Cs, AbstractTimeDependentOperator))
    if all_constant
        return (_, _) -> (Cs_tup, Cdags_tup) # more efficient version for constants
    end
    return let Cops = Cs_tup, Cdops = Cdags_tup
        function _tdop_master_wrapper_2(t, _)
            f = Base.Fix2(set_time!, t)
            foreach(f, Cops)
            foreach(f, Cdops)
            return Cops, Cdops
        end
    end
end

"""
    mcfw_dynamic_function(H, Js)

Returns a function of the form `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
mcfw_dynamic_function(H, Js) = master_h_dynamic_function(H, Js)

"""
    mcfw_nh_dynamic_function(Hnh, Js)

Returns a function of the form `f(t, state) -> Hnh(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the (time
independent) jump operators.
"""
mcfw_nh_dynamic_function(Hnh, Js) = master_h_dynamic_function(Hnh, Js)
