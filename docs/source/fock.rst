.. _section-fock:

Fock Basis
==========

To create a :jl:type:`FockBasis` an upper cutoff and optionally a lower cutoff have to be supplied::

    b1 = FockBasis(10)
    b2 = FockBasis(2,12)

In this example even though the dimensions of the Hilbert spaces described by these bases are the same ``b1`` and ``b2`` are not and mixing operators in one bases with operators in the other basis will result in an explicit error.

The definition of :jl:type:`FockBasis` is essentially::

    type FockBasis <: Basis
        shape::Vector{Int}
        Nmin::Int
        Nmax::Int
    end

Many common operators are already defined:

* :jl:func:`number`
* :jl:func:`destroy`
* :jl:func:`create`

Fock states and coherent states can be created using the functions:

* :jl:func:`fockstate`
* :jl:func:`coherentstate`

Additional functions:

.. epigraph::

    .. jl:autofunction:: fock.jl qfunc(rho, alpha)
