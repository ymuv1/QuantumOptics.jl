.. _section-bases:

Bases
=====

The primary purpose of bases in **Quantumoptics.jl** is to specify the dimension of the Hilbert space covered by them and to make sure that quantum objects which correspond to different bases can't be combined accidentally in an incorrect way. Many of the commonly used basis types like

* :ref:`section-spin`
* :ref:`section-fock`
* :ref:`section-particle`

are already built into **Quantumoptics.jl**. Hilbert spaces of combined systems can be handled automatically with :jl:type:`quantumoptics.bases.CompositeBasis` which can for example be created using the :jl:func:`quantumoptics.tensor` function::

    basis_fock = FockBasis(10)
    basis_particle = MomentumBasis(0., 10., 50)
    basis = tensor(basis_fock, basis_particle)

For cases not covered by these one can either use the :jl:type:`quantumoptics.bases.GenericBasis` or implement own special purpose bases by deriving from the abstract :jl:abstract:`quantumoptics.bases.Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of their Hilbert space. E.g. a spin 1/2 basis could be implemented as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        SpinBasis() = new(Int[2]) # Constructor
    end

The interaction with other bases can be determined by overloading the `==` operator as well as the :jl:func:`quantumoptics.bases.multiplicable` function::

    ==(b1::SpinBasis, b2::SpinBasis) = true
    multiplicable(b1::SpinBasis, b2::SpinBasis) = true


.. toctree::
    :hidden:

    spin
    fock
    particle