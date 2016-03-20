Subspace Basis
==============

The subspace of a given Hilbert space, defined by in this context called superbasis, can be described by a :jl:type:`SubspaceBasis`. It is defined by N, not necessarily orthogonal states. Its dimension has to be equal to or smaller than the dimension of the embedding Hilbert space. When the dimensions are equal, the subspace basis can be viewed as basis change in regard to the original basis of the Hilbert space.

.. math::

    | x^\prime \rangle
            = \sum_{u \in subspace}
                    |u_o \rangle \langle u^o
                        | x \rangle

Any SubspaceBasis can be made to an orthogonal basis with help of the :jl:func:`orthonormalize` function which utilizes the numerical stable modified Gram-Schmidt algorithm.

Changing between the superbasis and any subspaces bases can be done by first creating a projection operator with the :jl:func:`projector` function::

    b = FockBasis(5)
    b_sub = SubspaceBasis(b, [fockstate(b, 1), fockstate(b, 2)])

    P = projector(b_sub, b)

    x = coherentstate(b, 0.5)
    x_sub1 = P*x

The projection operation is irreversible if the original state was not already contained in the subspace. However, it is of course possible to represent any state in the subspace in the superbasis::

    y = dagger(P)*x_sub1 # Not equal to x


Interface:

.. jl:autotype:: subspace.jl SubspaceBasis

.. jl:autofunction:: subspace.jl orthonormalize

.. jl:autofunction:: subspace.jl projector