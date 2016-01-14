Bases
=====

The primary purpose of bases is to specify the dimension of the Hilbert space covered by them and to make sure that quantum objects which correspond to different bases can't be combined accidentally in an incorrect way. Many of the commonly used basis types like

- :jl:type:`quantumoptics.fock.FockBasis`
- :jl:type:`quantumoptics.spins.SpinBasis`
- :jl:type:`quantumoptics.particle.PositionBasis`
- :jl:type:`quantumoptics.particle.MomentumBasis`

are already built into julia-quantumoptics. For cases not covered by these built-in bases one can either use the

- :jl:type:`quantumoptics.bases.GenericBasis`

or implement own special purpose bases by deriving from the abstract :jl:abstract:`quantumoptics.bases.Basis` type. The only mandatory property of all basis types is that they have a field :obj:`shape` which specifies the dimensionality of the Hilbert space. The interaction with other bases can be determined by overloading the :obj:`==` operator as well as the :jl:func:`quantumoptics.bases.multiplicable` function.

Hilbert spaces of combined systems can be handled automatically with

- :jl:type:`quantumoptics.bases.CompositeBasis`

which can for example be created using the :jl:func:`quantumoptics.tensor` function.


Spin basis
----------


Fock basis
----------

.. jl:autotype:: fock.jl FockBasis

.. jl:autofunction:: fock.jl number

.. jl:autofunction:: fock.jl destroy

.. jl:autofunction:: fock.jl create


Particle basis
--------------
