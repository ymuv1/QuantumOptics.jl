Quantum objects
===============

One of the core concepts of julia-quantumoptics is that every quantum object, i.e. state vectors and operators have knowledge about which Hilbert space they live in via the various basis types. This on the one hand increases readability and on the other hand prevents at least some of careless errors that can happen during algebraic combination of quantum objects. The general procedure to study a certain quantum system usually follows roughly the steps:

#. Characterize Hilbert space of quantum system by creating appropriate bases.
#. Create all necessary basic operators (Choose between sparse and dense).
#. Build up Hamiltonians and Jump operators.
#. Choose initial states.
#. Perform time evolution (Schroedinger, MCWF, Master).
#. Calculate interesting expectation values.


Bases
-----

The main point bases are used for is to specify the dimension of the Hilbert space covered by them and to make sure that quantum objects which corresponding to different bases can't be combined accidentally in an incorrect way. Many of the commonly used basis types like

- :jl:class:`quantumoptics.fock.FockBasis`
- :jl:class:`quantumoptics.spins.SpinBasis`
- :jl:class:`quantumoptics.particle.PositionBasis`
- :jl:class:`quantumoptics.particle.MomentumBasis`

are already built into julia-quantumoptics. For cases not covered by these built-in bases one can either use the

- :jl:class:`quantumoptics.bases.GenericBasis`

or implement own special purpose bases by deriving from the abstract :jl:class:`quantumoptics.bases.Basis` class. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of the Hilbert space. The interaction with other bases can be determined by overloading the `==` operator as well as the :jl:func:`quantumoptics.bases.multiplicable` function.

Hilbert spaces of combined systems can be handled automatically with

- :jl:class:`quantumoptics.bases.CompositeBasis`

which can for example be created using the :jl:func:`quantumoptics.tensor` function.


State vectors
-------------

State vectors in julia-quantumoptics are always stored as coefficients in respect to a certain basis.


Operators
---------
