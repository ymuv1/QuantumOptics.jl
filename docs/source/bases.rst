Bases
=====

The primary purpose of bases is to specify the dimension of the Hilbert space covered by them and to make sure that quantum objects which correspond to different bases can't be combined accidentally in an incorrect way. Many of the commonly used basis types like

- :jl:type:`FockBasis`
- :jl:type:`SpinBasis`
- :jl:type:`PositionBasis`
- :jl:type:`MomentumBasis`

are already built into julia-quantumoptics. For cases not covered by these built-in bases one can either use the

- :jl:type:`quantumoptics.bases.GenericBasis`

or implement own special purpose bases by deriving from the abstract :jl:abstract:`quantumoptics.bases.Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of the Hilbert space. The interaction with other bases can be determined by overloading the `==` operator as well as the :jl:func:`quantumoptics.bases.multiplicable` function.

Hilbert spaces of combined systems can be handled automatically with

- :jl:type:`quantumoptics.bases.CompositeBasis`

which can for example be created using the :jl:func:`quantumoptics.tensor` function.


Spin basis
^^^^^^^^^^

**Basis definition**

.. epigraph::

    .. jl:autotype:: spin.jl SpinBasis


**Default operators**

.. epigraph::

    .. jl:autofunction:: spin.jl sigmax

    .. jl:autofunction:: spin.jl sigmay

    .. jl:autofunction:: spin.jl sigmaz

    .. jl:autofunction:: spin.jl sigmap

    .. jl:autofunction:: spin.jl sigmam


**Default states**


.. epigraph::

    .. jl:autofunction:: spin.jl spinup

    .. jl:autofunction:: spin.jl spindown


Fock basis
^^^^^^^^^^

**Basis definition**

.. epigraph::

    .. jl:autotype:: fock.jl FockBasis



**Default operators**

.. epigraph::

    .. jl:autofunction:: fock.jl number

    .. jl:autofunction:: fock.jl destroy

    .. jl:autofunction:: fock.jl create


**Default states**

.. epigraph::

    .. jl:autofunction:: fock.jl fockstate

    .. jl:autofunction:: fock.jl coherentstate


Particle basis
^^^^^^^^^^^^^^

**Bases definitions**

.. epigraph::

    .. jl:autotype:: particle.jl PositionBasis

    .. jl:autotype:: particle.jl MomentumBasis


**Helper functions**

.. epigraph::

    .. jl:autofunction:: particle.jl PositionBasis

    .. jl:autofunction:: particle.jl MomentumBasis

    .. jl:autofunction:: particle.jl spacing

    .. jl:autofunction:: particle.jl samplepoints


**Default operators**

.. epigraph::

    .. jl:autofunction:: particle.jl positionoperator

    .. jl:autofunction:: particle.jl momentumoperator

    .. jl:autofunction:: particle.jl laplace_x

    .. jl:autofunction:: particle.jl laplace_p

    .. jl:autofunction:: particle.jl FFTOperator


**Default states**

.. epigraph::

    .. jl:autofunction:: particle.jl gaussianstate