Introduction
============

One of the core concepts of julia-quantumoptics is that every quantum object, i.e. state vectors and operators have knowledge about which Hilbert space they live in via the various basis types. This on the one hand increases readability and on the other hand prevents at least some of careless errors that can happen during algebraic combination of quantum objects. The general procedure to study a certain quantum system usually follows roughly the steps:

#. Characterize Hilbert space of quantum system by creating an appropriate bases.
#. Create all necessary basic operators (Choose between sparse and dense).
#. Build up Hamiltonians and Jump operators.
#. Choose initial states.
#. Perform time evolution (Schroedinger, MCWF, Master).
#. Calculate interesting expectation values.
