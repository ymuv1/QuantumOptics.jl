Introduction
============

**Quantumoptics.jl** is a numerical framework used to simulate various kinds of quantum systems similarly to the `Quantum Optics Toolbox for MATLAB <http://qo.phy.auckland.ac.nz/toolbox/>`_ and its python equivalent `QuTiP <http://qutip.org/>`_. It allows to easily model quantum systems by providing convenient tools to define states and (density)-operators which can be used to construct Hamiltonians and Liouvillians to study e.g. the time-evolution or to find ground states.

One of the core concepts of **Quantumoptics.jl** is that every quantum object, i.e. state vectors and operators have knowledge about which Hilbert space they live in. This prevents many common mistakes when working with composite systems and at the same time improves readability. The Hilbert spaces are defined implicitly by specifying appropriate bases like :ref:`fock <section-fock>` bases and :ref:`spin <section-spin>` bases. These bases can in turn be combined to describe composite systems like e.g. a particle in a cavity or a multi-spin system. Working with bases is explained in great detail in the :ref:`section-bases` section.

After choosing a basis **Quantumoptics.jl** provides many useful functions to create common :ref:`section-operators` and :ref:`section-states` which can be combined in all the expected ways. Consequently creating arbitrary Hamiltonians and Liouvillians and specifying initial states is straight forward. These can in turn be used to perform time evolutions according to :ref:`Schroedinger <section-schroedinger>`, :ref:`Master <section-master>` and :ref:`MCWF <section-mcwf>` equations.

Although the main focus is on simulating dynamics of (open) quantum systems, there are nevertheless many additional features available to find :ref:`steadystates <section-steadystate>`, the :ref:`energy spectrum, eigenbasis and groundstate <section-spectralanalysis>`, and :ref:`correlation functions <section-correlationfunctions>`.
