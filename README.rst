QuantumOptics.jl
================

**QuantumOptics.jl** is a numerical framework written in `Julia <http://julialang.org/>`_ that makes it easy to simulate various kinds of quantum systems. It is similar to the `Quantum Optics Toolbox <http://qo.phy.auckland.ac.nz/toolbox/>`_ for MATLAB and its Python equivalent `QuTiP <http://qutip.org/>`_.


Example
-------

.. code-block:: julia

    using QuantumOptics

    b = SpinBasis(1//2)
    H = sigmap(b) + sigmam(b)
    psi0 = spindown(b)

    T = [0:0.1:1;]
    tout, psit = timeevolution.schroedinger(T, psi0, H)

More involved examples created using jupyter notebooks can be found at

    https://bastikr.github.io/QuantumOptics.jl/examples.html


Documentation
-------------

The documentation written with `Sphinx <http://www.sphinx-doc.org/>`_ using the `Sphinx-Julia <https://github.com/bastikr/sphinx-julia>`_ plugin is available at

    https://bastikr.github.io/QuantumOptics.jl/
