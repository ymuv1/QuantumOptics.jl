Quantumoptics.jl
================

**Quantumoptics.jl** is a numerical framework written in `Julia <http://julialang.org/>`_ that makes it easy to simulate various kinds of quantum systems. It is similar to the `Quantum Optics Toolbox <http://qo.phy.auckland.ac.nz/toolbox/>`_ for MATLAB and its Python equivalent `QuTiP <http://qutip.org/>`_.

Example
-------

.. code-block:: julia

    using Quantumoptics

    b = SpinBasis(1//2)
    H = sigmap(b) + sigmam(b)
    psi0 = spindown(b)

    T = [0:0.1:1;]
    tout, psit = timeevolution.schroedinger(T, psi0, H)


Documentation
-------------

The documentation written with `Sphinx <http://www.sphinx-doc.org/>`_ with the `Sphinx-Julia <https://github.com/bastikr/sphinx-julia>`_ plugin can be found at https://bastikr.github.io/Quantumoptics.jl/

