Time-evolution
==============

Schroedinger time evolution
---------------------------

.. math::

    i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H |\Psi(t)\rangle

    - i\hbar\frac{\mathrm{d}}{\mathrm{d} t} \langle \Psi(t)| = \langle\Psi(t)| H


.. .. function:: schroedinger(tspan, psi0, H; fout=nothing, kwargs...)
.. jl:autofunction:: timeevolution.jl schroedinger


Master time evolution
---------------------

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
                 + \sum_i \big(
                        J_i \rho J_i^\dagger
                        - \frac{1}{2} J_i^\dagger J_i \rho
                        - \frac{1}{2} \rho J_i^\dagger J_i
                    \big)

Equation for non-hermitian Hamiltonian :math:`H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i`

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H_\mathrm{nh},\rho\big]
                 + \sum_i J_i \rho J_i^\dagger


.. jl:autofunction:: timeevolution.jl master

.. .. function:: master(tspan, rho0, Hnh, J; Gamma, Jdagger, fout, tmp, kwargs...)
.. .. function:: master_h(tspan, rho0, H, J; Gamma, Jdagger, fout, tmp, kwargs...)
.. .. function:: master_nh(tspan, rho0, Hnh, J; Gamma, Jdagger, fout, tmp, kwargs...)


MCWF time evolution
-------------------

Solve the master equation

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
                 + \sum_i \big(
                        J_i \rho J_i^\dagger
                        - \frac{1}{2} J_i^\dagger J_i \rho
                        - \frac{1}{2} \rho J_i^\dagger J_i
                    \big)

using the stochastic MCWF method. Basically calculate trajectories with coherent time evolution according to

.. math::

    i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H_\mathrm{nh} |\Psi(t)\rangle

with jumps at randomly determined times

.. math::

    |\Psi(t)\rangle \rightarrow \frac{J_i |\Psi(t)\rangle}{||J_i |\Psi(t)\rangle||}

The stochastic average of these trajectories is then equal to the solution of the master equation :math:`\rho(t)`

.. math::

    \lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N |\Psi^k(t)\rangle\langle\Psi^k(t)| = \rho(t)

and also the stochastic average of the single trajectory expectation values is equal to the expectation value according to the master equation

.. math::

    \lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N \langle\Psi^k(t)| A |\Psi^k(t)\rangle = \mathrm{Tr}\big\{A \rho(t)\big\}

avoiding explicit calculations of density matrices.

.. jl:autofunction:: timeevolution.jl mcwf

.. .. function:: mcwf(tspan, rho0, Hnh, J; seed, fout, Jdagger, display_beforeevent, display_afterevent, kwargs...)
.. .. function:: mcwf_h(tspan, rho0, H, J; seed, fout, Jdagger, display_beforeevent, display_afterevent, kwargs...)
.. .. function:: mcwf_nh(tspan, rho0, Hnh, J; seed, fout, Jdagger, display_beforeevent, display_afterevent, kwargs...)
