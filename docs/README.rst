Building the documentation
==========================

First create a folder ``QuantumOptics.jl-docs`` in the same directory as the QuantumOptics.jl repository. Inside this directory clone the QuantumOptics.jl repository and change to the gh-pages branch::

    >> git clone git@github.com:bastikr/QuantumOptics.jl.git html
    >> cd html
    >> git checkout

This pulls the documentation as it is available on https://bastikr.github.io/QuantumOptics.jl/.
To build the latest documentation change the current directory to ``QuantumOptics.jl/docs`` and use make::

    >> make html