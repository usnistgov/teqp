Algorithms
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   VLE
   VLLE
   VLLE-p
   critical_curves

Information
-----------

The algorithms are written in a very generic way; they take an instance of a thermodynamic model, and the necessary derivatives are calculated from this model with automatic differentiation (or similar). In that way, implementing a model is all that is required to enable its use in the calculation of critical curves or to trace the phase equilibria.  Determining the starting values, on the other hand, may require model-specific assistance, for instance with superancillary equations.
