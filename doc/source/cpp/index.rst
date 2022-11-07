C++ interface
===============

Introduction
------------

The abstract base class defining the public C++ interface of teqp is documented in :teqp:`AbstractModel`.  This interface was developed because re-compilation of the core of ``teqp`` is VERY slow, due to the heavy use of templates, which makes the code very flexible, but difficult to work with when doing development. Especially users that would like to only use the library but not be forced to pay the price of recompilation benefit from this approach.

The models that are allowed in this abstract interface are defined in :teqp:`AllowedModels`.  A new model instance can be created by passing properly formatted JSON data structure to the :teqp:`make_model` function.