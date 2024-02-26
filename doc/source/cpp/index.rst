C++ interface
===============

Introduction
------------

The abstract base class defining the public C++ interface of teqp is documented in :teqp:`AbstractModel`.  This interface was developed because re-compilation of the core of ``teqp`` is VERY slow, due to the heavy use of templates, which makes the code very flexible, but difficult to work with when doing development. Especially users that would like to only *use* the library but not be forced to pay the price of recompilation benefit from this approach.

As a user, a new model instance (a std::unique_ptr<teqp::AbstactModel*>) can be created by passing properly formatted JSON data structure to the :teqp:`make_model` function. 

Object Model
------------

The object model in teqp is convoluted because of the requirements to have models that use templated types to allow the use of automatic differentiation types. Instances of classes with templated methods cannot be stored directly in generic STL containers like ``std::vector`` or ``std::list`` (though they can be stored in ``std::tuple``, but ``tuple`` cannot be constructed at runtime because they have complete type knowledge and C++ is strongly typed). Thus, some sort of wrapping is required (in C++ the technical term is type erasure) to store objects of a homogenous interface in dynamic containers like ``std::vector``.  

A number of type-erasure classes are defined, especially the :teqp:`DerivativeAdapter` class which does type erasure on a model that it holds. This :teqp:`DerivativeAdapter` class has an interface that takes STL types (and Eigen arrays in some cases) as input arguments, and then calls lower-level methods that can operate with a range of different numerical types, and call the templated methods of a model.

As a developer/implementer of a thermodynamic model, the class implementing the thermodynamic model for a contribution to :math:`\alpha` must satisfy the following requirements:

* It must have a method called ``alphar`` that takes three arguments that are all generic types. The first argument is the temperature, the second argument is the molar density, and the third is the mole fractions. In the case of some equations of state for model potentials, the temperature and density are treated as being in reduced units. The function should be called ``alphar`` even for Helmholtz energy contributions that are for ideal gases. You can think of the ``r`` in ``alphar`` standing for ``reduced`` instead of ``residual`` if that helps.
* It must have a method called ``R`` that takes a single argument that is the mole fractions of the components. It then returns the molar gas constant of the mixture. For most models it suffices to return 8.31446261815324, which is `the CODATA value of the molar gas constant <https://en.wikipedia.org/wiki/Gas_constant>`_ , and is available in the :teqp:`teqp::constants` namespace.  The reason the ``R`` method must be implemented is the multiparameter models in which the molar gas constant of diffent components is slightly different based upon when the EOS was published. Also, some of the other models used different values of R (or Avogadro's constant) when being developed and if you want to get perfect reproducibility these details matter.

This model instance is then passed to one of two methods in the :teqp:`teqp::cppinterface::adapter` namespace: :teqp:`teqp::cppinterface::adapter::make_owned` or :teqp:`teqp::cppinterface::adapter::make_cview`. As the name suggests, if you pass the class instance to the ``make_owned`` function, it takes ownership of the model and the argument passed to the function is invalidated. On the contrary, the ``make_cview`` method is just a "viewer" of the model without taking ownership, so you need to watch out that the lifetime of the model you pass to this function is longer than the time you are using the wrapper model.

For instance this minimal working model of the van der Waals EOS demonstrates some of the things to be aware of:

.. code-block:: C++

    /// A (very) simple implementation of the van der Waals EOS
    class myvdWEOS1 {
    public:
        const double a, b;
        myvdWEOS1(double a, double b) : a(a), b(b) {};

        /// \brief Get the universal gas constant
        template<class VecType>
        auto R(const VecType& /*molefrac*/) const { return constants::R_CODATA2017; }

        /// The evaluation of \f$ \alpha^{\rm r}=a/(RT) \f$
        /// \param T The temperature
        /// \param rhotot The molar density
        /// \param molefrac The mole fractions of each component
        template<typename TType, typename RhoType, typename VecType>
        auto alphar(const TType &T, const RhoType& rhotot, const VecType &molefrac) const {
            return teqp::forceeval(-log(1.0 - b * rhotot) - (a / (R(molefrac) * T)) * rhotot);
        }
    };
    
The name of the class is entirely arbitrary, you could call it just as well ``GreatVdWModel`` instead of ``myvdWEOS1``.

A complete example could then read:

.. code-block:: C++

    #include <catch2/catch_test_macros.hpp>

    #include "teqp/cpp/teqpcpp.hpp"
    #include "teqp/cpp/deriv_adapter.hpp"
    #include "teqp/types.hpp"
    #include "teqp/constants.hpp"

    /// A (very) simple implementation of the van der Waals EOS
    class myvdWEOS1 {
    public:
        const double a, b;
        myvdWEOS1(double a, double b) : a(a), b(b) {};

        /// \brief Get the universal gas constant
        template<class VecType>
        auto R(const VecType& /*molefrac*/) const { return constants::R_CODATA2017; }

        /// The evaluation of \f$ \alpha^{\rm r}=a/(RT) \f$
        /// \param T The temperature
        /// \param rhotot The molar density
        /// \param molefrac The mole fractions of each component
        template<typename TType, typename RhoType, typename VecType>
        auto alphar(const TType &T, const RhoType& rhotot, const VecType &molefrac) const {
            return teqp::forceeval(-log(1.0 - b * rhotot) - (a / (R(molefrac) * T)) * rhotot);
        }
    };

    TEST_CASE("Check adding a model at runtime"){
        using namespace teqp::cppinterface;
        using namespace teqp::cppinterface::adapter;

        auto j = R"(
        {"kind": "myvdW", "model": {"a": 1.2, "b": 3.4}}
        )"_json;

        ModelPointerFactoryFunction func = [](const nlohmann::json& j){ return make_owned(myvdWEOS1(j.at("a"), j.at("b"))); };
        add_model_pointer_factory_function("myvdW", func);

        auto ptr = make_model(j);
    }

    
In this runnable example (runnable once the include paths are correct and the code is linked against the ``teqpcpp`` C++ library), a new factory function is registered with the :teqp:`add_model_pointer_factory_function` function and then this function is used to generate a ``std::unique_ptr<AbstractModel*>``. Once the model has been created, it is possible to cast it back to the original type, but you must know the type of the class that you are holding (at compile time). The :teqp:`teqp::cppinterface::adapter::get_model_cref` is a convenience function to do this casting.

C++ Details
-----------

Don't return expressions
^^^^^^^^^^^^^^^^^^^^^^^^

The most important thing to be sure of when developing models in teqp is that you do not return expressions from functions. For instance in the simple function:

.. code-block:: C++

    template<typename T1, typename T2>
    auto alphar(const T1 &v1, const T2& v2) {
        return v1 + v2;
    }
    
if the types of ``T1`` and ``T2`` are both ``autodiff::real`` (the same problem occurs for other autodiff types), the value of ``v1 + v2`` is an expression type that is lazily evaluated, and the expression holds references to the actual values of the variables ``v1`` and ``v2``. This lazy evaluation is how autodiff can be so fast. Once the expression is returned from this function, the variables that it was pointing to are no longer valid because they have fallen out of scope and you can silently be pointing to invalid memory locations.

In order to avoid this problem you can use the function ``teqp::forceeval`` to force the evaluation of the expression, copying all the variables into the expression, and removing the possibility of dangling references after the function returns.

One way to ensure that you are not running into this problem is to enable the Address Sanitizer option "Detect Use of stack after return" in XCode (its in the Diagnostic panel of the "Edit Scheme..." option). Other address sanitizer tools have similar functionality.

Generic return types
^^^^^^^^^^^^^^^^^^^^

Taking the example shown above, in the function ``alphar`` all the arguments have templated type. Sometimes you will need to make use of one or more of the types in intermediate calculations within the function, and you might need to determine the type of an expression to for instance allocate a vector of this type. As an example, let's say that we are going to multiply three different variables together. In the ``alphar`` context, let's assume that ``T`` is of type double, ``rhomolar`` is of type ``std::complex<double>`` and ``molefracs`` is of type ``Eigen::ArrayXcd``. In the case of the expression ``T*rhomolar*molefracs[0]``, the result will be calculated based on the type promotion to a ``std::complex<double>``, so the result type of this product is ``std::complex<double>``. If you want to let the compiler determine this type for you, you can do:

.. code-block:: C++

    using resulttype = std::common_type_t<double, std::complex<double>, decltype(molefracs[0])>;
    
and if you need want to work with the types of the variables, usually because you need to cover all your bases for all the templat permutations, you can do instead

.. code-block:: C++

    using resulttype = std::common_type_t<decltype(T), decltype(rhomolar), decltype(molefracs[0])>;
    std::vector<resulttype> buffer;
    
and if you need to remove the ``const`` of your variable types, you can do with ``std::decay_t< >``.