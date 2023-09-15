#include <catch2/catch_test_macros.hpp>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/types.hpp"

/// A (very) simple implementation of the van der Waals EOS,
class myvdWEOS1 {
private:
    double a, b;
public:
    myvdWEOS1(double a, double b) : a(a), b(b) {};
    
    /// Accessor functions
    double get_a() const{ return a; }
    double get_b() const{ return b; }

    const double Ru = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A

    /// \brief Get the universal gas constant
    /// \note Here the real universal gas constant, with no composition dependence
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const { return Ru; }

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
    
    auto j = R"protect(
    {"kind": "myvdW", "model": {"a":1.2, "b": 3.4}}
    )protect"_json;
    
    ModelPointerFactoryFunction func = [](const nlohmann::json& j){ return make_owned(myvdWEOS1(j.at("a"), j.at("b"))); };
    add_model_pointer_factory_function("myvdW", func);

    auto ptr = make_model(j);
}
