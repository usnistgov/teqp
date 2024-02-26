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
    auto R(const VecType& /*molefrac*/) const { return teqp::constants::R_CODATA2017; }

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
    
    // Random values for a and b, JUST for demonstration purposes
    auto j = R"(
    {"kind": "myvdW", "model": {"a": 1.2345, "b": 3.4567}}
    )"_json;
    
    ModelPointerFactoryFunction func = [](const nlohmann::json& j){ return make_owned(myvdWEOS1(j.at("a"), j.at("b"))); };
    add_model_pointer_factory_function("myvdW", func);

    auto ptr = make_model(j);
    auto ref = get_model_cref<myvdWEOS1>(ptr.get());
    CHECK(ref.a == 1.2345);
}
