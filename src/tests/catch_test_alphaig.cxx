#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "nlohmann/json.hpp"
#include "teqp/ideal_eosterms.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;

nlohmann::json demo_pure_term(double a_1, double a_2){
    nlohmann::json j0 = nlohmann::json::array();
    j0.push_back({ {"type", "Lead"}, { "a_1", 1 }, { "a_2", 2 } });
    return {{"R", 8.31446261815324}, {"terms", j0}};
}

TEST_CASE("Simplest case","[alphaig]") {
    double a_1 = 1, a_2 = 2, T = 300;
    nlohmann::json j = nlohmann::json::array();
    j.push_back(demo_pure_term(a_1, a_2));
    IdealHelmholtz ih(j);
    std::valarray<double> molefrac{1.0};
    REQUIRE(ih.alphaig(T, 1, molefrac) == log(1) + a_1 + a_2 / T);
}

TEST_CASE("alphaig derivative", "[alphaig]") {
    double a_1 = 1, a_2 = 2, T = 300, rho = 1;
    nlohmann::json j = nlohmann::json::array();
    j.push_back(demo_pure_term(a_1, a_2));
    IdealHelmholtz ih(j);
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    auto wih = AlphaCallWrapper<AlphaWrapperOption::idealgas, decltype(ih)>(ih);
    wih.alpha(T, rho, molefrac);
    using tdx = TDXDerivatives<decltype(ih), double, Eigen::ArrayXd>;
    SECTION("All cross derivatives should be zero") {
        REQUIRE(tdx::get_Agenxy<1, 1, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
        REQUIRE(tdx::get_Agenxy<1, 2, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
        REQUIRE(tdx::get_Agenxy<1, 3, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
        REQUIRE(tdx::get_Agenxy<2, 1, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
        REQUIRE(tdx::get_Agenxy<2, 2, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
        REQUIRE(tdx::get_Agenxy<2, 3, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
    }
}

TEST_CASE("Ammonia derivative", "[alphaig][NH3]") {
    double T = 300, rho = 10;
    double c0 = 4;
    double a1 = -6.59406093943886, a2 = 5.60101151987913;
    double Tcrit = 405.56, rhocrit = 13696.0;
    std::valarray<double> n = { 2.224, 3.148, 0.9579 }, theta = { 1646, 3965, 7231 };

    using o = nlohmann::json::object_t;
    nlohmann::json j0terms = {
          o{ {"type", "Lead"}, { "a_1", a1 - log(rhocrit)  }, { "a_2", a2 * Tcrit } },
          o{ {"type", "LogT"}, { "a", -(c0 - 1) } },
          o{ {"type", "Constant"}, { "a", (c0 - 1) * log(Tcrit) } }, // Term from ln(tau)
          o{ {"type", "PlanckEinstein"}, { "n",  n}, {"theta", theta}}
    };
    nlohmann::json j = {{ {"R", 8.31446261815324}, {"terms", j0terms} }};
    IdealHelmholtz ih(j);
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    auto wih = AlphaCallWrapper<AlphaWrapperOption::idealgas, decltype(ih)>(ih);
    auto calc = wih.alpha(T, rho, molefrac);
    auto expected = -5.3492909452728545;
    REQUIRE(calc == Approx(expected));
    
    DerivativeHolderSquare<2, AlphaWrapperOption::idealgas> dhs(ih, T, rho, molefrac);
}
