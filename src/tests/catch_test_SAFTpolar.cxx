#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/saft/correlation_integrals.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include "teqp/types.hpp"

using namespace teqp;
using namespace teqp::SAFTpolar;

TEST_CASE("Evaluation of J^{(n)}", "[LuckasJn]")
{
    LuckasJIntegral J12{12};
    auto Jval = J12.get_J(3.0, 1.0);
}

TEST_CASE("Evaluation of K(xxx, yyy)", "[LuckasKnn]")
{
    auto Kval23 = LuckasKIntegral(222, 333).get_K(1.0, 0.9);
    CHECK(Kval23 == Approx(0.03332).margin(0.02));
    
    auto Kval45 = LuckasKIntegral(444, 555).get_K(1.0, 0.9);
    CHECK(Kval45 == Approx(0.01541).margin(0.02));
}

TEST_CASE("Evaluation of Gubbins and Twu", "[GTLPolar]")
{
    auto sigma_m = (Eigen::ArrayXd(2) << 3e-10, 3.1e-10).finished();
    auto epsilon_over_k= (Eigen::ArrayXd(2) << 200, 300).finished();
    auto mubar2 = (Eigen::ArrayXd(2) << 0.0, 0.5).finished();
    auto Qbar2 = (Eigen::ArrayXd(2) << 0.5, 0).finished();
    MCGTL GTL{sigma_m, epsilon_over_k, mubar2, Qbar2};
    auto z = (Eigen::ArrayXd(2) << 0.1, 0.9).finished();
    auto rhoN = std::complex<double>(300, 1e-100);
    GTL.eval(300.0, rhoN, z);
}

// This test is used to make sure that replacing std::abs with a more flexible function
// that can handle differentation types like std::complex<double> is still ok

TEST_CASE("Check derivative of |x|", "[diffabs]")
{
    double h = 1e-100;
    auto safe_abs1 = [](const auto&x) { return sqrt(x*x); };
    SECTION("|x|"){
        CHECK(safe_abs1(3.1) == 3.1);
        CHECK(safe_abs1(-3.1) == 3.1);
    }
    SECTION("sqrt(x^2)"){
        CHECK(safe_abs1(std::complex<double>(3.1, h)).imag()/h == 1);
        CHECK(safe_abs1(std::complex<double>(-3.1, h)).imag()/h == -1);
    }
    auto safe_abs2 = [](const auto&x) { return (getbaseval(x) < 0) ? -x : x; };
    SECTION("check base"){
        CHECK(safe_abs2(std::complex<double>(3.1, h)).imag()/h == 1);
        CHECK(safe_abs2(std::complex<double>(-3.1, h)).imag()/h == -1);
    }
}
