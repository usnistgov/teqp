#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/core.hpp"
#include "teqp/models/pcsaft.hpp"
using namespace teqp::PCSAFT;

TEST_CASE("Single alphar check value", "[PCSAFT]")
{
    std::vector<std::string> names = { "Methane" };
    auto model = PCSAFTMixture(names);
    double T = 200, Dmolar = 300;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    using tdx = teqp::TDXDerivatives<decltype(model), double>;
    CHECK(tdx::get_Ar00(model, T, Dmolar, z) == Approx(-0.032400020930842724));
}

TEST_CASE("Check PCSAFT with kij", "[PCSAFT]")
{
    std::vector<std::string> names = { "Methane", "Ethane" };
    Eigen::ArrayXXd kij_right(2, 2); kij_right.setZero();
    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();

    SECTION("No kij") {
        CHECK_NOTHROW(PCSAFTMixture(names));
    }
    SECTION("Correctly shaped kij matrix") {
        CHECK_NOTHROW(PCSAFTMixture(names, kij_right));
    }
    SECTION("Incorrectly shaped kij matrix") {
        CHECK_THROWS(PCSAFTMixture(names, kij_bad));
    }
}

TEST_CASE("Check PCSAFT with kij and coeffs", "[PCSAFT]")
{
    std::vector<teqp::PCSAFT::SAFTCoeffs> coeffs;
    std::vector<double> eoverk = { 120,130 }, m = { 1,2 }, sigma = { 0.9, 1.1 };
    for (auto i = 0; i < eoverk.size(); ++i) {
        teqp::PCSAFT::SAFTCoeffs c;
        c.m = m[i];
        c.sigma_Angstrom = sigma[i];
        c.epsilon_over_k = eoverk[i];
        coeffs.push_back(c);
    }

    Eigen::ArrayXXd kij_right(2, 2); kij_right.setZero();
    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();

    SECTION("No kij") {
        CHECK_NOTHROW(PCSAFTMixture(coeffs));
    }
    SECTION("Correctly shaped kij matrix") {
        CHECK_NOTHROW(PCSAFTMixture(coeffs, kij_right));
    }
    SECTION("Incorrectly shaped kij matrix") {
        CHECK_THROWS(PCSAFTMixture(coeffs, kij_bad));
    }
}