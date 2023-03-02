#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/core.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/finite_derivs.hpp"
#include "teqp/algorithms/critical_pure.hpp"
using namespace teqp::PCSAFT;
using namespace teqp;

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/cpp_complex.hpp"

TEST_CASE("Single alphar check value", "[PCSAFT]")
{
    std::vector<std::string> names = { "Methane" };
    auto model = PCSAFTMixture(names);
    double T = 200, Dmolar = 300;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    using tdx = teqp::TDXDerivatives<decltype(model), double>;
    CHECK(tdx::get_Ar00(model, T, Dmolar, z) == Approx(-0.032400020930842724));
}

TEST_CASE("Check 0n derivatives", "[PCSAFT]")
{
    std::vector<std::string> names = { "Methane", "Ethane" };
    auto model = PCSAFTMixture(names);

    const double T = 100.0;
    const double rho = 126.1856883066021;
    const auto rhovec = (Eigen::ArrayXd(2) << rho, 0).finished();
    const auto molefrac = rhovec / rhovec.sum();

    using my_float_type = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100U>>;
    my_float_type D = rho, h = pow(my_float_type(10.0), -10);
    auto fD = [&](const auto& x) { return model.alphar(T, x, molefrac); };

    using tdx = TDXDerivatives<decltype(model)>;
    auto Ar02 = tdx::get_Ar02(model, T, rho, molefrac);
    auto Ar02n = tdx::get_Ar0n<2>(model, T, rho, molefrac)[2];
    auto Ar02mp = static_cast<double>((D * D) * centered_diff<2, 4>(fD, D, h));
    auto Ar02mcx = tdx::get_Ar0n<2, ADBackends::multicomplex>(model, T, rho, molefrac)[2];
    CAPTURE(Ar02);
    CAPTURE(Ar02n);
    CAPTURE(Ar02mp);
    CAPTURE(Ar02mcx);
    CHECK(std::abs(Ar02 - Ar02n) < 1e-13);
    CHECK(std::abs(Ar02 - Ar02mp) < 1e-13);
    CHECK(std::abs(Ar02 - Ar02mcx) < 1e-13);

    auto Ar01 = tdx::get_Ar01(model, T, rho, molefrac);
    auto Ar01n = tdx::get_Ar0n<1>(model, T, rho, molefrac)[1];
    auto Ar01mcx = tdx::get_Ar0n<1, ADBackends::multicomplex>(model, T, rho, molefrac)[1];
    auto Ar01csd = tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, molefrac);
    auto Ar01mp = static_cast<double>(D * centered_diff<1, 4>(fD, D, h));
    CAPTURE(Ar01);
    CAPTURE(Ar01n);
    CAPTURE(Ar01mp);
    CAPTURE(Ar01mcx);
    CAPTURE(Ar01csd);
    CHECK(std::abs(Ar01 - Ar01n) < 1e-13);
    CHECK(std::abs(Ar01 - Ar01mp) < 1e-13);
    CHECK(std::abs(Ar01 - Ar01mcx) < 1e-13);
    CHECK(std::abs(Ar01 - Ar01csd) < 1e-13);

    auto Ar03 = tdx::get_Arxy<0, 3, ADBackends::autodiff>(model, T, rho, molefrac);
    auto Ar03n = tdx::get_Ar0n<3>(model, T, rho, molefrac)[3];
    auto Ar03mp = static_cast<double>((D * D * D) * centered_diff<3, 4>(fD, D, h));
    auto Ar03mcx = tdx::get_Ar0n<3, ADBackends::multicomplex>(model, T, rho, molefrac)[3];
    CAPTURE(Ar03);
    CAPTURE(Ar03n);
    CAPTURE(Ar03mp);
    CAPTURE(Ar03mcx);
    CHECK(std::abs(Ar03 - Ar03n) < 1e-13);
    CHECK(std::abs(Ar03 - Ar03mp) < 1e-13);
    CHECK(std::abs(Ar03 - Ar03mcx) < 1e-13);

    auto Ar04 = tdx::get_Arxy<0, 4, ADBackends::autodiff>(model, T, rho, molefrac);
    auto Ar04n = tdx::get_Ar0n<4>(model, T, rho, molefrac)[4];
    auto Ar04mp = static_cast<double>((D * D * D * D) * centered_diff<4, 4>(fD, D, h));
    auto Ar04mcx = tdx::get_Ar0n<4, ADBackends::multicomplex>(model, T, rho, molefrac)[4];
    CAPTURE(Ar04);
    CAPTURE(Ar04n);
    CAPTURE(Ar04mp);
    CAPTURE(Ar04mcx);
    CHECK(std::abs(Ar04 - Ar04n) < 1e-13);
    CHECK(std::abs(Ar04 - Ar04mp) < 1e-13);
    CHECK(std::abs(Ar04 - Ar04mcx) < 1e-13);
}

TEST_CASE("Check neff", "[virial]")
{
    double T = 298.15;
    double rho = 3.0;
    const Eigen::Array2d molefrac = { 0.5, 0.5 };
    auto f = [&T, &rho, &molefrac](const auto& model) {
        auto neff = TDXDerivatives<decltype(model)>::get_neff(model, T, rho, molefrac);
        CAPTURE(neff);
        CHECK(neff > 0);
        CHECK(neff < 100);
    };
    // This quantity is undefined for the van der Waals EOS because Ar20 is always 0
    //SECTION("vdW") {
    //    f(build_simple());
    //}
    SECTION("PCSAFT") {
        std::vector<std::string> names = { "Methane", "Ethane" };
        f(PCSAFTMixture(names));
    }
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

TEST_CASE("Check PCSAFT with dipole for acetone", "[PCSAFTD]")
{
    std::vector<teqp::PCSAFT::SAFTCoeffs> coeffs;
    std::vector<double> eoverk = { 232.99 }, m = { 2.7447 }, sigma = { 3.2742 };
    for (auto i = 0; i < eoverk.size(); ++i) {
        teqp::PCSAFT::SAFTCoeffs c;
        c.m = m[i];
        c.sigma_Angstrom = sigma[i];
        c.epsilon_over_k = eoverk[i];
        coeffs.push_back(c);
    }
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto modelPCSAFT = PCSAFT::PCSAFTMixture(coeffs);
    auto alpharPCSAFT = modelPCSAFT.alphar(300.0, 300.0, z);
    
    // The conversion factor with inputs are in Debye, Angstroms, and K to non-dimensional quantity
    auto conv_factor = pow(3.33564e-30,2)/(4*EIGEN_PI*8.8541878128e-12*1.380649e-23*1e-30);
    conv_factor = 1e4/1.3807;
    auto muD = 2.88; // [D]
    auto mustar2 = (Eigen::ArrayXd(1) << conv_factor*muD*muD/(m[0]*eoverk[0]*pow(sigma[0], 3))).finished();
    
    auto n = (Eigen::ArrayXd(1) << 1).finished();
    auto model = PCSAFT::PCSAFTDMixture(coeffs, {}, mustar2, n);
    
    auto alphar = model.alphar(300.0, 300.0, z);
    CHECK(alpharPCSAFT != alphar);
    
    double rhoc = 275/0.05808; // [kg/m^3] to [mol/m^3]
    auto crit = solve_pure_critical(model, 510.0, rhoc);
    CHECK(std::get<0>(crit) == Approx(520).margin(10));
}

