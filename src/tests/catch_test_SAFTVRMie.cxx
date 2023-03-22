#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include <fstream>

#include "teqp/core.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/math/finite_derivs.hpp"
#include "teqp/json_builder.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/constants.hpp"
#include "teqp/algorithms/critical_pure.hpp"
using namespace teqp::SAFTVRMie;
using namespace teqp;

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/cpp_complex.hpp"
#include "teqp/math/quadrature.hpp"

TEST_CASE("Check integration", "[SAFTVRMIE]"){
    std::function<double(double)> f = [](const double&x){ return x*sin(x); };
    auto exact = -2*cos(1) + 2*sin(1);
    auto deg3 = quad<3, double>(f, -1, 1);
    auto deg4 = quad<4, double>(f, -1, 1);
    auto deg5 = quad<5, double>(f, -1, 1);
    CHECK(deg4 == Approx(exact).margin(1e-12));
    CHECK(deg5 == Approx(exact).margin(1e-12));
}

TEST_CASE("Check integration for d", "[SAFTVRMIE]"){
    // Check integration for d
    auto m = (Eigen::ArrayXd(1) << 1.4373).finished();
    auto eps = (Eigen::ArrayXd(1) << 206.12).finished();
    auto sigma = (Eigen::ArrayXd(1) << 3.7257e-10).finished();
    auto lambda_r = (Eigen::ArrayXd(1) << 12.4).finished();
    auto lambda_a = (Eigen::ArrayXd(1) << 6.0).finished();
    Eigen::ArrayXXd kmat = Eigen::ArrayXXd::Zero(1,1);
    SAFTVRMieChainContributionTerms terms(m, eps, sigma, lambda_r, lambda_a, kmat);
    double T = 100.0;
    std::function<double(double)> integrand = [&terms, &T](const double& r){
        return 1.0-exp(-terms.get_uii_over_kB(0, r)/T);
    };
    auto d30 = quad<30, double>(integrand, 0.0, terms.sigma_m[0]);
    auto d15 = quad<15, double>(integrand, 0.0, terms.sigma_m[0]);
    auto d7 = quad<7, double>(integrand, 0.0, terms.sigma_m[0]);
    auto d5 = quad<5, double>(integrand, 0.0, terms.sigma_m[0]);
    auto d3 = quad<3, double>(integrand, 0.0, terms.sigma_m[0]);
//    CHECK(d30 == Approx(exact).margin(1e-12));
}

//TEST_CASE("Check an*", "[SAFTVRMie]"){
//    auto m = (Eigen::ArrayXd(1) << 1.4373).finished();
//    auto eps = (Eigen::ArrayXd(1) << 206.12).finished();
//    auto sigma = (Eigen::ArrayXd(1) << 3.7257e-10).finished();
//    auto lambda_a = (Eigen::ArrayXd(1) << 6.0).finished();
//    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
//    Eigen::ArrayXXd kmat = Eigen::ArrayXXd::Zero(1,1);
//
//    std::ofstream ofs("perturb_terms.csv", std::ofstream::out);
//    ofs << "lambda_r,rhos*,a1*,a2*,a3*" << std::endl;
//    for (auto lambda_r_ : {8, 12, 14, 20, 30}){
//        auto lambda_r = (Eigen::ArrayXd(1) << lambda_r_).finished();
//
//        SAFTVRMieChainContributionTerms terms(m, eps, sigma, lambda_r, lambda_a, kmat);
//
//        for (auto rho = 100; rho < 40000; rho *= 1.05){
//            auto core = terms.get_core_calcs(eps[0], rho, z);
//            auto rhosstar = core.rhos*pow(sigma[0], 3);
//            if (rhosstar > 1.0){
//                break;
//            }
//            auto a1star = core.a1kB/eps[0];
//            auto a2star = core.a2kB2/pow(eps[0],2);
//            auto a3star = core.a3kB3/pow(eps[0],3);
//
//            ofs << lambda_r_ << "," << rhosstar << "," << a1star << "," << a2star << "," << a3star << std::endl;
//        }
//    }
//}

TEST_CASE("Single alphar check value", "[SAFTVRMie]")
{
    auto m = (Eigen::ArrayXd(1) << 1.4373).finished();
    auto eps = (Eigen::ArrayXd(1) << 206.12).finished();
    auto sigma = (Eigen::ArrayXd(1) << 3.7257e-10).finished();
    auto lambda_r = (Eigen::ArrayXd(1) << 12.4).finished();
    auto lambda_a = (Eigen::ArrayXd(1) << 6.0).finished();
    Eigen::ArrayXXd kmat = Eigen::ArrayXXd::Zero(1,1);
    SAFTVRMieChainContributionTerms terms(m, eps, sigma, lambda_r, lambda_a, kmat);
    
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto core = terms.get_core_calcs(300.0, 12000.0, z);
    CHECK(core.a1kB == Approx(-694.2608061145818));
    CHECK(core.a2kB2 == Approx(-6741.101051705587));
    CHECK(core.a3kB3 == Approx(-81372.77460911816));
    CHECK(core.a_mono == Approx(-1.322739797471788));
    CHECK(core.a_chain == Approx(-0.0950261207746853));
}

//TEST_CASE("Check 0n derivatives", "[PCSAFT]")
//{
//    std::vector<std::string> names = { "Methane", "Ethane" };
//    auto model = PCSAFTMixture(names);
//
//    const double T = 100.0;
//    const double rho = 126.1856883066021;
//    const auto rhovec = (Eigen::ArrayXd(2) << rho, 0).finished();
//    const auto molefrac = rhovec / rhovec.sum();
//
//    using my_float_type = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100U>>;
//    my_float_type D = rho, h = pow(my_float_type(10.0), -10);
//    auto fD = [&](const auto& x) { return model.alphar(T, x, molefrac); };
//
//    using tdx = TDXDerivatives<decltype(model)>;
//    auto Ar02 = tdx::get_Ar02(model, T, rho, molefrac);
//    auto Ar02n = tdx::get_Ar0n<2>(model, T, rho, molefrac)[2];
//    auto Ar02mp = static_cast<double>((D * D) * centered_diff<2, 4>(fD, D, h));
//    auto Ar02mcx = tdx::get_Ar0n<2, ADBackends::multicomplex>(model, T, rho, molefrac)[2];
//    CAPTURE(Ar02);
//    CAPTURE(Ar02n);
//    CAPTURE(Ar02mp);
//    CAPTURE(Ar02mcx);
//    CHECK(std::abs(Ar02 - Ar02n) < 1e-13);
//    CHECK(std::abs(Ar02 - Ar02mp) < 1e-13);
//    CHECK(std::abs(Ar02 - Ar02mcx) < 1e-13);
//
//    auto Ar01 = tdx::get_Ar01(model, T, rho, molefrac);
//    auto Ar01n = tdx::get_Ar0n<1>(model, T, rho, molefrac)[1];
//    auto Ar01mcx = tdx::get_Ar0n<1, ADBackends::multicomplex>(model, T, rho, molefrac)[1];
//    auto Ar01csd = tdx::get_Ar01<ADBackends::complex_step>(model, T, rho, molefrac);
//    auto Ar01mp = static_cast<double>(D * centered_diff<1, 4>(fD, D, h));
//    CAPTURE(Ar01);
//    CAPTURE(Ar01n);
//    CAPTURE(Ar01mp);
//    CAPTURE(Ar01mcx);
//    CAPTURE(Ar01csd);
//    CHECK(std::abs(Ar01 - Ar01n) < 1e-13);
//    CHECK(std::abs(Ar01 - Ar01mp) < 1e-13);
//    CHECK(std::abs(Ar01 - Ar01mcx) < 1e-13);
//    CHECK(std::abs(Ar01 - Ar01csd) < 1e-13);
//
//    auto Ar03 = tdx::get_Arxy<0, 3, ADBackends::autodiff>(model, T, rho, molefrac);
//    auto Ar03n = tdx::get_Ar0n<3>(model, T, rho, molefrac)[3];
//    auto Ar03mp = static_cast<double>((D * D * D) * centered_diff<3, 4>(fD, D, h));
//    auto Ar03mcx = tdx::get_Ar0n<3, ADBackends::multicomplex>(model, T, rho, molefrac)[3];
//    CAPTURE(Ar03);
//    CAPTURE(Ar03n);
//    CAPTURE(Ar03mp);
//    CAPTURE(Ar03mcx);
//    CHECK(std::abs(Ar03 - Ar03n) < 1e-13);
//    CHECK(std::abs(Ar03 - Ar03mp) < 1e-13);
//    CHECK(std::abs(Ar03 - Ar03mcx) < 1e-13);
//
//    auto Ar04 = tdx::get_Arxy<0, 4, ADBackends::autodiff>(model, T, rho, molefrac);
//    auto Ar04n = tdx::get_Ar0n<4>(model, T, rho, molefrac)[4];
//    auto Ar04mp = static_cast<double>((D * D * D * D) * centered_diff<4, 4>(fD, D, h));
//    auto Ar04mcx = tdx::get_Ar0n<4, ADBackends::multicomplex>(model, T, rho, molefrac)[4];
//    CAPTURE(Ar04);
//    CAPTURE(Ar04n);
//    CAPTURE(Ar04mp);
//    CAPTURE(Ar04mcx);
//    CHECK(std::abs(Ar04 - Ar04n) < 1e-13);
//    CHECK(std::abs(Ar04 - Ar04mp) < 1e-13);
//    CHECK(std::abs(Ar04 - Ar04mcx) < 1e-13);
//}
//
//TEST_CASE("Check neff", "[virial]")
//{
//    double T = 298.15;
//    double rho = 3.0;
//    const Eigen::Array2d molefrac = { 0.5, 0.5 };
//    auto f = [&T, &rho, &molefrac](const auto& model) {
//        auto neff = TDXDerivatives<decltype(model)>::get_neff(model, T, rho, molefrac);
//        CAPTURE(neff);
//        CHECK(neff > 0);
//        CHECK(neff < 100);
//    };
//    // This quantity is undefined for the van der Waals EOS because Ar20 is always 0
//    //SECTION("vdW") {
//    //    f(build_simple());
//    //}
//    SECTION("PCSAFT") {
//        std::vector<std::string> names = { "Methane", "Ethane" };
//        f(PCSAFTMixture(names));
//    }
//}
//
//
//TEST_CASE("Check PCSAFT with kij", "[PCSAFT]")
//{
//    std::vector<std::string> names = { "Methane", "Ethane" };
//    Eigen::ArrayXXd kij_right(2, 2); kij_right.setZero();
//    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();
//
//    SECTION("No kij") {
//        CHECK_NOTHROW(PCSAFTMixture(names));
//    }
//    SECTION("Correctly shaped kij matrix") {
//        CHECK_NOTHROW(PCSAFTMixture(names, kij_right));
//    }
//    SECTION("Incorrectly shaped kij matrix") {
//        CHECK_THROWS(PCSAFTMixture(names, kij_bad));
//    }
//}
//
//TEST_CASE("Check PCSAFT with kij and coeffs", "[PCSAFT]")
//{
//    std::vector<teqp::PCSAFT::SAFTCoeffs> coeffs;
//    std::vector<double> eoverk = { 120,130 }, m = { 1,2 }, sigma = { 0.9, 1.1 };
//    for (auto i = 0; i < eoverk.size(); ++i) {
//        teqp::PCSAFT::SAFTCoeffs c;
//        c.m = m[i];
//        c.sigma_Angstrom = sigma[i];
//        c.epsilon_over_k = eoverk[i];
//        coeffs.push_back(c);
//    }
//
//    Eigen::ArrayXXd kij_right(2, 2); kij_right.setZero();
//    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();
//
//    SECTION("No kij") {
//        CHECK_NOTHROW(PCSAFTMixture(coeffs));
//    }
//    SECTION("Correctly shaped kij matrix") {
//        CHECK_NOTHROW(PCSAFTMixture(coeffs, kij_right));
//    }
//    SECTION("Incorrectly shaped kij matrix") {
//        CHECK_THROWS(PCSAFTMixture(coeffs, kij_bad));
//    }
//}
//
//TEST_CASE("Check PCSAFT with dipole for acetone", "[PCSAFTD]")
//{
//    std::vector<teqp::PCSAFT::SAFTCoeffs> coeffs;
//    std::vector<double> eoverk = { 232.99 }, m = { 2.7447 }, sigma = { 3.2742 };
//    // The conversion factor with inputs in Debye, Angstroms, and K to non-dimensional quantity
//    auto conv_factor = pow(3.33564e-30,2)/(4*EIGEN_PI*8.8541878128e-12*1.380649e-23*1e-30);
//    conv_factor = 1e4/1.3807;
//    auto muD = 2.88; // [D]
//    auto mustar2 = conv_factor*muD*muD/(m[0]*eoverk[0]*pow(sigma[0], 3));
//
//    for (auto i = 0; i < eoverk.size(); ++i) {
//        teqp::PCSAFT::SAFTCoeffs c;
//        c.m = m[i];
//        c.sigma_Angstrom = sigma[i];
//        c.epsilon_over_k = eoverk[i];
//        c.mustar2 = mustar2;
//        c.nmu = 1;
//        coeffs.push_back(c);
//    }
//    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
//    auto model = PCSAFT::PCSAFTMixture(coeffs, {});
//    auto alphar = model.alphar(300.0, 300.0, z);
//
//    // Build from JSON
//    nlohmann::json jcoeffs = nlohmann::json::array();
//    jcoeffs.push_back({ {"name", "acetone"}, { "m", m[0] }, { "sigma_Angstrom", sigma[0]},{"epsilon_over_k", eoverk[0]}, {"BibTeXKey", "Gross-IECR-2001"}, {"(mu^*)^2", mustar2}, {"nmu", 1.0} });
//    nlohmann::json jmodel = {
//        {"coeffs", jcoeffs}
//    };
//    nlohmann::json j = {
//        {"kind", "PCSAFT"},
//        {"model", jmodel}
//    };
//    auto modelj = cppinterface::make_model(j);
//    auto alpharj = modelj->get_Ar00(300.0, 300.0, z);
//
//    double rhoc = 275/0.05808; // [kg/m^3] to [mol/m^3]
//    auto crit = solve_pure_critical(model, 510.0, rhoc);
//    CHECK(std::get<0>(crit) == Approx(520).margin(10));
//    CHECK(alphar == alpharj);
//}
//
//TEST_CASE("Check PCSAFT with quadrupole for CO2", "[PCSAFTQ]")
//{
//    std::vector<teqp::PCSAFT::SAFTCoeffs> coeffs;
//    std::vector<double> eoverk = { 169.33 }, m = { 1.5131 }, sigma = { 3.1869 };
//    // The conversion factor with inputs in Debye, Angstroms, and K to non-dimensional quantity
//    auto conv_factor = 1e-69/1.380649e-23/1e-50;
//    auto QDA = 4.4; // [DA]
//    auto conv_factorme = pow(3.33564e-40,2)/(4*EIGEN_PI*8.8541878128e-12*1.380649e-23*1e-50);
//    auto Qstar2 = conv_factor*QDA*QDA/(m[0]*eoverk[0]*pow(sigma[0], 5));
//    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
//
//    // Build from JSON
//    nlohmann::json jcoeffs = nlohmann::json::array();
//    jcoeffs.push_back({ {"name", "CO2"}, { "m", m[0] }, { "sigma_Angstrom", sigma[0]},{"epsilon_over_k", eoverk[0]}, {"BibTeXKey", "Gross-IECR-2001"}, {"(Q^*)^2", Qstar2}, {"nQ", 1.0} });
//    nlohmann::json jmodel = {
//        {"coeffs", jcoeffs}
//    };
//    nlohmann::json j = {
//        {"kind", "PCSAFT"},
//        {"model", jmodel}
//    };
//    auto modelj = cppinterface::make_model(j);
//    auto alpharj = modelj->get_Ar00(300.0, 300.0, z);
//
//    for (auto i = 0; i < eoverk.size(); ++i) {
//        teqp::PCSAFT::SAFTCoeffs c;
//        c.m = m[i];
//        c.sigma_Angstrom = sigma[i];
//        c.epsilon_over_k = eoverk[i];
//        c.Qstar2 = Qstar2;
//        c.nQ = 1;
//        coeffs.push_back(c);
//    }
//
//    auto model = PCSAFT::PCSAFTMixture(coeffs, {});
//    auto alphar = model.alphar(300.0, 300.0, z);
//    CHECK(alpharj == Approx(alphar));
//
//    double rhoc = 275/0.05808; // [kg/m^3] to [mol/m^3]
//    auto crit = solve_pure_critical(model, 310.0, rhoc);
//    CHECK(std::get<0>(crit) == Approx(325).margin(10));
//}
