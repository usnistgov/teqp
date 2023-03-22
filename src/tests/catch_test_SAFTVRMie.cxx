#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include <fstream>

#include "teqp/core.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/models/pcsaft.hpp"
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
    double epskB = 206.12;
    double sigma_m = 3.7257e-10;
    double lambda_r = 12.4;
    double lambda_a = 6.0;
    double C = lambda_r/(lambda_r-lambda_a)*::pow(lambda_r/lambda_a,lambda_a/(lambda_r-lambda_a));
    double T = 100.0;
    std::function<double(double)> integrand = [&epskB, &sigma_m, &C, &T, &lambda_a, &lambda_r](const double& r_m){
        auto u = C*epskB*(::pow(sigma_m/r_m, lambda_r) - ::pow(sigma_m/r_m, lambda_a));
        return 1.0-exp(-u/T);
    };
    auto d30 = quad<30, double>(integrand, 0.0, sigma_m);
    auto d15 = quad<15, double>(integrand, 0.0, sigma_m);
    auto d7 = quad<7, double>(integrand, 0.0, sigma_m);
    auto d5 = quad<5, double>(integrand, 0.0, sigma_m);
    auto d3 = quad<3, double>(integrand, 0.0, sigma_m);
    CHECK(d30 == Approx(3.668937640717724e-10).margin(1e-12));
}

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
//    CHECK(core.a1kB == Approx(-694.2608061145818));
    //CHECK(core.a2kB2 == Approx(-6741.101051705587));
    CHECK(core.a3kB3 == Approx(-81372.77460911816));
    //CHECK(core.alphar_mono == Approx(-1.322739797471788));
    //CHECK(core.alphar_chain == Approx(-0.0950261207746853));
}

template<int i, int j, typename Model, typename TTYPE, typename RhoType, typename VecType>
auto ijcheck(const Model& model, const TTYPE& T, const RhoType& rho, const VecType& z, double margin=1e-103){
    using tdx = TDXDerivatives<decltype(model)>;
    auto Arxy = tdx::template get_Arxy<i, j, ADBackends::autodiff>(model, T, rho, z);
    auto Arxymcx = tdx::template get_Arxy<i, j, ADBackends::multicomplex>(model, T, rho, z);
    CAPTURE(i);
    CAPTURE(j);
    CHECK(Arxymcx == Approx(Arxy).margin(margin));
    CHECK(std::isfinite(Arxymcx));
}

TEST_CASE("Check all xy derivs", "[SAFTVRMie]")
{
    Eigen::ArrayXXd kmat = Eigen::ArrayXXd::Zero(1,1);
    std::vector<std::string> names = {"Ethane"};
    
    SAFTVRMieMixture model{names, kmat};
    
    double T = 300.0, rho = 10000.0;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    ijcheck<0,0>(model, T, rho, z);
    ijcheck<0,1>(model, T, rho, z);
    ijcheck<0,2>(model, T, rho, z);
    ijcheck<0,3>(model, T, rho, z);
    ijcheck<1,0>(model, T, rho, z);
    ijcheck<1,1>(model, T, rho, z);
    ijcheck<1,2>(model, T, rho, z);
    ijcheck<1,3>(model, T, rho, z);
    ijcheck<2,0>(model, T, rho, z);
    ijcheck<2,1>(model, T, rho, z);
    ijcheck<2,2>(model, T, rho, z);
    ijcheck<2,3>(model, T, rho, z);
    int rr = 0;
}

TEST_CASE("Solve for critical point with three interface approaches", "[SAFTVRMie]")
{
    Eigen::ArrayXXd kmat = Eigen::ArrayXXd::Zero(1,1);
    std::vector<std::string> names = {"Ethane"};
    SAFTVRMieMixture model_{names, kmat};
    double T = 300.0, rho = 10000.0;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto crit1 = solve_pure_critical(model_, 300.0, 10000.0);
    
    nlohmann::json jcoeffs = nlohmann::json::array();
    jcoeffs.push_back({
        {"name", "Ethane"},
        { "m", 1.4373 },
        { "sigma_m", 3.7257e-10},
        {"epsilon_over_k", 206.12},
        {"lambda_r", 12.4},
        {"lambda_a", 6.0},
        {"BibTeXKey", "Lafitte-JCP"}
    });
    nlohmann::json model = {
        {"coeffs", jcoeffs}
    };
    nlohmann::json j = {
        {"kind", "SAFT-VR-Mie"},
        {"model", model}
    };
    auto modelc = cppinterface::make_model(j);
    auto crit2 = modelc->solve_pure_critical(300.0, 10000.0, {});
    CHECK(std::get<0>(crit1) == Approx(std::get<0>(crit2)));
    
    nlohmann::json model2 = {
        {"names", {"Ethane"}}
    };
    nlohmann::json j2 = {
        {"kind", "SAFT-VR-Mie"},
        {"model", model2}
    };
    auto model3 = cppinterface::make_model(j2);
    auto crit3 = model3->solve_pure_critical(300.0, 10000.0, {});
    CHECK(std::get<0>(crit1) == Approx(std::get<0>(crit3)));
    
    auto z3 = (Eigen::ArrayXd(1) << 1.0).finished();
    CHECK(std::isfinite(model3->get_Ar11(300.0, 300.0, z3)));
    
    int rr = 0;
}
