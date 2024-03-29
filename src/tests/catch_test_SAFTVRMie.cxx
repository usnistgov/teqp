#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include <fstream>

#include "teqp/types.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/math/finite_derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/constants.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/algorithms/VLE.hpp"
using namespace teqp::SAFTVRMie;
using namespace teqp;

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/cpp_complex.hpp"
#include "teqp/math/quadrature.hpp"

TEST_CASE("Check get_names and get_BibTeXKeys", "[SAFTVRMIE]")
{
    std::vector<std::string> names = { "Methane", "Ethane" };
    auto model = SAFTVRMieMixture(names);
    CHECK(model.get_names()[0] == "Methane");
    CHECK(model.get_BibTeXKeys()[0] == "Lafitte-JCP-2001");
}

TEST_CASE("Check integration", "[SAFTVRMIE]"){
    std::function<double(double)> f = [](const double&x){ return x*sin(x); };
    auto exact = -2*cos(1) + 2*sin(1);
    auto deg3 = quad<3, double>(f, -1.0, 1.0);
    auto deg4 = quad<4, double>(f, -1.0, 1.0);
    auto deg5 = quad<5, double>(f, -1.0, 1.0);
    CHECK(deg4 == Approx(exact).margin(1e-12));
    CHECK(deg5 == Approx(exact).margin(1e-12));
}

TEST_CASE("Check integration for d", "[SAFTVRMIE]"){
    double epskB = 206.12;
    double sigma_m = 3.7257e-10;
    double lambda_r = 12.4;
    double lambda_a = 6.0;
    double C = lambda_r/(lambda_r-lambda_a)*::pow(lambda_r/lambda_a,lambda_a/(lambda_r-lambda_a));
    double T = 300.0;
    std::function<double(double)> integrand = [&epskB, &sigma_m, &C, &T, &lambda_a, &lambda_r](const double& r_m){
        auto u = C*epskB*(::pow(sigma_m/r_m, lambda_r) - ::pow(sigma_m/r_m, lambda_a));
        return -expm1(-u/T);
    };
    auto d30 = quad<30, double>(integrand, 0.0, sigma_m);
    auto d15 = quad<15, double>(integrand, 0.0, sigma_m);
    auto d7 = quad<7, double>(integrand, 0.0, sigma_m);
    auto d5 = quad<5, double>(integrand, 0.0, sigma_m);
    auto d3 = quad<3, double>(integrand, 0.0, sigma_m);
    CHECK(d30 == Approx(3.597838581533227e-10).margin(1e-12));
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
    
    // Check values from Clapeyron.jl
    CHECK(core.a1kB == Approx(-694.2608061145818));
    CHECK(core.a2kB2 == Approx(-6741.101051705587));
    CHECK(core.a3kB3 == Approx(-81372.77460911816));
    CHECK(core.alphar_mono == Approx(-1.322739797471788));
    CHECK(core.alphar_chain == Approx(-0.0950261207746853));
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

TEST_CASE("A mixture calculation", "[SAFTVRMie]"){
    std::vector<std::string> names = {"Methane","Ethane","Propane"};
    SAFTVRMieMixture model_{names};
    double T = 300.0, rho = 1000.0;
    nlohmann::json model2 = {
        {"names", names}
    };
    nlohmann::json j2 = {
        {"kind", "PCSAFT"},
        {"model", model2}
    };
    auto model3 = cppinterface::make_model(j2);
    auto z = (Eigen::ArrayXd(3) << 0.3, 0.4, 0.3).finished();
    auto rhovec = (rho*z).eval();
    auto fugcoeff = model3->get_fugacity_coefficients(T, rhovec);
}

TEST_CASE("Trace critical locus", "[SAFTVRMie]"){
    std::vector<std::string> names = {"Methane", "Ethane"};
    nlohmann::json model2 = {
        {"names", names}
    };
    nlohmann::json j2 = {
        {"kind", "SAFT-VR-Mie"},
        {"model", model2}
    };
    auto model3 = cppinterface::make_model(j2);
    auto [T0, rho0] = model3->solve_pure_critical(300, 10000, nlohmann::json{{"alternative_pure_index", 0},{"alternative_length", names.size()}});
    auto rhovec0 = (Eigen::ArrayXd(2) << rho0, 0).finished();
    auto al = model3->trace_critical_arclength_binary(T0, rhovec0, "aa.txt");
    int rr = 0;
}

TEST_CASE("VLE pure tracing", "[SAFTVRMieVLE]"){
    std::vector<std::string> names1 = {"Ethane"};
    SAFTVRMieMixture pure{names1};
    nlohmann::json spec1{
        {"Tcguess", 300.0},
        {"rhocguess", 10000.0},
        {"Tred", 0.999},
        {"Nstep", 100},
        {"with_deriv", true}
    };
    auto o1 = pure_trace_VLE(pure, 300, spec1);
    
    std::vector<std::string> names = {"Methane", "Ethane"};
    SAFTVRMieMixture model{names};
    nlohmann::json pure_spec{{"alternative_pure_index", 1}, {"alternative_length", names.size()}};
    nlohmann::json spec{
        {"Tcguess", 300.0},
        {"rhocguess", 10000.0},
        {"pure_spec", pure_spec},
        {"Tred", 0.999},
        {"Nstep", 100},
        {"with_deriv", true}
    };
    auto o = pure_trace_VLE(model, 300, spec);
    CHECK(o[0] == Approx(o1[0]));
}

TEST_CASE("VLE isotherm tracing", "[SAFTVRMieVLE]"){
    
    std::vector<std::string> names = {"Ethane", "Propane"};
    SAFTVRMieMixture model{names};
    nlohmann::json pure_spec{{"alternative_pure_index", 0}, {"alternative_length", names.size()}};
    nlohmann::json spec{
        {"Tcguess", 300.0},
        {"rhocguess", 10000.0},
        {"pure_spec", pure_spec},
        {"Tred", 0.999},
        {"Nstep", 100},
        {"with_deriv", true}
    };
    double T = 280.0;
    auto purestart = pure_trace_VLE(model, T, spec);
    Eigen::ArrayXd rhovecL0(2), rhovecV0(2);
    int ipure = pure_spec.at("alternative_pure_index");
    rhovecL0(ipure) = purestart[0]; rhovecV0(ipure) = purestart[1];
    auto der = get_drhovecdp_Tsat(model, T, rhovecL0, rhovecV0);
    auto opt = TVLEOptions(); opt.revision = 2; opt.polish = true; opt.init_c = -1;
    auto iso = trace_VLE_isotherm_binary(model, T, rhovecL0, rhovecV0, opt);
//    std::cout << iso.dump(2) << std::endl;
}

template<typename Model>
auto get_Theta2_dilute(const Model& model, double T, const Eigen::ArrayXd& z = (Eigen::ArrayXd(1) << 1.0).finished()){
    auto B = model->get_B2vir(T, z);
    auto dBdT = model->get_dmBnvirdTm(2, 1, T, z);
    return B + T*dBdT;
}

TEST_CASE("Get the Theta_2 from the virial coefficients", "[SAFTVRMie]"){
    nlohmann::json j{{"kind","SAFT-VR-Mie"},{"model", {{"names", {"Propane"}}}}};
    auto model = cppinterface::make_model(j);
    CHECK(std::isfinite(get_Theta2_dilute(model, 300.0)));
}

TEST_CASE("Check that bad kmat options throw", "[SAFTVRMie]"){
    nlohmann::json kmat1 = nlohmann::json::array(); // empty
    auto kmat2 = kmat1; kmat2.push_back(0); // just one entry, 1D (should be 2D)
    auto kmat3 = kmat2; kmat3.push_back(0); // two entries, but 1D (should be 2D)
    for (auto bad_kmat: {kmat2, kmat3}){
        nlohmann::json j{{"kind","SAFT-VR-Mie"},{"model", {{"names", {"Propane","Ethane"}},{"kmat",bad_kmat}}}};
//        std::cout << j << std::endl;
        auto sj = j.dump(2);
        CAPTURE(sj);
        CHECK_THROWS(cppinterface::make_model(j));
//        try{
//            cppinterface::make_model(j);
//        }
//        catch(const teqpException& e){
//            std::cout << e.what() << std::endl;
//        }
    }
}

TEST_CASE("Test diameter calculations", "[SAFTVRMie]"){
    std::vector<std::string> names = {"Ethane"};
    
    SAFTVRMieMixture model{names};
    auto tol = 1e-20;
    // Check values are for the reciprocal of the variable used here
    std::vector<std::tuple<double, double>> check_vals{
        {50.0, 1/0.88880166306088},
        {100.0, 1/0.8531852567786484},
        {300.0, 1/0.7930471375700721},
        {1000.0, 1/0.7265751754644022}
    };
    SECTION("double"){
        for (auto [T,jval] : check_vals){
            auto j = model.get_terms().get_j_cutoff_dii(0, T);
            auto d = model.get_terms().get_dii(0, T);
            CHECK(j == Approx(jval).margin(tol));
        }
    }
    SECTION("std::complex<double>"){
        for (auto [T, jval] : check_vals){
            CHECK(std::real(model.get_terms().get_j_cutoff_dii(0,std::complex<double>(T,1e-100))) == Approx(jval).margin(tol));
        }
    }
    SECTION("autodiff<double>"){
        for (auto [T, jval] : check_vals){
            using adtype = autodiff::HigherOrderDual<1, double>;
            adtype Tad = T;
            CHECK(getbaseval(model.get_terms().get_j_cutoff_dii(0,Tad)) == Approx(jval).margin(tol));
        }
    }
}

TEST_CASE("Check B and its temperature derivatives", "[SAFTVRMie],[B]")
{
    auto j = nlohmann::json::parse(R"({
        "kind": "SAFT-VR-Mie",
        "model": {
            "names": ["Methane"]
        }
    })");
    CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    auto model = teqp::cppinterface::make_model(j);
    double rhotest = 1e-3; double Tspec = 100;
    Eigen::ArrayXd z(1); z[0] = 1.0;
    
    auto Bnondilute = model->get_Ar00(Tspec, rhotest, z)/rhotest;
    auto B = model->get_dmBnvirdTm(2, 0, Tspec, z);
    CHECK(B == Approx(Bnondilute));

    auto TdBdTnondilute = -model->get_Ar10(Tspec, rhotest, z)/rhotest;
    auto TdBdT = Tspec*model->get_dmBnvirdTm(2, 1, Tspec, z);
    CHECK(TdBdT == Approx(TdBdTnondilute));
}

TEST_CASE("Check output of dmat", "[SAFTVRMiedmat]")
{
    auto j = nlohmann::json::parse(R"({
        "kind": "SAFT-VR-Mie",
        "model": {
            "names": ["Methane", "Ethane"]
        }
    })");
    CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    auto ptr = teqp::cppinterface::make_model(j);
    const auto& model = teqp::cppinterface::adapter::get_model_cref<SAFTVRMie::SAFTVRMieMixture>(ptr.get());
    auto z = (Eigen::ArrayXd(2) << 0.300, 0.7).finished();
    auto jj = model.get_core_calcs(300, 0.5, z);
    // std::cout << jj.dump(2) << std::endl;
}

TEST_CASE("Check ln(phi) and its derivatives", "[SAFTVRMielnphi]")
{
    std::vector<std::string> names = {"Methane", "Ethane"};
    SAFTVRMieMixture model{names};
    double T = 300.0, dT = 1e-5;
    auto rhovec = (Eigen::ArrayXd(2) << 300, 200).finished();
    Eigen::ArrayXd molefracs = forceeval(rhovec/rhovec.sum());
    using iso = IsochoricDerivatives<decltype(model)>;
    SECTION("T deriv"){
        auto lnphin1 = iso::get_ln_fugacity_coefficients(model, T-dT, rhovec);
        auto lnphip1 = iso::get_ln_fugacity_coefficients(model, T+dT, rhovec);
        auto analytical = iso::get_d_ln_fugacity_coefficients_dT_constrhovec(model, T, rhovec);
        auto findiff = (lnphip1-lnphin1)/(2*dT);
        auto err = ((findiff - analytical)/analytical).cwiseAbs().mean();
        CHECK(err < 1e-7);
    }
    SECTION("dZdrho"){
        auto [lnZ, Z, dZdrho] = iso::get_lnZ_Z_dZdrho(model, T, rhovec);
        auto rhovecplus = rhovec*1.001, rhovecminus = rhovec*0.999;
        auto rhoplus = rhovecplus.sum(), rhominus = rhovecminus.sum();
        auto lnZn1 = std::get<0>(iso::get_lnZ_Z_dZdrho(model, T, rhovecminus));
        auto lnZp1 = std::get<0>(iso::get_lnZ_Z_dZdrho(model, T, rhovecplus));
        auto findiff = ((lnZp1-lnZn1)/(rhoplus-rhominus));
        auto analytical = dZdrho/Z;
        auto err = ((findiff - analytical)/analytical);
        CHECK(err < 1e-7);
    }
    SECTION("rho deriv"){
        auto analytical = iso::get_d_ln_fugacity_coefficients_drho_constTmolefracs(model, T, rhovec);
        auto rhovecplus = rhovec*1.001, rhovecminus = rhovec*0.999;
        auto rhoplus = rhovecplus.sum(), rhominus = rhovecminus.sum();
        auto lnphin1 = iso::get_ln_fugacity_coefficients(model, T, rhovecminus);
        auto lnphip1 = iso::get_ln_fugacity_coefficients(model, T, rhovecplus);
        auto findiff = ((lnphip1-lnphin1)/(rhoplus-rhominus)).eval();
        auto err = ((findiff - analytical)/analytical).cwiseAbs().mean();
        CHECK(err < 1e-7);
    }
    SECTION("mole frac derivs"){
        auto analytical = iso::get_d_ln_fugacity_coefficients_dmolefracs_constTrho(model, T, rhovec);
        auto rhotot = rhovec.sum();
        auto N = names.size();
        double dx = 1e-5;
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> numerical(N, N);
        for (auto i = 0; i < N; ++i){
            auto molefracsplus = molefracs; molefracsplus[i] += dx;
            auto molefracsminus = molefracs; molefracsminus[i] -= dx;
            // Here you need to use the special method for testing that takes T, rho, molefracs because
            // if you provide molar concentrations that are shifted, you introduce an inconsistency
            // during the mole fraction calculation step. Either you get the right molar density or the
            // right mole fraction, impossible to do both simultaneously.
            auto plus = iso::get_ln_fugacity_coefficients_Trhomolefracs(model, T, rhotot, molefracsplus);
            auto minus = iso::get_ln_fugacity_coefficients_Trhomolefracs(model, T, rhotot, molefracsminus);
            numerical.col(i) = (plus-minus)/(2*dx);
        }
        auto worst_error = (analytical.array() - numerical.array()).cwiseAbs().maxCoeff();
        CHECK(worst_error < 1e-10);
    }
}
