#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "nlohmann/json.hpp"
#include "teqp/cpp/teqpcpp.hpp"

#include "teqp/models/saft/correlation_integrals.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/derivs.hpp"
#include "teqp/finite_derivs.hpp"

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/cpp_complex.hpp"

using namespace teqp;
using namespace teqp::SAFTpolar;

TEST_CASE("Evaluation of J^{(n)}", "[LuckasJn]")
{
    LuckasJIntegral J12{12};
    auto Jval = J12.get_J(3.0, 1.0);
    CHECK(Jval != 0);
}

TEST_CASE("Evaluation of K(xxx, yyy)", "[LuckasKnn]")
{
    auto Kval23 = LuckasKIntegral(222, 333).get_K(1.0, 0.9);
    CHECK(Kval23 == Approx(0.03332).margin(0.02));
    
    auto Kval45 = LuckasKIntegral(444, 555).get_K(1.0, 0.9);
    CHECK(Kval45 == Approx(0.01541).margin(0.02));
}

TEST_CASE("Evaluation of J^{(n)}", "[checkJvals]")
{
    double Tstar = 2.0, rhostar = 0.4;
    SECTION("Luckas v. G&T"){
        for (auto n = 4; n < 16; ++n){
            CAPTURE(n);
            auto L = LuckasJIntegral(n).get_J(Tstar, rhostar);
            auto G = GubbinsTwuJIntegral(n).get_J(Tstar, rhostar);
            CHECK(L == Approx(G).margin(0.05));
        }
    }
}

TEST_CASE("Evaluation of different J^{(n)}", "[checkJvals]")
{
    double Tstar = 2.0, rhostar = 0.4;
    SECTION("Luckas v. G&T v. Gottschalk"){
        for (auto n = 4; n < 16; ++n){
            CAPTURE(n);
            auto L = LuckasJIntegral(n).get_J(Tstar, rhostar);
            auto G = GubbinsTwuJIntegral(n).get_J(Tstar, rhostar);
            auto Go = GottschalkJIntegral(n).get_J(Tstar, rhostar);
            CHECK(L == Approx(G).margin(0.05));
            CHECK(Go == Approx(G).margin(0.05));
        }
    }
}

TEST_CASE("Evaluation of K^{(n,m)}", "[checkKvals]")
{
    double Tstar = 1.095, rhostar = 0.88;
    SECTION("Luckas v. G&T v. Gottschalk"){
        std::vector<std::tuple<int, int>> nm = {{222,333},{233,344},{334,445},{444,555}};
        for (auto [n,m] : nm){
            auto L = LuckasKIntegral(n,m).get_K(Tstar, rhostar);
            auto G = GubbinsTwuKIntegral(n,m).get_K(Tstar, rhostar);
            auto Go = GottschalkKIntegral(n,m).get_K(Tstar, rhostar);
            CAPTURE(n);
            CAPTURE(m);
            CHECK(L == Approx(G).margin(std::abs(L)/2));
            CHECK(Go == Approx(G).margin(std::abs(L)/2));
        }
    }
}


using my_float_type = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100U>>;

TEST_CASE("Evaluate higher derivatives of K", "[GTK]")
{
    std::vector<std::tuple<int, int>> nm = {{222,333},{233,344},{334,445},{444,555}};
    for (auto [n_, m_] : nm){
        const int n = n_, m = m_;
        double Tstar = 1.65;
        auto frho = [&](const auto& rho_) { return LuckasKIntegral(n, m).get_K(Tstar, rho_); };
        
        my_float_type rho__ = 0.32, h = 1e-20;
        auto f1 = static_cast<double>(teqp::centered_diff<1, 4>(frho, rho__, h));
        auto f2 = static_cast<double>(teqp::centered_diff<2, 4>(frho, rho__, h));
        auto f3 = static_cast<double>(teqp::centered_diff<3, 4>(frho, rho__, h));
        auto f4 = static_cast<double>(teqp::centered_diff<4, 4>(frho, rho__, h));
        
        autodiff::Real<6, double> rho_ = 0.32;
        auto ad = derivatives(frho, along(1), at(rho_));
        
        CAPTURE(n);
        CAPTURE(m);
        CAPTURE(f4);
        CHECK(f1 == Approx(ad[1]));
        CHECK(f2 == Approx(ad[2]));
        CHECK(f3 == Approx(ad[3]));
        CHECK(f4 == Approx(ad[4]));
    }
}


using MCGTL = MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>;
using MCGG = MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>;

TEST_CASE("Evaluation of Gubbins and Twu combos ", "[GTLPolar]")
{
    auto sigma_m = (Eigen::ArrayXd(2) << 3e-10, 3.1e-10).finished();
    auto epsilon_over_k= (Eigen::ArrayXd(2) << 200, 300).finished();
    auto mubar2 = (Eigen::ArrayXd(2) << 0.0, 0.5).finished();
    auto Qbar2 = (Eigen::ArrayXd(2) << 0.5, 0).finished();
    SECTION("+ Luckas"){
        MCGTL GTL{sigma_m, epsilon_over_k, mubar2, Qbar2, multipolar_rhostar_approach::calculate_Gubbins_rhostar};
        auto z = (Eigen::ArrayXd(2) << 0.1, 0.9).finished();
        auto rhoN = std::complex<double>(300, 1e-100);
        GTL.eval(300.0, rhoN, rhoN, z);
    }
    SECTION("+ Gubbins&Twu"){
        MCGG GTL{sigma_m, epsilon_over_k, mubar2, Qbar2, multipolar_rhostar_approach::calculate_Gubbins_rhostar};
        auto z = (Eigen::ArrayXd(2) << 0.1, 0.9).finished();
        auto rhoN = std::complex<double>(300, 1e-100);
        GTL.eval(300.0, rhoN, rhoN, z);
    }
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
        CHECK(safe_abs1(std::complex<double>(3.1, h)).imag()/h == Approx(1));
        CHECK(safe_abs1(std::complex<double>(-3.1, h)).imag()/h == Approx(-1));
    }
    auto safe_abs2 = [](const auto&x) { return (getbaseval(x) < 0) ? -x : x; };
    SECTION("check base"){
        CHECK(safe_abs2(std::complex<double>(3.1, h)).imag()/h == Approx(1));
        CHECK(safe_abs2(std::complex<double>(-3.1, h)).imag()/h == Approx(-1));
    }
}

TEST_CASE("Check Stockmayer critical points with polarity terms", "[SAFTVRMiepolar]"){
    double rhostar_guess_init = 0.27;
    double Tstar_guess_init = 1.5;
    double ek = 100;
    double sigma_m = 3e-10;
    auto j = nlohmann::json::parse(R"({
        "kind": "SAFT-VR-Mie",
        "model": {
            "coeffs": [
                {
                    "name": "Stockmayer126",
                    "m": 1.0,
                    "sigma_m": 3e-10,
                    "epsilon_over_k": 100,
                    "lambda_r": 12,
                    "lambda_a": 6,
                    "BibTeXKey": "me"
                }
            ]
        }
    })");
    CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    auto nonpolar_model = teqp::cppinterface::make_model(j);
    
    //    SECTION("Backwards-compat Gross&Vrabec"){
    //        const bool print = false;
    //        double Tstar_guess = Tstar_guess_init, rhostar_guess = rhostar_guess_init;
    //        if (print) std::cout << "(mu^*)^2, T^*, rho^*" << std::endl;
    //
    //        auto critpure = nonpolar_model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
    //        Tstar_guess = std::get<0>(critpure)/ek;
    //        rhostar_guess = std::get<1>(critpure)*N_A*pow(sigma_m, 3);
    //        if (print) std::cout << "0, " << Tstar_guess << ", " << rhostar_guess << std::endl;
    //
    //        for (double mustar2 = 0.1; mustar2 < 5; mustar2 += 0.1){
    //            j["model"]["coeffs"][0]["(mu^*)^2"] = mustar2;
    //            j["model"]["coeffs"][0]["nmu"] = 1;
    //            auto model = teqp::cppinterface::make_model(j);
    //            auto pure = model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
    //            if (print) std::cout << mustar2 << ", " << std::get<0>(pure)/ek << ", " << std::get<1>(pure)*N_A*pow(sigma_m, 3) << std::endl;
    //            Tstar_guess = std::get<0>(pure)/ek;
    //            rhostar_guess = std::get<1>(pure)*N_A*pow(sigma_m, 3);
    //        }
    //        CHECK(Tstar_guess == Approx(2.29743));
    //        CHECK(rhostar_guess == Approx(0.221054));
    //    }
    
    SECTION("With multipolar terms"){
        for (std::string polar_model : {"GubbinsTwu+Luckas", "GubbinsTwu+GubbinsTwu", "GrossVrabec"}){
            const bool print = false;
            double Tstar_guess = Tstar_guess_init, rhostar_guess = rhostar_guess_init;
            if (print) std::cout << "===== " << polar_model << " =====" << std::endl;
            if (print) std::cout << "(mu^*)^2, T^*, rho^*" << std::endl;
            
            auto critpure = nonpolar_model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
            Tstar_guess = std::get<0>(critpure)/ek;
            rhostar_guess = std::get<1>(critpure)*N_A*pow(sigma_m, 3);
            if (print) std::cout << "0, " << Tstar_guess << ", " << rhostar_guess << std::endl;
            
            double mustar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23);
//            double Qstar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23);
            j["model"]["polar_model"] = polar_model;
            
            for (double mustar2 = 0.1; mustar2 < 5; mustar2 += 0.1){
                j["model"]["coeffs"][0]["mu_Cm"] = sqrt(mustar2/mustar2factor*(ek*pow(sigma_m, 3)));
                j["model"]["coeffs"][0]["nmu"] = 1;
                auto model = teqp::cppinterface::make_model(j);
                auto pure = model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
                if (print) std::cout << mustar2 << ", " << std::get<0>(pure)/ek << ", " << std::get<1>(pure)*N_A*pow(sigma_m, 3) << std::endl;
                Tstar_guess = std::get<0>(pure)/ek;
                rhostar_guess = std::get<1>(pure)*N_A*pow(sigma_m, 3);
            }
            //            CHECK(Tstar_guess == Approx(2.29743));
            //            CHECK(rhostar_guess == Approx(0.221054));
        }
    }
    
}

TEST_CASE("Test Paricaud model w/ Gray and Gubbins code", "[Paricaud]"){
    std::string contents = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GrayGubbins+GubbinsTwu"}})";
    auto model = teqp::cppinterface::make_model(nlohmann::json::parse(contents));
    auto Tcrhoc = model->solve_pure_critical(300, 5000);
    std::cout << std::setprecision(20) << std::get<0>(Tcrhoc) << std::endl;

}

TEST_CASE("Benchmark CO2 with polar PC-SAFT model", "[CO2bench]"){
    auto contents = R"(
    { 
      "kind": "PCSAFT",
      "model": {
        "coeffs": [{
            "name": "CO2",
             "BibTeXKey": "Gross-AICHEJ",
             "m": 1.5131,
             "sigma_Angstrom": 3.1869,
             "epsilon_over_k": 169.33,
             "(Q^*)^2": 1.26,
             "nQ": 1
        }]
      }
    }
    )"_json;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
}

TEST_CASE("Benchmark methane with PC-SAFT model", "[methanebench]"){
    auto contents = R"(
    {
      "kind": "PCSAFT",
      "model": {
        "coeffs": [{
                "name": "METHANE",
                 "BibTeXKey": "Gross-AICHEJ",
                 "m": 1.0,
                 "sigma_Angstrom": 3.7039,
                 "epsilon_over_k": 150.03
            }]
      }
    }
    )"_json;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
}

TEST_CASE("Benchmark methane with pure PC-SAFT model", "[methanebench]"){
    auto contents = R"(
    {
      "kind": "PCSAFTPureGrossSadowski2001",
      "model": {"m": 1.593, "sigma / A": 3.445, "epsilon_over_k": 176.47}
    }
    )"_json;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
}


TEST_CASE("Benchmark CO2+methane with polar PC-SAFT model", "[CO2bench]"){
    auto contents = R"(
    {
      "kind": "PCSAFT",
      "model": {
        "coeffs": [{
            "name": "CO2",
             "BibTeXKey": "Gross-AICHEJ",
             "m": 1.5131,
             "sigma_Angstrom": 3.1869,
             "epsilon_over_k": 169.33,
             "(Q^*)^2": 1.26,
             "nQ": 1
            },
            {
                "name": "METHANE",
                 "BibTeXKey": "Gross-AICHEJ",
                 "m": 1.0,
                 "sigma_Angstrom": 3.7039,
                 "epsilon_over_k": 150.03,
                 "(Q^*)^2": 0.0,
                 "nQ": 1
            }]
      }
    }
    )"_json;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(2) << 0.15,0.85).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
}
    

TEST_CASE("Test Paricaud model", "[Paricaud]"){
    std::string contents = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GubbinsTwu+GubbinsTwu"}})";
    auto model = teqp::cppinterface::make_model(nlohmann::json::parse(contents));
    auto Tcrhoc = model->solve_pure_critical(300, 5000);
    std::cout << std::setprecision(20) << std::get<0>(Tcrhoc) << std::endl;
}

TEST_CASE("Test Paricaud model w/ mixtures", "[Paricaud]"){
    std::string contents1 = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}, {"name": "?", "BibTeXKey": "Paricaud", "m": 1.4656, "sigma_Angstrom": 4.7307, "epsilon_over_k": 289.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GubbinsTwu+GubbinsTwu"}})";
    std::string contents2 = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}, {"name": "?", "BibTeXKey": "Paricaud", "m": 1.4656, "sigma_Angstrom": 4.7307, "epsilon_over_k": 289.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GrayGubbins+GubbinsTwu"}})";
    using _t = decltype(teqp::SAFTVRMie::SAFTVRMiefactory({}));
    auto z = (Eigen::ArrayXd(2) << 0.7, 0.3).finished();
    for (auto contents : {contents1, contents2}){
        auto model = teqp::cppinterface::make_model(nlohmann::json::parse(contents));
        const auto& m = teqp::cppinterface::adapter::get_model_cref<_t>(model.get());
        std::cout << std::setprecision(18) << m.alphar(300.0, 5000.0, z) << std::endl;
    }
}


TEST_CASE("Test epsilon_ij approaches w/ Gray and Gubbins code", "[epsilon_ij]"){
    SECTION("invalid epsilon_ij"){
        std::string contents = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GrayGubbins+GubbinsTwu", "SAFTVRMie_flags": {"epsilon_ij": "??"} }})";
        CHECK_THROWS(teqp::cppinterface::make_model(nlohmann::json::parse(contents)));
    }
    SECTION("valid epsilon_ij"){
        std::string contents = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"name": "R1234YF", "BibTeXKey": "Paricaud", "m": 1.3656, "sigma_Angstrom": 4.5307, "epsilon_over_k": 299.424, "lambda_r": 21.7779, "lambda_a": 6.0, "mu_D": 2.2814, "nmu": 1.0, "Q_DA": 1.4151, "nQ": 1.0}], "polar_model": "GrayGubbins+GubbinsTwu", "SAFTVRMie_flags": {"epsilon_ij": "Lorentz-Berthelot"} }})";
        CHECK_NOTHROW(teqp::cppinterface::make_model(nlohmann::json::parse(contents)));
    }
}
TEST_CASE("Check Stockmayer critical points with polarity terms", "[SAFTVRMiepolarmuQ]"){
    double rhostar_guess_init = 0.27;
    double Tstar_guess_init = 1.5;
    double ek = 100;
    double sigma_m = 3e-10;
    auto j = nlohmann::json::parse(R"({
        "kind": "SAFT-VR-Mie",
        "model": {
            "coeffs": [
                {
                    "name": "Stockmayer126",
                    "m": 1.0,
                    "sigma_m": 3e-10,
                    "epsilon_over_k": 100,
                    "lambda_r": 12,
                    "lambda_a": 6,
                    "BibTeXKey": "me"
                }
            ]
        }
    })");
    CHECK_NOTHROW(teqp::cppinterface::make_model(j));
    auto nonpolar_model = teqp::cppinterface::make_model(j);
    
    const double mustar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23);
    const double Qstar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23);
    
    // Check the Vrabec&Gross values
    std::valarray<std::tuple<double, double>> mu2Q22CLJ = {{6,2}, {6,4}, {12,2}, {12,4}};
    
    SECTION("With mu&Q terms"){
        const bool print = false;
        
        double Tstar_guess = Tstar_guess_init, rhostar_guess = rhostar_guess_init;
        if (print) std::cout << "(mu^*)^2, T^*, rho^*" << std::endl;
        
        auto critpure = nonpolar_model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
        
        for (std::string polar_model : {"GubbinsTwu+GubbinsTwu", "GubbinsTwu+Luckas"}){
            for (auto [mustar22CLJ, Qstar22CLJ] : mu2Q22CLJ){
                
                j["model"]["polar_model"] = polar_model;
                
                double Qstar2 = Qstar22CLJ/4.0;
                double mustar2 = mustar22CLJ/4.0;
                
                j["model"]["coeffs"][0]["mu_Cm"] = sqrt(mustar2/mustar2factor*(ek*pow(sigma_m, 3)));
                j["model"]["coeffs"][0]["Q_Cm2"] = sqrt(Qstar2/Qstar2factor*(ek*pow(sigma_m, 5)));
                j["model"]["coeffs"][0]["nmu"] = 1;
                j["model"]["coeffs"][0]["nQ"] = 1;
                
                auto model = teqp::cppinterface::make_model(j);
                auto pure = model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)));
                if (print) std::cout << mustar2 << ", " << std::get<0>(pure)/ek*4.0 << ", " << std::get<1>(pure)*N_A*pow(sigma_m, 3) << std::endl;
            }
//            CHECK(Tstar_guess == Approx(2.29743));
//            CHECK(rhostar_guess == Approx(0.221054));
        }
    }
}

TEST_CASE("Critical points with Stockmayer-LJ model of Mooji", "[polar]"){
    auto j = R"({"kind": "SAFT-VR-Mie", "model": {"coeffs": [{"m": 1.0, "lambda_r": 12, "lambda_a": 6, "sigma_m": 3.405e-10, "epsilon_over_k": 119.8, "mu_Cm": 5.390822314007849e-30, "Q_Cm2": 0.0, "name": "Stockmayer", "BibTeXKey": "?", "nQ": 1.0, "nmu": 1.0}, {"m": 1.0, "lambda_r": 12, "lambda_a": 6, "sigma_m": 3.405e-10, "epsilon_over_k": 119.8, "mu_Cm": 0.0, "Q_Cm2": 0.0, "name": "LJ", "BibTeXKey": "?", "nQ": 1.0, "nmu": 1.0}], "polar_model": "GrayGubbins+GubbinsTwu"}} )"_json;
    auto model = teqp::cppinterface::make_model(j);
    SECTION("Lennard-Jones"){
        double Tstar_guess = 1.3, rhostar_guess = 0.3;
        double ek = 119.8, sigma_m = 3.405e-10;
        nlohmann::json flags = {{"alternative_pure_index", 1}, {"alternative_length", 2}};
        auto pure = model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)), flags);
        double Tstar = std::get<0>(pure)/ek;
        int t = 0;
    }
    SECTION("Stockmayer"){
        double Tstar_guess = 2.3, rhostar_guess = 0.3;
        double ek = 119.8, sigma_m = 3.405e-10;
        nlohmann::json flags = {{"alternative_pure_index", 0}, {"alternative_length", 2}};
        auto pure = model->solve_pure_critical(Tstar_guess*ek, rhostar_guess/(N_A*pow(sigma_m, 3)), flags);
        double Tstar = std::get<0>(pure)/ek;
        int t = 0;
    }
    
}
