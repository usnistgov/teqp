#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/model_potentials/2center_ljf.hpp"
#include "teqp/derivs.hpp"

#include "nlohmann/json.hpp"
#include "teqp/cpp/teqpcpp.hpp"

using namespace teqp;
using namespace twocenterljf;

TEST_CASE("Test for pressure / internal enery for 2-Center Lennard-Jones Model (Mecke et al.)", "[2CLJF_Mecke]")
{
    // For the Lennard-Jones fluid with an L = 0.0, the temperature needs to be divided by 4
    // Test values are given in the paper available at: https://link.springer.com/article/10.1007/BF02575128
    std::valarray<double> T =   { 4.6/4.0 , 4.34 , 2.00 };
    std::valarray<double> rho = { 0.822, 0.792 , 0.458};
    std::valarray<double> L =   { 0.0  , 0.1 , 0.67 };

    // Test values for pressure
    std::valarray<double> p_eos = { 8.622  , 8.167 , 2.013 };
    
    // Test values for internal energy
    std::valarray<double> u_eos = { -22.0916 , -20.9868 , -10.9000 };

    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model_dipole("2CLJF_Mecke", L[i]);
        using tdx = TDXDerivatives<decltype(model)>;
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i]*T[i]*(1.0 + tdx::get_Ar01(model, T[i], rho[i], rhovec));
        auto u = T[i] * tdx::get_Ar10(model, T[i], rho[i], rhovec);
        if (L[i] == 0.0)
        {
            // For an elongation of L = 0.0 the model is evaulated at T*/4, so for the correct result
            // the pressure and internal energy need to be multiplied with 4
            CHECK(4.0 * p == Approx(p_eos[i]).epsilon(0.001));
            CHECK(4.0 * u == Approx(u_eos[i]).epsilon(0.001));
        }
        else
        {
            CHECK(p == Approx(p_eos[i]).epsilon(0.001));
            CHECK(u == Approx(u_eos[i]).epsilon(0.001));
        }
        
    }
   
}

TEST_CASE("Test for pressure for 2-Center Lennard-Jones Model (Lisal et al.)", "[2CLJF_Lisal]")
{

    // Test case for Lisal: Values are taken from a Fortran implementation
    std::valarray<double> T =   { 2.0 , 4.0/4.0};
    std::valarray<double> rho = { 0.2 ,  0.2};
    std::valarray<double> L =   { 1.0 ,  0.0};
    std::valarray<double> p_eos = { 0.180412522868240 , 9.946745291838006E-002 };
    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model_dipole("2CLJF_Lisal", L[i]);
        using tdx = TDXDerivatives<decltype(model)>;
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i] * T[i] * (1.0 + tdx::get_Ar01(model, T[i], rho[i], rhovec));
        if (L[i] == 0.0)
        {
            // For an elongation of L = 0.0 the model is evaulated at T*/4, so for the correct result
            // the pressure needs to be multiplied with 4
            CHECK(4.0 * p == Approx(p_eos[i]).epsilon(0.00001));
        }
        else
        {
            CHECK(p == Approx(p_eos[i]).epsilon(0.000001));
        }

    }

}

TEST_CASE("Test for pressure for 2-Center Lennard-Jones Model (Lisal et al.) plus dipole", "[2CLJF_Lisal+SaagerD]")
{
    // Original model developed by Saager et al, function taken from Kriebel and Winkelmann (https://aip.scitation.org/doi/10.1063/1.472764)
    // Test case for Lisal: Values are taken from a Fortran implementation
    std::valarray<double> T = { 4.0 , 4.0};
    std::valarray<double> rho = { 0.2 , 0.2};
    std::valarray<double> L = { 0.5 , 0.0};

    // The dipolar moment input here is the square of the dipole moment
    std::valarray<double> mu_sq = { 2.0 , 4.0*2.0}; // if the elongation equals zero put in 4 times the square of the dipole moment
    std::valarray<double> p_eos = { 0.611721649982786 , 3.40675650036849};
    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model_dipole("2CLJF_Lisal", L[i], mu_sq[i]);
        using tdx = TDXDerivatives<decltype(model)>;
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i] * T[i] * (1.0 + tdx::get_Ar01(model, T[i], rho[i], rhovec));
        if (L[i] == 0.0)
        {
            // For an elongation of L = 0.0 the model is evaulated at T*/4, so for the correct result
            // the pressure needs to be multiplied with 4
            CHECK(4.0 * p == Approx(p_eos[i]).epsilon(0.00001));
        }
        else
        {
            CHECK(p == Approx(p_eos[i]).epsilon(0.000001));
        }

    }

}

TEST_CASE("Test for pressure for 2-Center Lennard-Jones Model (Lisal et al.) plus quadrupol", "[2CLJF_Lisal+SaagerQ]")
{
    // Original model developed by Saager et al, function taken from Kriebel and Winkelmann (https://aip.scitation.org/doi/10.1063/1.472764)
    // No test values are available, thus a comparison with the data given in Saager et al. (https://www.sciencedirect.com/science/article/abs/pii/0378381292850195) is performed
    std::valarray<double> T = { 3.0780 , 3.0780 , 2.1546 , 3.0780 };
    std::valarray<double> rho = { 0.06084 , 0.06084 , 0.46644 , 0.38532 };
    std::valarray<double> L = { 0.505 , 0.505 , 0.505 , 0.505 };

    // Statistical errors of the simulation data given in Saager et al.
    std::valarray<double> eps_p = { 0.002 , 0.002 , 0.025 , 0.04 };
    std::valarray<double> eps_u = { 0.045 , 0.045 , 0.015 , 0.025 };
      

    // The polar moment input here is the square of the quadrupolar moment
    std::valarray<double> Q_sq = { 0.0 , 0.5 , 1.0 , 4.0 };
    std::valarray<double> p_sim = { 0.146 , 0.145 , 0.153 , 0.286 };
    std::valarray<double> u_sim = { -1.598 , -1.621 , -11.75 , -11.465 };
    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model_quadrupole("2CLJF_Lisal", L[i], Q_sq[i]);
        using tdx = TDXDerivatives<decltype(model)>;
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i] * T[i] * (1.0 + tdx::get_Ar01(model, T[i], rho[i], rhovec));
        auto u = T[i] * tdx::get_Ar10(model, T[i], rho[i], rhovec);
            CHECK(p == Approx(p_sim[i]).margin(eps_p[i]));
            CHECK(u == Approx(u_sim[i]).margin(eps_u[i]));

    }

}


TEST_CASE("2CLJF tests with make_model", "[2CLJF]"){
    
    auto m1 = teqp::cppinterface::make_model(R"({
      "kind": "2CLJF-Dipole",
      "model": {
          "author": "2CLJF_Lisal",
          "L^*": 0.5,
          "(mu^*)^2": 0.1
      }
    })"_json);
    auto cr1 = m1->solve_pure_critical(1.3, 0.3);

    auto m2 = teqp::cppinterface::make_model(R"({
      "kind": "2CLJF-Quadrupole",
      "model": {
          "author": "2CLJF_Lisal",
          "L^*": 0.5,
          "(Q^*)^2": 0.1
      }
    })"_json);
    auto cr2 = m2->solve_pure_critical(1.3, 0.3);
    
}
