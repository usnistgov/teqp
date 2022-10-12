#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/model_potentials/2center_ljf.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;
using namespace twocenterljf;

TEST_CASE("Test for pressure of 2-Center Lennard-Jones Model (Mecke et al.)", "[2CLJF_Mecke]")
{
    // For the Lennard-Jones fluid with an L = 0.0, the temperature needs to be divided by 4
    std::valarray<double> T =   { 4.6/4.0 , 4.34 , 2.00 };
    std::valarray<double> rho = { 0.822, 0.792 , 0.458};
    std::valarray<double> L =   { 0.0  , 0.1 , 0.67 };
    std::valarray<double> p_eos = { 8.622  , 8.167 , 2.013 };
    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model("2CLJF_Mecke", L[i]);
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i]*T[i]*(1.0 + TDXDerivatives<decltype(model)>::get_Ar01(model, T[i], rho[i], rhovec));
        if (L[i] == 0.0)
        {
            // For an elongation of L = 0.0 the model is evaulated at T*/4, so for the correct result
            // the pressure needs to be multiplied with 4
            CHECK(4.0 * p == Approx(p_eos[i]).epsilon(0.001));
        }
        else
        {
            CHECK(p == Approx(p_eos[i]).epsilon(0.001));
        }
        
    }
   
}

TEST_CASE("Test for pressure of 2-Center Lennard-Jones Model (Lisal et al.)", "[2CLJF_Lisal]")
{

    // Test case for Lisal with values from Mecke publication
    // Epsilon is chosen higher, as no exactly equal values are to be expected, but the test can be used to check the general correctness of the model.
    // For the Lennard-Jones fluid with an L = 0.0, the temperature needs to be divided by 4
    std::valarray<double> T = { 4.6 / 4.0 , 4.34 , 2.00 };
    std::valarray<double> rho = { 0.822, 0.792 , 0.458 };
    std::valarray<double> L = { 0.0  , 0.1 , 0.67 };
    std::valarray<double> p_eos = { 8.622  , 8.167 , 2.013 };
    std::valarray<double> molefrac = { 1.0 };
    for (size_t i = 0; i < T.size(); i++)
    {
        const auto model = build_two_center_model("2CLJF_Lisal", L[i]);
        auto rhovec = (Eigen::ArrayXd(1) << rho[i]).finished();
        auto p = rho[i] * T[i] * (1.0 + TDXDerivatives<decltype(model)>::get_Ar01(model, T[i], rho[i], rhovec));
        if (L[i] == 0.0)
        {
            // For an elongation of L = 0.0 the model is evaulated at T*/4, so for the correct result
            // the pressure needs to be multiplied with 4
            CHECK(4.0 * p == Approx(p_eos[i]).epsilon(0.01));
        }
        else
        {
            CHECK(p == Approx(p_eos[i]).epsilon(0.01));
        }

    }

}