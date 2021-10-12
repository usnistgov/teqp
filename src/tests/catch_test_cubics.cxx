#include "catch/catch.hpp"

#include "teqp/models/cubics.hpp"
#include "teqp/derivs.hpp"

TEST_CASE("Test construction of cubic", "[cubic]")
{
    // Values taken from http://dx.doi.org/10.6028/jres.121.011
    std::valarray<double> Tc_K = { 190.564, 154.581, 150.687 },
                pc_Pa = { 4599200, 5042800, 4863000 }, 
               acentric = { 0.011, 0.022, -0.002};
    auto modelSRK = canonical_SRK(Tc_K, pc_Pa, acentric);
    auto modelPR = canonical_PR(Tc_K, pc_Pa, acentric);

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(2); molefrac = 0.5;
    
    auto Ar02SRK = TDXDerivatives<decltype(modelSRK)>::get_Ar02(modelSRK, T, rho, molefrac);
    auto Ar02PR = TDXDerivatives<decltype(modelPR)>::get_Ar02(modelPR, T, rho, molefrac);
    int rr = 0;
}