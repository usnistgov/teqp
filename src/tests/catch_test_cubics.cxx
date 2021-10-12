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

    double T = 800, rho = 5000;
    auto molefrac = (Eigen::ArrayXd(3) << 0.5, 0.3, 0.2).finished();
    
    auto Ar02SRK = TDXDerivatives<decltype(modelSRK)>::get_Ar02(modelSRK, T, rho, molefrac);
    auto Ar01PR = TDXDerivatives<decltype(modelPR)>::get_Ar01(modelPR, T, rho, molefrac);
    auto Ar02PR = TDXDerivatives<decltype(modelPR)>::get_Ar02(modelPR, T, rho, molefrac);
    auto Ar03PR = TDXDerivatives<decltype(modelPR)>::get_Ar0n<3>(modelPR, T, rho, molefrac)[3];
    auto Ar04PR = TDXDerivatives<decltype(modelPR)>::get_Ar0n<4>(modelPR, T, rho, molefrac)[4];
    int rr = 0;
}