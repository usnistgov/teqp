#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;

TEST_CASE("Test for critical point of Kolafa-Nezbeda", "[LJ126]")
{
    auto m = LJ126KolafaNezbeda1994();
    auto soln = solve_pure_critical(m, 1.3, 0.3);
    auto Tc = 1.3396, rhoc = 0.3108;
    CHECK(std::get<0>(soln) == Approx(Tc).margin(0.001));
    CHECK(std::get<1>(soln) == Approx(rhoc).margin(0.001));
}
TEST_CASE("Test for critical point of Thol", "[LJ126-Thol]")
{
    auto m = build_LJ126_TholJPCRD2016();
    auto soln = solve_pure_critical(m, 1.3, 0.3);
    auto Tc = 1.32, rhoc = 0.31;
    CHECK(std::get<0>(soln) == Approx(Tc).margin(0.01));
    CHECK(std::get<1>(soln) == Approx(rhoc).margin(0.01));
}
TEST_CASE("Test for critical point of Johnson", "[LJ126]")
{
    auto m = LJ126Johnson1993();
    auto soln = solve_pure_critical(m, 1.3, 0.3);
    auto Tc = 1.313, rhoc = 0.310;
    CHECK(std::get<0>(soln) == Approx(Tc).margin(0.001));
    CHECK(std::get<1>(soln) == Approx(rhoc).margin(0.001));
}
TEST_CASE("Test single point values for Johnson against calculated values from S. Stephan", "[LJ126]")
{
    auto m = LJ126Johnson1993();
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    auto Bn = VirialDerivatives<decltype(m)>::get_B2vir(m, 0.8, z);
    CHECK(Bn == Approx(-7.821026827));
    auto Bnmcx = VirialDerivatives<decltype(m)>::get_Bnvir<2,ADBackends::multicomplex>(m, 0.8, z)[2];
    CHECK(Bnmcx == Approx(-7.821026827));
    auto Bnad = VirialDerivatives<decltype(m)>::get_Bnvir<2,ADBackends::autodiff>(m, 0.8, z)[2];
    CHECK(Bnad == Approx(-7.821026827));
    
    auto ar = m.alphar(1.350, 0.600, z);
    CHECK(ar == Approx(-1.237403479));
}
TEST_CASE("Test single point values for K-N against calculated values from S. Stephan", "[LJ126]")
{
    auto m = LJ126KolafaNezbeda1994();
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    auto Bn = VirialDerivatives<decltype(m)>::get_B2vir(m, 0.8, z);
    CHECK(Bn == Approx(-7.821026827).margin(0.0005));
    auto Bnmcx = VirialDerivatives<decltype(m)>::get_Bnvir<2,ADBackends::multicomplex>(m, 0.8, z)[2];
    CHECK(Bnmcx == Approx(-7.821026827).margin(0.0005));
    auto Bnad = VirialDerivatives<decltype(m)>::get_Bnvir<2,ADBackends::autodiff>(m, 0.8, z)[2];
    CHECK(Bnad == Approx(-7.821026827).margin(0.0005));
}
