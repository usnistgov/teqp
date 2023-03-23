#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/models/mie/mie.hpp"
#include "teqp/models/model_potentials/LJChain.hpp"
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
    
    auto Tc = 1.32, rhoc = 0.31;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    using tdx = TDXDerivatives<decltype(m)>;
    // Generated with ljfeos.cpp file included with paper
    double tol = 1e-100;
    CHECK(tdx::get_Arxy<0,0>(m, Tc, rhoc, z) == Approx(-0.848655028421735).margin(tol));
    CHECK(tdx::get_Arxy<1,0>(m, Tc, rhoc, z) == Approx(-1.72121486947745 ).margin(tol));
    CHECK(tdx::get_Arxy<0,1>(m, Tc, rhoc, z) == Approx(-0.682159784922984).margin(tol));
    CHECK(tdx::get_Arxy<2,0>(m, Tc, rhoc, z) == Approx(-3.06580477714764).margin(tol));
    CHECK(tdx::get_Arxy<1,1>(m, Tc, rhoc, z) == Approx(-1.43011286241234).margin(tol));
    CHECK(tdx::get_Arxy<0,2>(m, Tc, rhoc, z) == Approx(0.364319525724386).margin(tol));
    
    auto soln = solve_pure_critical(m, 1.3, 0.3);
    CHECK(std::get<0>(soln) == Approx(Tc).margin(0.2));
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



TEST_CASE("Test single point values from S. Pohl", "[Mien6]")
{
    // lambda, T, rho, A00r, A01r, A10r
    std::vector<std::tuple<double,double,double,double,double,double>> Polhpoints = { {11,0.90000000000000002,0.01,-0.06952516353249083,-0.069542431477273445,-0.11873280589797616},
        {11,1.5,0.40000000000000002,-0.89040984647608235,-0.62314635460528112,-1.9131686363749276},
        {12,0.90000000000000002,0.01,-0.063851766177926053,-0.063824241541615145,-0.11168428604885823},
        {12,1.5,0.40000000000000002,-0.78162739523488389,-0.52826518610654949,-1.8053764386984716},
        {13, 0.90000000000000002, 0.01, -0.0591627515759926, -0.059104247825106412,-0.10549495831376893},
        {13, 1.5, 0.40000000000000002, -0.69296349469424379, -0.44914074548281258, -1.717738121771077}
    };
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    for (auto [lambda_a, T, rho, A00r, A01r, A10r] : Polhpoints){
        Mie::Mie6Pohl2023 model(lambda_a);
        using TDX = TDXDerivatives<decltype(model)>;
        double Ar00rcalc = TDX::get_Arxy<0,0>(model, T, rho, z);
        double Ar10rcalc = TDX::get_Arxy<1,0>(model, T, rho, z);
        double Ar01rcalc = TDX::get_Arxy<0,1>(model, T, rho, z);
        CAPTURE(lambda_a);
        CHECK(Ar00rcalc == Approx(A00r).margin(1e-14));
        CHECK(Ar10rcalc == Approx(A10r).margin(1e-14));
        CHECK(Ar01rcalc == Approx(A01r).margin(1e-14));
    }
}

TEST_CASE("Test LJChain models", "[LJChain]"){
    auto Johnson = LJ126Johnson1993();
    auto m1 = LJChain::LJChain(std::move(Johnson), 1);
    auto m2 = LJChain::LJChain(std::move(Johnson), 2);
    auto m3 = LJChain::LJChain(std::move(Johnson), 3);
    
    auto crit1 = solve_pure_critical(m1, 1.32, 0.3);
    auto crit2 = solve_pure_critical(m2, 1.32, 0.3);
    auto crit3 = solve_pure_critical(m3, 1.32, 0.3);
    CHECK(std::get<0>(crit2) == Approx(1.82).margin(0.1));
}
