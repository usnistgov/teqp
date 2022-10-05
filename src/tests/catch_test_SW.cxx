#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/algorithms/critical_pure.hpp"

#include "teqp/models/squarewell.hpp"
#include "teqp/derivs.hpp"
#include <Eigen/Dense>

using namespace teqp;

TEST_CASE("simple evaluation of s^+ for square well EOS", "[squarewell]")
{
    auto model = squarewell::EspindolaHeredia2009(1.5);
    std::valarray<double> z = {1.0};
    auto Tstar=1.3144366466267958, rhostar = 0.2862336473147125;
    auto ar = model.alphar(Tstar, rhostar, z);
    
    using id = IsochoricDerivatives<decltype(model)>;
    auto rhovec = (Eigen::ArrayXd(1) << rhostar).finished();
    auto splus = id::get_splus(model, Tstar, rhovec);
    auto alphar_target = -0.8061758248466638;
    auto splus_target = 1.0031288550954747;
    CHECK(splus_target == Approx(splus));
    
    auto cr = teqp::get_pure_critical_conditions_Jacobian(model, Tstar, rhostar);
    CHECK(std::isfinite(std::get<1>(cr)(0,0)));
}
