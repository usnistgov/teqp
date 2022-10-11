#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/algorithms/critical_pure.hpp"

#include "teqp/models/model_potentials/exp6.hpp"
#include "teqp/derivs.hpp"
#include <Eigen/Dense>

using namespace teqp;

TEST_CASE("simple evaluation of s^+ for EXP6 EOS", "[exp6]")
{
    
    auto model = exp6::Kataoka1992(16.20689655172410);
    std::valarray<double> z = {1.0};
    
    auto Tstar = 1.26503685842324, rhostar = 0.32024121151801;
    auto ar = model.alphar(Tstar, rhostar, z);
    
    using id = IsochoricDerivatives<decltype(model)>;
    auto rhovec = (Eigen::ArrayXd(1) << rhostar).finished();
    auto splus = id::get_splus(model, Tstar, rhovec);
    
    auto splus_target = 0.93797012263388;
    CHECK(splus_target == Approx(splus));
    
    auto cr = teqp::get_pure_critical_conditions_Jacobian(model, Tstar, rhostar);
    CHECK(std::isfinite(std::get<1>(cr)(0,0)));
}
