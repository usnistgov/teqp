#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/models/ammonia_water.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;

TEST_CASE("Simplest case", "[NH3H2O]") {
	auto model = build_NH3H2O_TillnerRoth();
	auto z = (Eigen::ArrayXd(2) <<  0.7, 0.3).finished();
	auto Ar01 = teqp::TDXDerivatives<decltype(model)>::get_Ar01(model, 300, 300, z);
}