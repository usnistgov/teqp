#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "nlohmann/json.hpp"
#include "teqp/ideal_eosterms.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;

TEST_CASE("Simplest case","[alphaig]") {
	double a_1 = 1, a_2 = 2, T = 300;
	nlohmann::json j0 = nlohmann::json::array();
	j0.push_back({ {"type", "Lead"}, { "a_1", 1 }, { "a_2", 2 } });
	nlohmann::json j = nlohmann::json::array();
	j.push_back(j0);
	IdealHelmholtz ih(j);
	std::valarray<double> molefrac{1.0};
	REQUIRE(ih.alphaig(T, 1, molefrac) == log(1) + a_1 + a_2 / T);
}

TEST_CASE("alphaig derivative", "[alphaig]") {
	double a_1 = 1, a_2 = 2, T = 300, rho = 1;
	nlohmann::json j0 = nlohmann::json::array();
	j0.push_back({ {"type", "Lead"}, { "a_1", 1 }, { "a_2", 2 } }); // For the first component
	nlohmann::json j = nlohmann::json::array();
	j.push_back(j0);
	IdealHelmholtz ih(j);
	auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
	auto wih = AlphaCallWrapper<1, decltype(ih)>(ih);
	wih.alpha(T, rho, molefrac);
	using tdx = TDXDerivatives<decltype(ih), double, Eigen::ArrayXd>;
	SECTION("All cross derivatives should be zero") {
		REQUIRE(tdx::get_Agenxy<1, 1, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
		REQUIRE(tdx::get_Agenxy<1, 2, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
		REQUIRE(tdx::get_Agenxy<1, 3, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
		REQUIRE(tdx::get_Agenxy<2, 1, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
		REQUIRE(tdx::get_Agenxy<2, 2, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
		REQUIRE(tdx::get_Agenxy<2, 3, ADBackends::autodiff>(wih, T, rho, molefrac) == 0);
	}
}