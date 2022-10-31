#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/models/multifluid.hpp"

#include "teqp/derivs.hpp"
#include "teqp/ideal_eosterms.hpp"

using namespace teqp;

TEST_CASE("multifluid derivatives", "[mf]")
{
    std::vector<std::string> names = { "Propane" };
    auto model = build_multifluid_model(names, "../mycp");

    double T = 300, rho = 2;
    Eigen::ArrayX<double> z(1); z.fill(1.0);
    
    auto json = nlohmann::json::parse(model.get_meta());
    auto jig = convert_CoolProp_idealgas(json.at("pures")[0].dump(), 0);
    nlohmann::json jigs = nlohmann::json::array(); jigs.push_back(jig);
    auto aig = teqp::IdealHelmholtz(jigs);
    
    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    
    BENCHMARK("All residual derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,AlphaWrapperOption::residual>(model, T, rho, z).derivs;
    };
    
    BENCHMARK("All ideal-gas derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,AlphaWrapperOption::idealgas>(aig, T, rho, z).derivs;
    };
}
