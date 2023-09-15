#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/models/multifluid.hpp"

#include "teqp/derivs.hpp"
#include "teqp/ideal_eosterms.hpp"

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/algorithms/iteration.hpp"

using namespace teqp;

TEST_CASE("multifluid derivatives", "[mf]")
{
    std::vector<std::string> names = { "Ethane" };
    auto model = build_multifluid_model(names, "../mycp");

    double T = 300, rho = 2;
    Eigen::ArrayX<double> z(1); z.fill(1.0);
    
    auto json = nlohmann::json::parse(model.get_meta());
    auto jig = convert_CoolProp_idealgas(json.at("pures")[0].dump(), 0);
    nlohmann::json jigs = nlohmann::json::array(); jigs.push_back(jig);
    auto aig = teqp::IdealHelmholtz(jigs);
    
    using namespace teqp::cppinterface;
    std::shared_ptr<AbstractModel> amm = teqp::cppinterface::make_multifluid_model(names, "../mycp");
    std::shared_ptr<AbstractModel> aigg = teqp::cppinterface::make_model({{"kind","IdealHelmholtz"}, {"model", jigs}});
    
    std::vector<char> vars = {'P', 'S'};
    const auto vals = (Eigen::ArrayXd(2) << 300.0, 400.0).finished();
    
    Eigen::Ref<const Eigen::ArrayXd> rvals = vals, rz = z;
    teqp::iteration::NRIterator NR(amm.get(), aigg.get(), vars, rvals, T, rho, rz);
    
    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    
    BENCHMARK("All residual derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,AlphaWrapperOption::residual>(model, T, rho, z).derivs;
    };
    
    BENCHMARK("All ideal-gas derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2,AlphaWrapperOption::idealgas>(aig, T, rho, z).derivs;
    };
    
    BENCHMARK("All residual derivatives (via AbstractModel) needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return amm->get_deriv_mat2(T, rho, z);
    };
    BENCHMARK("All residual derivatives (via AbstractModel) needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return aigg->get_deriv_mat2(T, rho, z);
    };
    
    BENCHMARK("Newton-Raphson construction") {
        return teqp::iteration::NRIterator(amm.get(), aigg.get(), vars, rvals, T, rho, rz);
    };
    BENCHMARK("Newton-Raphson calc_step") {
        return NR.calc_step(T, rho);
    };
    BENCHMARK("Newton-Raphson take_step") {
        return NR.take_step();
    };
    BENCHMARK("Newton-Raphson take_steps") {
        return NR.take_steps(5);
    };
    
}
