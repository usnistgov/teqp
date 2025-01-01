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
    std::vector<std::string> names = { "n-Propane" };
    auto model = build_multifluid_model(names, "../teqp/fluiddata");

    double T = 400, rho = 5000;
    Eigen::ArrayX<double> z(1); z.fill(1.0);
    
    auto json = nlohmann::json::parse(model.get_meta());
    auto jig = convert_CoolProp_idealgas(json.at("pures")[0].dump(), 0);
    nlohmann::json jigs = nlohmann::json::array(); jigs.push_back(jig);
    auto aig = teqp::IdealHelmholtz(jigs);
    
    using namespace teqp::cppinterface;
    const std::shared_ptr<AbstractModel> amm = teqp::cppinterface::make_multifluid_model(names, "../teqp/fluiddata");
    const std::shared_ptr<AbstractModel> aigg = teqp::cppinterface::make_model({{"kind","IdealHelmholtz"}, {"model", jigs}});
    
    std::vector<char> vars = {'P', 'S'};
    const auto vals = (Eigen::ArrayXd(2) << 6646000.0, 99).finished();
    
    Eigen::Ref<const Eigen::ArrayXd> rvals = vals, rz = z;
    double R = amm->R(rz);
    teqp::iteration::AlphaModel alpha{aigg, amm};
    const std::tuple<bool, bool> &relative_error = {true, false};
    std::vector<std::shared_ptr<teqp::iteration::StoppingCondition>> stopping_conditions;
    stopping_conditions.emplace_back(std::make_shared<teqp::iteration::MaxAbsErrorCondition>(1e-16));
    stopping_conditions.emplace_back(std::make_shared<teqp::iteration::StepCountErrorCondition>(20));
    stopping_conditions.emplace_back(std::make_shared<teqp::iteration::NanXDXErrorCondition>());
    stopping_conditions.emplace_back(std::make_shared<teqp::iteration::NegativeXErrorCondition>());
    stopping_conditions.emplace_back(std::make_shared<teqp::iteration::MinRelStepsizeCondition>(1e-16));
    
    teqp::iteration::NRIterator NR(alpha, vars, rvals, T, rho, rz, relative_error, stopping_conditions);
    
    BENCHMARK("alphar") {
        return model.alphar(T, rho, z);
    };
    BENCHMARK("alpha.get_A00A10A01") {
        return alpha.get_A00A10A01(T, rho, z);
    };
    BENCHMARK("alpha.get_vals") {
        return alpha.get_vals(vars, R, T, rho, z);
    };
    BENCHMARK("Ar11") {
        return amm->get_Ar11(T, rho, z);
    };
    
    BENCHMARK("All residual derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2>(model, T, rho, z).derivs;
    };
    
    BENCHMARK("All ideal-gas derivatives needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return DerivativeHolderSquare<2>(aig, T, rho, z).derivs;
    };
    
    BENCHMARK("All residual derivatives (via AbstractModel) needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return amm->get_deriv_mat2(T, rho, z);
    };
    BENCHMARK("All ideal-gas derivatives (via AbstractModel) needed for first derivatives of h,s,u,p w.r.t. T&rho") {
        return aigg->get_deriv_mat2(T, rho, z);
    };
    
    BENCHMARK("Newton-Raphson construction") {
        return teqp::iteration::NRIterator(alpha, vars, rvals, T, rho, rz, relative_error, stopping_conditions);
    };
    BENCHMARK("Inefficient Newton-Raphson construction") {
        return teqp::iteration::NRIterator(alpha, vars, (Eigen::Array2d() << vals[0], vals[1]).finished(), T, rho, rz, relative_error, stopping_conditions);
    };
    BENCHMARK("Newton-Raphson calc_matrices") {
        return NR.calc_matrices(T, rho);
    };
    BENCHMARK("Newton-Raphson calc_step") {
        return NR.calc_step(T, rho);
    };
    BENCHMARK("Newton-Raphson take_steps(1)") {
        teqp::iteration::NRIterator NR(alpha, vars, rvals, T, rho, rz, relative_error, stopping_conditions);
        return NR.take_steps(1);
    };
    BENCHMARK("Newton-Raphson take_steps(4)") {
        teqp::iteration::NRIterator NR(alpha, vars, rvals, T, rho, rz, relative_error, stopping_conditions);
        auto steps = NR.take_steps(4);
        return steps;
    };
    BENCHMARK("Newton-Raphson take_steps(4) without stopping conditions") {
        teqp::iteration::NRIterator NR(alpha, vars, rvals, T, rho, rz, relative_error, stopping_conditions);
        auto steps = NR.take_steps(4, false);
        return steps;
    };
}

TEST_CASE("Time very low level operations", "[mf]"){
    std::unordered_map<int, double> delta_map{{1, 3.7}, {3,9.7}, {5, -3}};
    
    std::valarray<double> delta_array{3.7, 3, -6};
    
    BENCHMARK("return delta_map") {
        return delta_map[3];
    };
    BENCHMARK("return delta_array") {
        return delta_array[2];
    };
    BENCHMARK("calc powi(delta, 3)") {
        return powi(1.3, 3);
    };
}
