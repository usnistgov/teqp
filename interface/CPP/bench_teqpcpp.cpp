#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqpcpp.hpp"

using namespace teqp::cppinterface;

#include "teqp/derivs.hpp"
#include "teqp/models/multifluid.hpp"

using namespace teqp;

TEST_CASE("Test C++ interface", "[C++]")
{
    auto modelnovar = teqp::build_multifluid_model({ "Methane","Ethane" }, "../mycp");
    auto model = teqp::cppinterface::make_multifluid_model({ "Methane","Ethane" }, "../mycp");
    auto z = (Eigen::ArrayXd(2) << 0.5, 0.5).finished();
    SECTION("Ar01") {
        double Ar01 = model->get_Arxy(0, 1, 300, 3, z);
    }
    SECTION("critical trace") {
        double Tc1 = modelnovar.redfunc.Tc(0);
        auto rhovec0 = (Eigen::ArrayXd(2) << 1/modelnovar.redfunc.vc(0), 0).finished();
        auto cr = model->trace_critical_arclength_binary(Tc1, rhovec0);
        std::cout << cr.dump(1) << std::endl;
    }
}

TEST_CASE("Benchmark C++ interface", "[C++]")
{
    auto model = teqp::cppinterface::make_multifluid_model({ "Methane", "Ethane"}, "../mycp");
    auto z = (Eigen::ArrayXd(2) << 0.5, 0.5).finished();
    
    auto model1novar = teqp::build_multifluid_model({ "Methane" }, "../mycp");
    auto model1 = teqp::cppinterface::make_multifluid_model({ "Methane" }, "../mycp");
    auto z1 = (Eigen::ArrayXd(1) << 1).finished();
    
    BENCHMARK("Ar01 two components") {
        return model->get_Arxy(0, 1, 300, 3, z);
    }; 
    BENCHMARK("Ar01 one component w/ Arxy (C++ interface, runtime lookup)") {
        return model->get_Arxy(0, 1, 300, 3, z1);
    }; 
    BENCHMARK("Ar01 one component w/ Ar01 directly") {
        return TDXDerivatives<decltype(model1novar)>::get_Arxy<0,1,ADBackends::autodiff>(model1novar, 300, 3, z1);
    }; 
}
