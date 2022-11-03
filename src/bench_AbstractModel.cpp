#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/derivs.hpp"

using namespace teqp;

TEST_CASE("multifluid derivatives", "[mf]")
{
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", {
            {"components", {"../mycp/dev/fluids/Methane.json"}},
            {"BIP", "../mycp/dev/mixtures/mixture_binary_pairs.json"},
            {"departure", "../mycp/dev/mixtures/mixture_departure_functions.json"}
        }
    }};
    //std::cout << j.dump(2);
    auto am = teqp::cppinterface::make_model(j);
    auto am2 = teqp::cppinterface::make_vdW1(2, 3);

    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto rhovec = 300.0* z;
    
    BENCHMARK("alphar") {
        return am->get_Arxy(0, 0, 300, 3.0, z);
    };
    BENCHMARK("Ar20") {
        return am->get_Ar20(300, 3.0, z);
    };
    BENCHMARK("get_Ar02n") {
        return am->get_Ar02n(300, 3.0, z);
    };
    BENCHMARK("fugacity coefficients") {
        return am->get_fugacity_coefficients(300.0, rhovec);
    };
    BENCHMARK("cvr/R") {
        return -1*am->get_Arxy(2, 0, 300, 3.0, z);
    };
    BENCHMARK("partial_molar_volumes") {
        return am->get_partial_molar_volumes(300.0, rhovec);
    };
    BENCHMARK("get_deriv_mat2") {
        return am->get_deriv_mat2(300.0, 3.0, z);
    };
    BENCHMARK("build_iteration_Jv") {
        auto mat = am->get_deriv_mat2(300.0, 3.0, z);
        auto mat2 = am->get_deriv_mat2(300.0, 3.0, z);
        const std::vector<char> vars = {'T','D','P','S'};
        return teqp::cppinterface::build_iteration_Jv(vars, mat, mat2, 8.3144, 300.0, 300.0, z);
    };
}
