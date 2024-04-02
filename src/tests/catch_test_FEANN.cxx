#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRelMatcher;
using Catch::Matchers::WithinRel;

#include <iostream>

#include "teqp/models/mie/mie.hpp"
#include "teqp/math/finite_derivs.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;

#include "teqp/constants.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include "teqp/derivs.hpp"

TEST_CASE("FeANN", "[FeANN]"){
    teqp::FEANN::ChaparroJCP2023 model{12.0, 6.0};
    auto z = std::valarray<double>{};
    CHECK_THAT(model.alphar(1.4, 0.135, z), WithinRelMatcher(-0.509239537652789/1.4, 1e-12));
    auto crit = teqp::solve_pure_critical(model, 1.4, 0.3);
    
    using my_float_type = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100U>>;
    double T_ = 1.3302;
    my_float_type D = 0.3039, h = 1e-30;
    auto alphar_rho = [&](const auto& x) { return model.alphar(T_, x, z); };
    
    auto zero = static_cast<double>(alphar_rho(D));
    using tdx = teqp::TDXDerivatives<decltype(model)>;
    
    auto firstderivad = tdx::get_Ar01(model, T_, 0.3039, {})/0.3039;
    auto secondderivad = tdx::get_Ar02(model, T_, 0.3039, {})/0.3039/0.3039;
    auto thirdderivad = tdx::get_Ar03(model, T_, 0.3039, {})/0.3039/0.3039/0.3039;
    
    double firstderiv = static_cast<double>(teqp::centered_diff<1, 6>(alphar_rho, D, h));
    double secondderiv = static_cast<double>(teqp::centered_diff<2, 6>(alphar_rho, D, h));
    CHECK_THAT(firstderiv, WithinRel(firstderivad, 1e-14));
    CHECK_THAT(secondderiv, WithinRel(secondderivad, 1e-14));
    
    auto first = D*firstderivad;
    auto second = D*D*secondderivad;
    auto third = D*D*D*thirdderivad;
    double R = teqp::constants::R_CODATA2017;
    auto dpdrho = static_cast<double>((1 + 2*first + second)*R*T_);
    auto d2pdrho2 = static_cast<double>((2*first + 4*second + third)/D*T_*R);
    CHECK_THAT(std::get<0>(crit), WithinRelMatcher(1.330255219, 1e-6));
    CHECK_THAT(std::get<1>(crit), WithinRelMatcher(0.30398356, 1e-6));
}
