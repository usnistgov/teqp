#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRelMatcher;

#include <iostream>

#include "teqp/models/mie/mie.hpp"

#include "teqp/algorithms/critical_pure.hpp"

TEST_CASE("FeANN", "[FeANN]"){
    teqp::FEANN::ChaparroJCP2023 model{12.0, 6.0};
    CHECK_THAT(model.alphar(1.4, 0.135, std::valarray<double>{}), WithinRelMatcher(-0.509239537652789, 1e-12));
    auto crit = teqp::solve_pure_critical(model, 1.4, 0.3);
    CHECK_THAT(std::get<0>(crit), WithinRelMatcher(1.3, 1e-12));
    CHECK_THAT(std::get<1>(crit), WithinRelMatcher(0.3, 1e-12));
}
