#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/saft/correlation_integrals.hpp"

using namespace teqp::SAFTpolar;

TEST_CASE("Evaluation of J^{(n)}", "[LuckasJn]")
{
    LuckasJIntegral<12> J12;
    auto Jval = J12.get_J(3.0, 1.0);
}

TEST_CASE("Evaluation of K(xxx, yyy)", "[LuckasKnn]")
{
    auto Kval23 = LuckasKIntegral<222, 333>().get_K(1.0, 0.9);
    CHECK(Kval23 == Approx(0.03332).margin(0.02));
    
    auto Kval45 = LuckasKIntegral<444, 555>().get_K(1.0, 0.9);
    CHECK(Kval45 == Approx(0.01541).margin(0.02));
}
