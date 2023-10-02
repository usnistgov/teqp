#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/algorithms/ancillary_builder.hpp"
#include "teqp/cpp/teqpcpp.hpp"

TEST_CASE("build ancillaries", "[ancillaries]")
{
    auto j = R"(
    {
      "kind": "SAFT-VR-Mie",
      "model": {"names": ["Propane"]}
    }
    )"_json;
    auto model = teqp::cppinterface::make_model(j);
    auto anc = teqp::ancillaries::build_ancillaries(*model, 370, 5000, 75);
}
