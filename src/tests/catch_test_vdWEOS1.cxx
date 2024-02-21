#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/cpp/teqpcpp.hpp"

TEST_CASE("Simplest case","[vdW1]") {
    auto model = teqp::cppinterface::make_model(R"(  {"kind": "vdW1", "model": {"a": 1, "b": 2}} )"_json);
}
