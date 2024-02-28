#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/cpp/teqpcpp.hpp"
#include "RPinterop/interop.hpp"

using namespace teqp;

TEST_CASE("Check RPinterop conversion", "[RPinterop]") {
    nlohmann::json model = {
        {"components", {"../doc/source/models/R152A.FLD", "../doc/source/models/NEWR1234YF.FLD"}},
        {"HMX.BNC", "../doc/source/models/HMX.BNC"},
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    auto model_ = cppinterface::make_model(j);
}

TEST_CASE("Check RPinterop conversion with passing JSON structures directly", "[RPinterop]") {
    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    auto c0 = RPinterop::FLDfile("../doc/source/models/R152A.FLD").make_json("");
    auto c1 = RPinterop::FLDfile("../doc/source/models/NEWR1234YF.FLD").make_json("");
    
    nlohmann::json model = {
        {"components", {c0, c1}},
        {"BIP", BIP},
        {"departure", DEP},
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    auto model_ = cppinterface::make_model(j);
}
