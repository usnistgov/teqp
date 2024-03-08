#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/cpp/teqpcpp.hpp"
#include "RPinterop/interop.hpp"

using namespace teqp;

#include "test_common.in"

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

TEST_CASE("Check RPinterop conversion with passing FLDFILE:: prefix for both fluids", "[RPinterop]") {
    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    nlohmann::json model = {
        {"components", {
            "FLDPATH::../doc/source/models/R152A.FLD",
            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
        {"BIP", BIP},
        {"departure", DEP},
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    auto model_ = cppinterface::make_model(j);
}

TEST_CASE("Check RPinterop conversion with passing FLDFILE:: prefix for both fluids, one invalid", "[RPinterop]") {
    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    nlohmann::json model = {
        {"components", {
            "FLDPATH::../doc/source/models/NOTAREALFLUID.FLD",
            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
        {"BIP", BIP},
        {"departure", DEP},
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    CHECK_THROWS(cppinterface::make_model(j));
}

//TEST_CASE("Check RPinterop conversion with passing FLDFILE:: prefix for one fluid and CAS# for other (to load the aliasmap)", "[RPinterop]") {
//    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
//    nlohmann::json model = {
//        {"components", {
//            "75-37-6",  // CAS# of R-152a
//            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
//        {"root", FLUIDDATAPATH},
//        {"BIP", BIP},
//        {"departure", DEP},
//    };
//    nlohmann::json j = {
//        {"kind", "multifluid"},
//        {"model", model}
//    };
//    auto model_ = cppinterface::make_model(j);
//}


