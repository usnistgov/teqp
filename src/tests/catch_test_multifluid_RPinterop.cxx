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

TEST_CASE("Check RPinterop conversion with passing FLDFILE:: prefix for both fluids, and null departure function", "[RPinterop]") {
    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    BIP[2]["function"] = "";
    BIP[2]["F"] = 0.0;
    CAPTURE(BIP.dump(1));
    nlohmann::json model = {
        {"components", {
            "FLDPATH::../doc/source/models/R152A.FLD",
            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
        {"BIP", BIP}
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    CHECK_NOTHROW(cppinterface::make_model(j));
}

TEST_CASE("Check RPinterop conversion Fij instead of F in BIP", "[RPinterop]") {
    auto [BIP, DEP] = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    BIP[2]["function"] = "";
    BIP[2].erase("F");
    BIP[2]["Fij"] = 0.0;
    CAPTURE(BIP.dump(1));
    nlohmann::json model = {
        {"components", {
            "FLDPATH::../doc/source/models/R152A.FLD",
            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
        {"BIP", BIP}
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    CHECK_THROWS(cppinterface::make_model(j));
}

TEST_CASE("Check RPinterop departure present but empty", "[RPinterop]") {
    nlohmann::json BIP, DEP;
    std::tie(BIP, DEP) = RPinterop::HMXBNCfile("../doc/source/models/HMX.BNC").make_jsons();
    auto get_w_dep = [&](nlohmann::json dep){
        nlohmann::json model = {
            {"components", {
                "FLDPATH::../doc/source/models/R152A.FLD",
                "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
            {"BIP", BIP},
            {"departure", dep}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        return j;
    };
    CHECK_THROWS(cppinterface::make_model(get_w_dep(nullptr)));
    CHECK_THROWS(cppinterface::make_model(get_w_dep({})));
    CHECK_NOTHROW(cppinterface::make_model(get_w_dep(DEP)));
    CHECK_NOTHROW(cppinterface::make_model(get_w_dep(DEP.dump(1))));
}

TEST_CASE("Check RPinterop conversion with passing FLDFILE:: prefix for both fluids, specified BIP and null departure function", "[RPinterop]") {
    auto BIP = R"([{
        "hash1": "63f364b0",
        "hash2": "40377b40",
        "CAS1": "754-12-1",
        "CAS2": "7727-37-9",
        "Name1": "R152A",
        "Name2": "R1234YF",
        "function": "Methane-Nitrogen",
        "betaT": 0.99809883,
        "gammaT": 0.979273013,
        "betaV": 0.998721377,
        "gammaV": 1.013950311,
        "F": 0.0
    }])"_json;
    auto DEP = R"([{"Name": "Methane-Nitrogen", "type": "none"}])"_json;
    
    nlohmann::json model = {
        {"components", {
            "FLDPATH::../doc/source/models/R152A.FLD",
            "FLDPATH::../doc/source/models/NEWR1234YF.FLD"}},
        {"BIP", BIP},
        {"departure", DEP}
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    CHECK_NOTHROW(cppinterface::make_model(j));
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


