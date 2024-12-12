#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

#include "teqp/cpp/teqpcpp.hpp"

#include "nlohmann/json.hpp"
#include <map>
#include <variant>

#include "catch_fixtures.hpp"

#include "test_common.in"

auto LKPmethane = [](){
    // methane, check values from TREND
    std::vector<double> Tc_K = {190.564};
    std::vector<double> pc_Pa = {4.5992e6};
    std::vector<double> acentric = {0.011};
    std::vector<std::vector<double>> kmat{{1.0}};
    nlohmann::json modelspec{
        {"Tcrit / K", Tc_K},
        {"pcrit / Pa", pc_Pa},
        {"acentric", acentric},
        {"R / J/mol/K", 8.3144598},
        {"kmat", kmat}
    };
    nlohmann::json spec{
        {"kind", "LKP"},
        {"model", modelspec}
    };
    return spec;
};
auto TwoLJF = [](){
    return R"({
      "kind": "2CLJF",
      "model": {
          "author": "2CLJF_Mecke",
          "L^*": 0.5
      }
    })"_json;
};
auto TwoLJFDip = [](){
    return R"({
      "kind": "2CLJF-Dipole",
      "model": {
          "author": "2CLJF_Lisal",
          "L^*": 0.5,
          "(mu^*)^2": 0.1
      }
    })"_json;
};
auto TwoLJFQuad = [](){
    return R"({
      "kind": "2CLJF-Quadrupole",
      "model": {
          "author": "2CLJF_Lisal",
          "L^*": 0.5,
          "(Q^*)^2": 0.1
      }
    })"_json;
};

auto PCSAFT_ = [](){
    return  R"({
      "kind": "PCSAFT",
      "model": {
          "names": ["Methane"]
      }
    })"_json;
};
auto SAFTVRMie_ = [](){
    return  R"({
      "kind": "SAFT-VR-Mie",
      "model": {
          "names": ["Methane"]
      }
    })"_json;
};
auto PCSAFTPure_ = [](){
    return R"(
    {"kind": "PCSAFTPureGrossSadowski2001", "model": {"m": 1.593, "sigma / A": 3.445, "epsilon_over_k": 176.47}}
    )"_json;
};
auto SoftSAFT_ = [](){
    return R"(
    {"kind": "SoftSAFT", "model": {"m": [1.593], "sigma / m": [3.445e-10], "epsilon/kB / K": [176.47]}}
    )"_json;
};
auto GERG2004_ = [](){
    return R"({"kind":"GERG2004resid", "model":{"names": ["methane"]}} )"_json;
};
auto GERG2008_ = [](){
    return R"({"kind":"GERG2008resid", "model":{"names": ["methane"]}} )"_json;
};

/// A structure defining where the virial coefficient should be evaluated
struct VirialReference{
    double T, rho;
};

/// A structure defining where the EOS should be evaluated for a single-phase point
struct SinglePhaseReference{
    double T, rho;
};

using evalpoint = std::variant<VirialReference, SinglePhaseReference>;

// Pure models
std::map<std::string, std::pair<nlohmann::json, std::vector<evalpoint>>> PureFluidTestSet = {
    {"LJ126_TholJPCRD2016",{ { {"kind", "LJ126_TholJPCRD2016"}, {"model", {}} }, {VirialReference{1.3, 1e-10} } }},
    {"LJ126_KolafaNezbeda1994",{ { {"kind", "LJ126_KolafaNezbeda1994"}, {"model", {}} }, {VirialReference{1.3, 1e-10} } }},
    {"LJ126_Johnson1993",{ { {"kind", "LJ126_Johnson1993"}, {"model", {}} }, {VirialReference{1.3, 1e-10} } }},
    {"Mie_Pohl2023",{ { {"kind", "Mie_Pohl2023"}, {"model", {{"lambda_r", 12}}} }, {VirialReference{1.3, 1e-10} } }},
    {"Mie_Chaparro2023",{ { {"kind", "Mie_Chaparro2023"}, {"model", {{"lambda_r", 12},{"lambda_a", 6}}} }, {VirialReference{1.3, 1e-10} } }},
    {"SW_EspindolaHeredia2009",{ { {"kind", "SW_EspindolaHeredia2009"}, {"model", {{"lambda", 1.3}}} }, {VirialReference{1.3, 1e-10} } }},
    {"EXP6_Kataoka1992",{ { {"kind", "EXP6_Kataoka1992"}, {"model", {{"alpha", 12}}} }, {VirialReference{1.3, 1e-10} } }},
    
    {"2CLJF",{ TwoLJF(), {VirialReference{1.3, 1e-10}} }}, 
    {"2CLJF_Dipole",{ TwoLJFDip(), {} }}, // Virials are not valid because non-integer density exponents
    {"2CLJF_Quad",{ TwoLJFQuad(), {} }}, // Virials are not valid because non-integer density exponents
    
    {"LKPMethane", {LKPmethane(), {VirialReference{200.0, 1e-10} } }},
    {"PCSAFT", {PCSAFT_(), {VirialReference{200.0, 1e-6} } }},
    {"PCSAFTPure", {PCSAFTPure_(), {VirialReference{200.0, 1e-6} } }},
    {"SoftSAFT", {SoftSAFT_(), {VirialReference{200.0, 1e-10} } }},
    {"SAFTVRMie", {SAFTVRMie_(), {VirialReference{200.0, 1e-10} } }},
    
    {"GERG2004", {GERG2004_(), {VirialReference{200.0, 1e-6} } }},
    {"GERG2008", {GERG2008_(), {VirialReference{200.0, 1e-6} } }},
};

TEST_CASE("virials", "[virials]"){
    for (const auto& [kind, specdata] : PureFluidTestSet){
        auto [spec, points] = specdata;
        auto model = teqp::cppinterface::make_model(spec);
        CAPTURE(kind);
        for (const auto& point: points){
            if (std::holds_alternative<VirialReference>(point)){
                const auto& pt = std::get<VirialReference>(point);
                Eigen::ArrayXd z(1); z = 1.0;
                VirialTestFixture fix(model, z);
                fix.test_virial(2, pt.T, pt.rho, 1e-6);
                fix.test_virial(3, pt.T, pt.rho, 1e-6);
                fix.test_virial(4, pt.T, pt.rho, 1e-6);
            }
        }
    }
}

auto GERG2004metheth_ = [](){
    return R"({"kind":"GERG2004resid", "model":{"names": ["methane", "ethane"]}} )"_json;
};
auto GERG2008metheth_ = [](){
    return R"({"kind":"GERG2008resid", "model":{"names": ["methane", "ethane"]}} )"_json;
};
auto PCSAFTmetheth_ = [](){
    return  R"({
      "kind": "PCSAFT",
      "model": {
          "names": ["Methane", "Ethane"]
      }
    })"_json;
};

auto multifluidmetheth_ = [](){
    auto j = R"({
      "kind": "multifluid",
      "model": {
          "components": ["Methane", "Ethane"]
      }
    })"_json;
    j["model"]["root"] = FLUIDDATAPATH;
    j["model"]["BIP"] = FLUIDDATAPATH+"/dev/mixtures/mixture_binary_pairs.json";
    j["model"]["departure"] = FLUIDDATAPATH+"/dev/mixtures/mixture_departure_functions.json";
    return j;
};

auto AmmoniaWaterTillerRoth_ = [](){
    return R"({ "kind": "AmmoniaWaterTillnerRoth", "model": {} })"_json;
};


// Add a little static_assert to make sure the concept is working properly
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/GERG/GERG.hpp"
static_assert(teqp::cppinterface::adapter::CallableReducingTemperature<teqp::GERG2008::GERG2008ResidualModel, Eigen::ArrayXd>);
static_assert(teqp::cppinterface::adapter::CallableReducingDensity<teqp::GERG2008::GERG2008ResidualModel, Eigen::ArrayXd>);

// Binary multi-fluid models
std::map<std::string, std::pair<nlohmann::json, std::vector<evalpoint>>> MultifluidBinaryTestSet = {
    {"GERG2004", {GERG2004metheth_(), {}}},
    {"GERG2008", {GERG2008metheth_(), {}}},
    {"multifluid", {multifluidmetheth_(), {}}},
    {"AmmoniaWater", {AmmoniaWaterTillerRoth_(), {}}},
};

TEST_CASE("reducing", "[MFreducing]"){
    Eigen::ArrayXd z(2); z(0) = 0.4; z(1) = 0.6;
    for (const auto& [kind, specdata] : MultifluidBinaryTestSet){
        auto [spec, points] = specdata;
        auto model = teqp::cppinterface::make_model(spec);
        
        CAPTURE(kind);
        CHECK_NOTHROW(model->get_reducing_density(z));
        CHECK_NOTHROW(model->get_reducing_temperature(z));
    }
    
    SECTION("PCSAFT"){ // this is test that SHOULD throw, to prove the negative
        auto model = teqp::cppinterface::make_model(PCSAFTmetheth_());
        CHECK_THROWS(model->get_reducing_density(z));
    }
}
