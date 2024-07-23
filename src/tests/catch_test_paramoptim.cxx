#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
using namespace teqp;

#include "teqp/json_tools.hpp"
#include "teqp/algorithms/pure_param_optimization.hpp"
#include "test_common.in"
#include "teqp/models/multifluid_ancillaries.hpp"

using namespace teqp::algorithms::pure_param_optimization;

static auto Dufal_contents = R"(
{
  "nonpolar": {
    "kind": "SAFT-VR-Mie",
    "model": {
      "coeffs": [
        {
          "name": "Water",
          "BibTeXKey": "Dufal-2015",
          "m": 1.0,
          "sigma_Angstrom": 3.0555,
          "epsilon_over_k": 418.00,
          "lambda_r": 35.823,
          "lambda_a": 6.0
        }
      ]
    }
  }
}
)"_json;

static auto Dufale_contents = R"(
{
  "nonpolar": {
    "kind": "SAFT-VR-Mie",
    "model": {
      "coeffs": [
        {
          "name": "Water",
          "BibTeXKey": "Dufal-2015",
          "m": 1.0,
          "sigma_Angstrom": 3.0555,
          "epsilon_over_k": 418.00,
          "lambda_r": 35.823,
          "lambda_a": 6.0
        }
      ]
    }
  },
  "association": {
    "kind": "canonical",
    "model": {
      "b / m^3/mol": [0.0000145],
      "beta": [0.0692],
      "Delta_rule": "CR1",
      "epsilon / J/mol": [16655.0],
      "molecule_sites": [["e","e","H","H"]],
      "options": {"radial_dist": "CS"}
    }
  }
}
)"_json;

auto gen_sat_data(){
    auto contents = R"({
    "kind": "multifluid",
    "model": {
        "components": ["R32"],
        "root": ""
    }
    })"_json;
    contents["model"]["root"] = FLUIDDATAPATH;
    auto model = teqp::cppinterface::make_model(contents);
    
    auto jancillaries = load_a_JSON_file(FLUIDDATAPATH+"/dev/fluids/R32.json").at("ANCILLARIES");
    auto anc = teqp::MultiFluidVLEAncillaries(jancillaries);
    
    std::vector<PureOptimizationContribution> contribs;
    for (auto T = 270.0; T < 310; T += 0.4){
        SatRhoLPoint pt;
        pt.T = T;
        auto rhoLrhoV = model->pure_VLE_T(T, anc.rhoL(T), anc.rhoV(T), 10);
        pt.rhoL_exp = rhoLrhoV[0];
        pt.rhoL_guess = rhoLrhoV[0];
        pt.rhoV_guess = rhoLrhoV[1];
        contribs.push_back(pt);
    }
    return contribs;
}

TEST_CASE("Benchmark param optimization", "[paramoptim]"){
    
    nlohmann::json maincontents = {
        {"kind", "genericSAFT"},
        {"model", Dufal_contents}
    };
    
    std::vector<std::variant<std::string, std::vector<std::string>>> pointers = {"/model/nonpolar/model/coeffs/0/m"};
    PureParameterOptimizer ppo(maincontents, pointers);
    for (auto pt : gen_sat_data()){
        ppo.add_one_contribution(pt);
    }
    
    std::vector<double> xx = {1.3};
    BENCHMARK("build JSON"){
        return ppo.build_JSON(xx);
    };
    BENCHMARK("model building"){
        return ppo.prepare(xx);
    };
    BENCHMARK("cost_function evaluation"){
        return ppo.cost_function(xx);
    };
    BENCHMARK("cost_function evaluation threaded"){
        return ppo.cost_function_threaded(xx, 6);
    };
    SECTION("check cost functions"){
        CHECK(ppo.cost_function_threaded(xx, 6) == ppo.cost_function(xx));
    }
}
