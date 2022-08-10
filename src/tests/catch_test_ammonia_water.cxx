#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/models/ammonia_water.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

using namespace teqp;

TEST_CASE("Trace critical curve w/ Tillner-Roth", "[NH3H2O]") {
    auto model = AmmoniaWaterTillnerRoth();
    auto z = (Eigen::ArrayXd(2) <<  0.7, 0.3).finished();
    auto Ar01 = teqp::TDXDerivatives<decltype(model)>::get_Ar01(model, 300, 300, z);

    double T0 = 405.40;
    auto rhovec0 = (Eigen::ArrayXd(2) << 225/0.01703026, 0).finished();
    auto prc0 = IsochoricDerivatives<decltype(model)>::get_pr(model, T0, rhovec0);
    auto pig0 = rhovec0.sum() * model.R(rhovec0/rhovec0.sum())*T0;
    REQUIRE(prc0 + pig0 == Approx(11.33e6).margin(0.01e6));

    TCABOptions opt; opt.polish = true; opt.integration_order = 1; opt.init_dt = 100; 
    opt.pure_endpoint_polish = false; // Doesn't work for pure water
    CriticalTracing<decltype(model)>::trace_critical_arclength_binary(model, T0, rhovec0, "TillnerRoth_crit.csv", opt);
}

TEST_CASE("Bell et al. REFPROP 10", "[NH3H2O]") {
    auto model = build_multifluid_model({ "AMMONIA", "WATER" }, "../mycp", "", {{ "estimate","Lorentz-Berthelot" }});

    std::string s = R"({
        "0": {
            "1": {
                "BIP": {
                    "betaT": 0.933585,
                    "gammaT": 1.015826,
                    "betaV": 1.044759,
                    "gammaV": 1.189754,
                    "Fij": 1.0
                },
                "departure":{
                    "type": "Gaussian+Exponential",
                    "n": [-2.00211,3.0813,-1.75352,2.9816,-3.82588,-1.7385,0.42008],
                    "t": [0.25,2.0,0.5,2.0,1.0,4.0,1.0],
                    "d": [1.0,1.0,1.0,1.0,1.0,1.0,3.0],
                    "l": [2.0,1.0,0.0,0.0,0.0,0.0,0.0],
                    "eta": [0.0,0.0,0.0,0.746,4.25,0.7,3.0],
                    "beta": [0.0,0.0,0.27,0.86,3.0,0.5,4.0],
                    "gamma": [0.0,0.0,2.8,1.8,1.5,0.8,1.3],
                    "epsilon": [0.0,0.0,0.0,2.0,-0.25,1.85,0.3],
                    "Npower": 2
                }
            }
        }
    })";
    auto mutant = build_multifluid_mutant(model, nlohmann::json::parse(s));
    
    double T0 = model.redfunc.Tc[0];
    auto rhovec0 = (Eigen::ArrayXd(2) << 1/model.redfunc.vc[0], 0).finished();
    auto der0 = CriticalTracing<decltype(mutant)>::get_drhovec_dT_crit(mutant, T0, rhovec0);
    
    auto prc0 = IsochoricDerivatives<decltype(mutant)>::get_pr(mutant, T0, rhovec0);
    auto pig0 = rhovec0.sum() * mutant.R(rhovec0 / rhovec0.sum()) * T0;

    TCABOptions opt; opt.polish = true; opt.integration_order = 1; opt.init_dt = 100; opt.verbosity = 1000;
    opt.pure_endpoint_polish = false; // Doesn't work for pure water
    CriticalTracing<decltype(mutant)>::trace_critical_arclength_binary(mutant, T0, rhovec0, "BellREFPROP10_NH3.csv", opt);
}