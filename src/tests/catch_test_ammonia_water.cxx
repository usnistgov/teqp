#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "teqp/models/ammonia_water.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"

#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/VLE.hpp"

#include "teqp/finite_derivs.hpp"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

using namespace teqp;

#include "test_common.in"

TEST_CASE("Trace critical curve w/ Tillner-Roth", "[NH3H2O]") {
    auto model = AmmoniaWaterTillnerRoth();
    auto z = (Eigen::ArrayXd(2) <<  0.7, 0.3).finished();
    auto Ar01 = teqp::TDXDerivatives<decltype(model)>::get_Ar01(model, 300, 300, z);

    double T0 = 405.40;
    auto rhovec0 = (Eigen::ArrayXd(2) << 225/0.01703026, 0).finished();
    auto prc0 = IsochoricDerivatives<decltype(model)>::get_pr(model, T0, rhovec0);
    auto pig0 = rhovec0.sum() * model.R(rhovec0/rhovec0.sum())*T0;
    REQUIRE(prc0 + pig0 == Approx(11.33e6).margin(0.01e6));

    SECTION("simple Euler") {
        TCABOptions opt; opt.polish = true; opt.integration_order = 1; opt.init_dt = 100; opt.verbosity = 100;
        CriticalTracing<decltype(model)>::trace_critical_arclength_binary(model, T0, rhovec0, "", opt);
    }
    SECTION("adaptive RK45") {
        TCABOptions opt; opt.polish = true; opt.integration_order = 5; opt.init_dt = 100; opt.verbosity = 100; opt.polish_reltol_T = 10000; opt.polish_reltol_rho = 100000;
        CriticalTracing<decltype(model)>::trace_critical_arclength_binary(model, T0, rhovec0, "", opt);
    }
}

TEST_CASE("Bell et al. REFPROP 10", "[NH3H2O]") {
    auto model = build_multifluid_model({ "AMMONIA", "WATER" }, FLUIDDATAPATH, "", {{ "estimate","Lorentz-Berthelot" }});

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

    TCABOptions opt; opt.polish = true; opt.integration_order = 5; opt.init_dt = 100; opt.verbosity = 1000; opt.polish_reltol_T = 10000; opt.polish_reltol_rho = 100000;
    opt.pure_endpoint_polish = false; // Doesn't work for pure water
    CriticalTracing<decltype(mutant)>::trace_critical_arclength_binary(mutant, T0, rhovec0, "", opt);
}

TEST_CASE("pure water VLE should not crash for Tillner-Roth model","[NH3H2O]") {
    auto pure = build_multifluid_model({ "Water" }, FLUIDDATAPATH);
    auto jancillaries = nlohmann::json::parse(pure.get_meta()).at("pures")[0].at("ANCILLARIES");
    auto anc = teqp::MultiFluidVLEAncillaries(jancillaries);
    double T = 500;
    auto rhoLV = pure_VLE_T(pure, T, anc.rhoL(T), anc.rhoV(T), 10);
    auto rhoL = rhoLV[0], rhoV = rhoLV[1];

    auto model = AmmoniaWaterTillnerRoth();
    
    int k = 1;
    auto rhovecL = (Eigen::ArrayXd(2) << 0.0, 0).finished(); rhovecL(k) = rhoL;
    auto rhovecV = (Eigen::ArrayXd(2) << 0.0, 0).finished(); rhovecV(k) = rhoV;
    
    auto z = (rhovecL/rhovecL.sum()).eval();
    //auto alphar = model.alphar(T, rhovecL.sum(), z);
    //auto psir = IsochoricDerivatives<decltype(model)>::get_Psir(model, T, rhovecL);
    //auto [PsirL, PsirgradL, hessianL] = IsochoricDerivatives<decltype(model)>::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
    //auto [code, rhovecLnew, rhovecVnew] = mix_VLE_Tx(model, T, rhovecL, rhovecV, z, 1e-10, 1e-10, 1e-10, 1e-10, 10);
    CHECK_THROWS(mix_VLE_Tx(model, T, rhovecL, rhovecV, z, 1e-10, 1e-10, 1e-10, 1e-10, 10));
}

TEST_CASE("pure water derivatives ", "[NH3H2O]") {
    auto f = [](const auto& x) { return forceeval(x*(1 - pow(x, 1.1))); }; 
    
    autodiff::Real<6, double> x_ = 0.0;
    auto ders = derivatives(f, along(1), at(x_));
    CHECK(ders[0] == 0);
    CHECK(ders[1] == 1);

    using adtype = autodiff::HigherOrderDual<5, double>;
    adtype x__ = 0.0;
    auto dersdu = derivatives(f, wrt(x__,x__,x__,x__), at(x__));
    //CHECK(dersdu[0] = 0); // Bug in autodiff
    //CHECK(dersdu[1] = 1);
}