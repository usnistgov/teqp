#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

using namespace teqp;

#include "test_common.in"

TEST_CASE("Test construction of mutant", "[mutant]")
{

    std::string coolprop_root = FLUIDDATAPATH;
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, coolprop_root, BIPcollection);

    std::string s0 = R"({"0": {} })";
    nlohmann::json j0 = nlohmann::json::parse(s0);

    std::string s = R"({
        "0": {
            "1": {
                "BIP": {
                    "betaT": 1.1,
                    "gammaT": 0.9,
                    "betaV": 1.05,
                    "gammaV": 1.3,
                    "Fij": 1.0
                },
                "departure":{
                    "type": "none"
                }
            }
        }
    })";
    nlohmann::json j = nlohmann::json::parse(s);
    auto mutant = build_multifluid_mutant(model, j);

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(2); molefrac = 0.5;
    auto Ar02base = TDXDerivatives<decltype(model)>::get_Ar02(model, T, rho, molefrac);
    auto Ar02mut = TDXDerivatives<decltype(mutant)>::get_Ar02(mutant, T, rho, molefrac);
    CHECK(Ar02base != Ar02mut);
}

TEST_CASE("Crashing mutant construction", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection =  root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {"type": "Exponential", "n": [], "t": [], "d": [], "l": []}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    auto mutant = build_multifluid_mutant(model, j);
}

TEST_CASE("Mutant with predefined departure function", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection = root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {"type": "lookup", "name": "KWR"}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    j["0"]["1"]["departure"] = get_departure_json("KWT", root);
    auto mutant = build_multifluid_mutant(model, j);
}

TEST_CASE("Mutant with NULL departure function (error)", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection = root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {"type": "NULL"}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    CHECK_THROWS(build_multifluid_mutant(model, j));
}

TEST_CASE("Mutant with none departure function (ok)", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection = root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 1.0, "gammaT": 1.0, "betaV": 1.0, "gammaV": 1.0, "Fij": 1.0}, "departure": {"type": "none"}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    CHECK_NOTHROW(build_multifluid_mutant(model, j));
}

TEST_CASE("Test construction of mutant with invariant departure function", "[mutant][invariant]")
{

    std::string coolprop_root = FLUIDDATAPATH;
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, coolprop_root, BIPcollection);

    // Check that missing fields throw
    nlohmann::json jbad = nlohmann::json::parse(R"(
    {
        "0": {
            "1": {
                "BIP": {
                    "phiT": 1.1
                },
                "departure":{
                    "type": "none"
                }
            }
        }
    }
    )");
    CHECK_THROWS(build_multifluid_mutant(model, jbad));

    std::string s = R"({
        "0": {
            "1": {
                "BIP": {
                    "type": "invariant",
                    "phiT": 1.1,
                    "lambdaT": 0.0,
                    "phiV": 1.0,
                    "lambdaV": 0.0,
                    "Fij": 1.0
                },
                "departure":{
                    "type": "none"
                }
            }
        }
    })";
    nlohmann::json j = nlohmann::json::parse(s);
    auto mutant = build_multifluid_mutant(model, j);

    //for (auto x0 = 0.0; x0 < 1.0; x0 += 0.1) {
    //    std::vector<double> z = { x0, 1 - x0 };
    //    std::cout << x0 << " " << mutant.redfunc.get_Tr(z) << std::endl;
    //}

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(2); molefrac = 0.5;
    auto Ar02base = TDXDerivatives<decltype(model)>::get_Ar02(model, T, rho, molefrac);
    auto Ar02mut = TDXDerivatives<decltype(mutant)>::get_Ar02(mutant, T, rho, molefrac);
    CHECK(Ar02base != Ar02mut);
}

TEST_CASE("Test infinite dilution critical locus derivatives for multifluid mutant with both orders", "[crit],[multifluid],[xxx]")
{
    std::string root = FLUIDDATAPATH;

    auto pure_endpoint = [&](const std::vector < std::string>& fluids, int i) {
        const auto model = build_multifluid_model(fluids, root);

        std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 0.0}, "departure": {"type": "Exponential", "n": [], "t": [], "d": [], "l": []}}}})";
        nlohmann::json j = nlohmann::json::parse(s0);
        if (fluids[0] == "Ethane"){
            double betaT = j["0"]["1"]["BIP"]["betaT"], betaV = j["0"]["1"]["BIP"]["betaV"];
            j["0"]["1"]["BIP"]["betaT"] = 1.0/betaT;
            j["0"]["1"]["BIP"]["betaV"] = 1.0/betaV;
        }
        auto rhoc0 = 1 / model.redfunc.vc[i];
        double T0 = model.redfunc.Tc[i]; 
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0; 
        
        auto mutant = build_multifluid_mutant(model, j);
        using ct = CriticalTracing<decltype(mutant), double, Eigen::ArrayXd>;
        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(mutant, T0, rhovec0);
        auto der = ct::get_derivs(mutant, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(mutant, T0, rhovec0);
        auto z = (rhovec0 / rhovec0.sum()).eval();
        auto alphar = mutant.alphar(T0, rhoc0, z);
        return std::make_tuple(T0, rhoc0, alphar, infdil, der);
    };
    auto [T0, rho0, alphar0, infdil0, der0] = pure_endpoint({ "Nitrogen", "Ethane" }, 0);
    auto [T1, rho1, alphar1, infdil1, der1] = pure_endpoint({ "Ethane", "Nitrogen" }, 1);
    CHECK(T0 == T1);
    CHECK(rho0 == rho1);
    CHECK(alphar0 == alphar1);
    CHECK(infdil0(0) == Approx(infdil1(1)));
    CHECK(infdil0(1) == Approx(infdil1(0)));
}

TEST_CASE("Mutant with Chebyshev departure function", "[mutant]") {
    std::string root = FLUIDDATAPATH;
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection = root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, FLUIDDATAPATH, BIPcollection, flags);
    
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 1.0, "gammaT": 1.0, "betaV": 1.0, "gammaV": 1.0, "Fij": 1.0}, 
    "departure": {"type": "Chebyshev2D", "a":[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4], "taumin": 1e-10, "taumax": 5, "deltamin": 1e-6, "deltamax": 4, "Ntau":3, "Ndelta":3
    }}}})";
    auto mutant0 = build_multifluid_mutant(model, nlohmann::json::parse(s0));

    std::string s1 = R"({"0": {"1": {"BIP": {"betaT": 1.0, "gammaT": 1.0, "betaV": 1.0, "gammaV": 1.0, "Fij": 1.0}, 
    "departure": {"type": "Chebyshev2D", "a":[1,2,3,5,1,2,3,4,1,2,3,4,1,2,3,4], "taumin": 1e-10, "taumax": 5, "deltamin": 1e-6, "deltamax": 4, "Ntau":3, "Ndelta":3
    }}}})";
    auto mutant1 = build_multifluid_mutant(model, nlohmann::json::parse(s1));
    
    double T = 340, rho = 300;
    auto z = (Eigen::ArrayXd(2) << 0.4, 0.6).finished();
    using tdx = TDXDerivatives<decltype(mutant0)>;
    CHECK(tdx::get_Ar00(mutant0, T, rho, z) != tdx::get_Ar00(mutant1, T, rho, z));
}

TEST_CASE("Exponential terms in the wrong order","[mutant]"){
    std::vector<std::string> fluids = { "Methane", "Water" };
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model(fluids, root);

    nlohmann::json jnormal = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {
        "Name": "Methane-WaterHerrig",
        "BibTeX": "Herrig (2018) / see Herrig (2018) PhD thesis",
        "aliases": [],
        "n": [3.3,-2.88,9.6,-11.7,2.13,-0.53],
        "t": [1.1,0.8,0.8,1,4,3.4],
        "d": [1,1,1,1,2,4],
        "l": [0,0,1,1,1,1],
        "type": "Exponential"
    }}}})"_json;
    CHECK_NOTHROW(build_multifluid_mutant(model, jnormal));
    
    nlohmann::json jbackwards = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {
        "Name": "Methane-WaterHerrig",
        "BibTeX": "Herrig (2018) / see Herrig (2018) PhD thesis",
        "aliases": [],
        "n": [3.3,9.6,-11.7,2.13,-0.53,-2.88],
        "t": [1.1,0.8,1,4,3.4,0.8],
        "d": [1,1,1,2,4,1],
        "l": [0,1,1,1,1,0],
        "type": "Exponential"
        }}}})"_json;
    CHECK_THROWS(build_multifluid_mutant(model, jbackwards));
    
//    double T = 300, rho = 1000;
//    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
//    double alphar1 = mutant1.alphar(T, rho, z);
//    double alphar2 = mutant2.alphar(T, rho, z);
//    CHECK(alphar1 == alphar2);
}
