#include "catch/catch.hpp"

#include "teqp/models/multifluid.hpp"

using namespace teqp;

TEST_CASE("Test construction of mutant", "[mutant]")
{

	std::string coolprop_root = "../mycp";
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
    std::string root = "../mycp";
    nlohmann::json flags = { {"estimate", "Lorentz-Berthelot"} };
    auto BIPcollection =  root + "/dev/mixtures/mixture_binary_pairs.json";
    auto model = build_multifluid_model({ "R32", "R1234ZEE" }, "../mycp", BIPcollection, flags);
    std::string s0 = R"({"0": {"1": {"BIP": {"betaT": 0.850879634551532, "gammaT": 1.2416653630048216, "betaV": 0.7616480056314916, "gammaV": 0.9947751468478655, "Fij": 1.0}, "departure": {"type": "Exponential", "n": [], "t": [], "d": [], "l": []}}}})";
    nlohmann::json j = nlohmann::json::parse(s0);
    auto mutant = build_multifluid_mutant(model, j);
}

TEST_CASE("Test construction of mutant with invariant departure function", "[mutant][invariant]")
{

    std::string coolprop_root = "../mycp";
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
    CHECK_THROWS(build_multifluid_mutant_invariant(model, jbad));

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
    auto mutant = build_multifluid_mutant_invariant(model, j);

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