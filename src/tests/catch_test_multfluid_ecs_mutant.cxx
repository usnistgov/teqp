#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/derivs.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ecs_mutant.hpp"

using namespace teqp;

#include "test_common.in"

TEST_CASE("Test construction of ecs mutant", "[ecs mutant]")
{

    std::string coolprop_root = FLUIDDATAPATH;
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    auto model = build_multifluid_model({ "CarbonDioxide", "Methane" }, coolprop_root, BIPcollection);

    std::string s0 = R"({"0": {} })";
    nlohmann::json j0 = nlohmann::json::parse(s0);

    std::string s = R"({
      "tr_coeffs": [
        [
          0.99193,
          -0.0882,
          0.03588
        ],
        [
          0.01099,
          -0.01506,
          0.0333
        ],
        [
          -0.0567,
          0.23606,
          -0.0787
        ],
        [
          -2e-05,
          -0.0135,
          0.00412
        ],
        [
          -0.021714,
          0.0172,
          -0.026087
        ],
        [
          0.03455,
          -0.03398,
          0.0148
        ]
      ],
      "dr_coeffs": [
        [
          0.95832,
          0.09096,
          -0.0863
        ],
        [
          0.008766,
          -0.048183,
          0.03912
        ],
        [
          0.03169,
          -0.25547,
          0.19474
        ],
        [
          -0.001782,
          0.01038,
          -0.00688
        ],
        [
          0.008965,
          0.00876,
          -0.01381
        ],
        [
          -0.032164,
          0.1144,
          -0.0818
        ]
      ]
    })";
    nlohmann::json j = nlohmann::json::parse(s);
    auto mutant = build_multifluid_ecs_mutant(model, j);

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(2); molefrac = 0.5;
    auto A00_test = -0.0214715237;
    auto Ar00mut = teqp::TDXDerivatives<decltype(mutant)>::get_Ar00(mutant, T, rho, molefrac);
    CHECK(A00_test != Ar00mut);
}

TEST_CASE("Test throwing error of ecs mutant when number of compositions is not 2", "[ecs mutant]")
{

    std::string coolprop_root = FLUIDDATAPATH;
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    auto model = build_multifluid_model({ "CarbonDioxide", "Methane" }, coolprop_root, BIPcollection);

    std::string s0 = R"({"0": {} })";
    nlohmann::json j0 = nlohmann::json::parse(s0);

    std::string s = R"({
      "tr_coeffs": [
        [
          0.99193,
          -0.0882,
          0.03588
        ],
        [
          0.01099,
          -0.01506,
          0.0333
        ],
        [
          -0.0567,
          0.23606,
          -0.0787
        ],
        [
          -2e-05,
          -0.0135,
          0.00412
        ],
        [
          -0.021714,
          0.0172,
          -0.026087
        ],
        [
          0.03455,
          -0.03398,
          0.0148
        ]
      ],
      "dr_coeffs": [
        [
          0.95832,
          0.09096,
          -0.0863
        ],
        [
          0.008766,
          -0.048183,
          0.03912
        ],
        [
          0.03169,
          -0.25547,
          0.19474
        ],
        [
          -0.001782,
          0.01038,
          -0.00688
        ],
        [
          0.008965,
          0.00876,
          -0.01381
        ],
        [
          -0.032164,
          0.1144,
          -0.0818
        ]
      ]
    })";
    nlohmann::json j = nlohmann::json::parse(s);
    auto mutant = build_multifluid_ecs_mutant(model, j);

    double T = 300, rho = 300;
    Eigen::ArrayXd molefrac(1); molefrac = 1.0;
    CHECK_THROWS(teqp::TDXDerivatives<decltype(mutant)>::get_Ar00(mutant, T, rho, molefrac));
}

TEST_CASE("Test throwing error when tr_coeffs are not in json", "[ecs mutant]")
{

    std::string coolprop_root = FLUIDDATAPATH;
    auto BIPcollection = coolprop_root + "/dev/mixtures/mixture_binary_pairs.json";

    auto model = build_multifluid_model({ "CarbonDioxide", "Methane" }, coolprop_root, BIPcollection);

    std::string s0 = R"({"0": {} })";
    nlohmann::json j0 = nlohmann::json::parse(s0);

    std::string s = R"({})";
    nlohmann::json j = nlohmann::json::parse(s);
    CHECK_THROWS(build_multifluid_ecs_mutant(model, j));
}