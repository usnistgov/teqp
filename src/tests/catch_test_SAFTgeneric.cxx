#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/models/saft/genericsaft.hpp"

using namespace teqp;

TEST_CASE("Benchmark generic PC-SAFT+Association model", "[SAFTgeneric]"){
    auto contents = R"(
    {
      "nonpolar": {
        "kind": "PCSAFT",
        "model": {
          "coeffs": [
            {
              "name": "Water",
              "BibTeXKey": "Gross-IECR-2002",
              "m": 1.0656,
              "sigma_Angstrom": 3.0007,
              "epsilon_over_k": 366.51
            }
          ]
        }
      },
      "association": {
        "kind": "canonical",
        "model": {
          "b / m^3/mol": [0.0000145],
          "betaAB": [0.0692],
          "epsAB/kB / J/mol": [2500.7],
          "molecule_sites": [["e","e","H","H"]]
        }
      }
    }
    )"_json;
    
    auto contents_PCSAFT = R"(
    {
      "nonpolar": {
        "kind": "PCSAFT",
        "model": {
          "coeffs": [
            {
              "name": "Water",
              "BibTeXKey": "Gross-IECR-2002",
              "m": 1.0656,
              "sigma_Angstrom": 3.0007,
              "epsilon_over_k": 366.51
            }
          ]
        }
      }
    }
    )"_json;
    
    teqp::genericsaft::GenericSAFT saftass(contents);
    teqp::genericsaft::GenericSAFT saft(contents_PCSAFT);
    teqp::PCSAFT::PCSAFTMixture basesaft = teqp::PCSAFT::PCSAFTfactory(contents_PCSAFT["nonpolar"]["model"]);
    BENCHMARK("Parsing and construction"){
        return teqp::genericsaft::GenericSAFT(contents);
    };
//    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar w/ assoc."){
        return saftass.alphar(300.0, 10000.0, z);
    };
    BENCHMARK("alphar w/o assoc."){
        return saft.alphar(300.0, 10000.0, z);
    };
    BENCHMARK("alphar w/o assoc."){
        return basesaft.alphar(300.0, 10000.0, z);
    };
//    BENCHMARK("Ar11"){
//        return model->get_Ar11(300, 10000, z);
//    };
//    BENCHMARK("Ar02"){
//        return model->get_Ar02(300, 10000, z);
//    };
//    BENCHMARK("Ar20"){
//        return model->get_Ar20(300, 10000, z);
//    };
}

