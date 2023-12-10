#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/models/GERG/GERG.hpp"

using namespace teqp;

TEST_CASE("Load all GERG2004 models", "[GERG2004]"){
    const auto& names = GERG2004::component_names;
    REQUIRE(names.size() == 18);
    
    for (auto &name : names){
        CHECK_NOTHROW(GERG2004::get_pure_info(name));
    }
    
    for (auto &name : names){
        CHECK_NOTHROW(GERG2004::get_pure_coeffs(name));
        CHECK(GERG2004::get_pure_coeffs(name).sizes().size() == 1);
    }
    CHECK_THROWS(GERG2004::get_pure_info("NOT A FLUID"));
    
    for (auto i = 0; i < names.size(); ++i){
        for (auto j = i+1; j < names.size(); ++j){
            CHECK_NOTHROW(GERG2004::get_betasgammas(names[i], names[j]));
            CHECK_NOTHROW(GERG2004::get_betasgammas(names[j], names[i]));
            CAPTURE(i);
            CAPTURE(j);
            CHECK_NOTHROW(GERG2004::GERG2004Reducing({names[i], names[j]}));
            auto Fij = GERG2004::get_Fij(names[i], names[j]);
            
            if (Fij){
                CHECK_NOTHROW(GERG2004::get_departurecoeffs(names[i], names[j]));
                CHECK(GERG2004::get_departurecoeffs(names[i], names[j]).sizes().size() == 1);
            }
            else{
                CHECK_THROWS(GERG2004::get_departurecoeffs(names[i], names[j]));
            }
        }
    }
    CHECK_THROWS(GERG2004::get_betasgammas("NOT A FLUID","water"));
    CHECK_NOTHROW(GERG2004::GERG2004CorrespondingStatesTerm(names));
    CHECK_NOTHROW(GERG2004::GERG2004ResidualModel(names));
}
