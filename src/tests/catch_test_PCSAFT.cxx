#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/core.hpp"
#include "teqp/models/pcsaft.hpp"
using namespace teqp::PCSAFT;

TEST_CASE("Check PCSAFT with kij", "[PCSAFT]")
{
    std::vector<std::string> names = { "Methane", "Ethane" };
    Eigen::ArrayXXd kij_right(2, 2); kij_right.setZero();
    Eigen::ArrayXXd kij_bad(2, 20); kij_bad.setZero();

    SECTION("No kij") {
        CHECK_NOTHROW(PCSAFTMixture(names));
    }
    SECTION("Correctly shaped kij matrix") {
        CHECK_NOTHROW(PCSAFTMixture(names, kij_right));
    }
    SECTION("Incorrectly shaped kij matrix") {
        CHECK_THROWS(PCSAFTMixture(names, kij_bad));
    }
}