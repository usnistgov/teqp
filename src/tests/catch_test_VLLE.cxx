#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/algorithms/VLLE.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"

using namespace teqp;

TEST_CASE("Test intersection for trisectrix", "[VLLE]"){
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    Eigen::ArrayXd t = Eigen::ArrayXd::LinSpaced(300, -3, 3);
    double a = 0.5;
    Eigen::ArrayXd x = a*(t.pow(2)-3)/(t.pow(2)+1);
    Eigen::ArrayXd y = a*t*(t.pow(2)-3)/(t.pow(2)+1);
    auto intersections = teqp::VLLE::get_self_intersections(x,y);
    CHECK(intersections.size() == 1);
    
    Eigen::ArrayXd y2 = 0.1*t + 0.1;
    auto crintersections = teqp::VLLE::get_cross_intersections(x,y,t,y2);
    CHECK(crintersections.size() == 3);
}

TEST_CASE("Test VLLE for nitrogen + ethane", "[VLLE]")
{
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    std::vector<std::string> names = {"Nitrogen", "Ethane"};
    using namespace teqp::cppinterface;
    auto model = make_multifluid_model(names, "../mycp");
    std::vector<decltype(model)> pures;
    pures.emplace_back(make_multifluid_model({names[0]}, "../mycp"));
    pures.emplace_back(make_multifluid_model({names[1]}, "../mycp"));

    double T = 120.3420;
    std::vector<nlohmann::json> traces;
    for (int ipure : {0, 1}){

        // Init at the pure fluid endpoint
        auto m0 = build_multifluid_model({names[ipure]}, "../mycp");
        auto pure0 = nlohmann::json::parse(m0.get_meta()).at("pures")[0];
        auto jancillaries = pure0.at("ANCILLARIES");
        auto anc = teqp::MultiFluidVLEAncillaries(jancillaries);

        auto rhoLpurerhoVpure = pures[ipure]->pure_VLE_T(T, anc.rhoL(T), anc.rhoV(T), 10);
        auto rhovecL = (Eigen::ArrayXd(2) << 0.0, 0.0).finished();
        auto rhovecV = (Eigen::ArrayXd(2) << 0.0, 0.0).finished();
        rhovecL[ipure] = rhoLpurerhoVpure[0];
        rhovecV[ipure] = rhoLpurerhoVpure[1];
        TVLEOptions opt; opt.p_termination = 1e8; opt.crit_termination=1e-4; opt.calc_criticality=true;
        auto trace = model->trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt);
        traces.push_back(trace);

    }
    auto VLLEsoln = VLLE::find_VLLE_T_binary(*model, traces);
    CHECK(VLLEsoln.size() == 1);
    // Molar concentrations of the first component in each phase
    std::valarray<double> rho0s(3);
    int i = 0;
    for (auto phase: VLLEsoln[0].at("polished")){
        rho0s[i] = phase[0];
        i++;
    }
    CHECK(rho0s.min() == Approx(3669.84793));
    CHECK(rho0s.max() == Approx(19890.1584));
}
