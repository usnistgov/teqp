#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/algorithms/VLLE.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"

using namespace teqp;

#include "test_common.in"

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

TEST_CASE("Test VLLE for nitrogen + ethane for isotherm", "[VLLE]")
{
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    std::vector<std::string> names = {"Nitrogen", "Ethane"};
    using namespace teqp::cppinterface;
    auto model = make_multifluid_model(names, FLUIDDATAPATH);
    std::vector<decltype(model)> pures;
    pures.emplace_back(make_multifluid_model({names[0]}, FLUIDDATAPATH));
    pures.emplace_back(make_multifluid_model({names[1]}, FLUIDDATAPATH));

    double T = 120.3420;
    std::vector<nlohmann::json> traces;
    for (int ipure : {0, 1}){

        // Init at the pure fluid endpoint
        auto m0 = build_multifluid_model({names[ipure]}, FLUIDDATAPATH);
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

TEST_CASE("Test VLLE for nitrogen + ethane for isobar", "[VLLE]")
{
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    std::vector<std::string> names = {"Nitrogen", "Ethane"};
    using namespace teqp::cppinterface;
    auto model = make_multifluid_model(names, FLUIDDATAPATH);
    std::vector<decltype(model)> pures;
    pures.emplace_back(make_multifluid_model({names[0]}, FLUIDDATAPATH));
    pures.emplace_back(make_multifluid_model({names[1]}, FLUIDDATAPATH));

    double p = 29.0e5; // [Pa] # From Antolovic
    std::vector<nlohmann::json> traces;
    for (int ipure : {1, 0}){

        // Init at the pure fluid endpoint for ethane
        auto m0 = build_multifluid_model({names[ipure]}, FLUIDDATAPATH);
        auto pure0 = nlohmann::json::parse(m0.get_meta()).at("pures")[0];
        auto jancillaries = pure0.at("ANCILLARIES");
        auto anc = teqp::MultiFluidVLEAncillaries(jancillaries);

        double T0 = anc.pV.T_r*0.9;
        for (auto counter = 0; counter < 5; ++counter){
            auto r = anc.pL(T0) - p;
            auto drdT = pures[ipure]->dpsatdT_pure(T0, anc.rhoL(T0), anc.rhoV(T0));
            T0 -= r/drdT;
        }
        auto rhoLpurerhoVpure = pures[ipure]->pure_VLE_T(T0, anc.rhoL(T0), anc.rhoV(T0), 10);

        auto rhovecL = (Eigen::ArrayXd(2) << 0.0, 0.0).finished();
        auto rhovecV = (Eigen::ArrayXd(2) << 0.0, 0.0).finished();
        rhovecL[ipure] = rhoLpurerhoVpure[0];
        rhovecV[ipure] = rhoLpurerhoVpure[1];
//        PVLEOptions opt; opt.p_termination = 1e8; opt.crit_termination=1e-4; opt.calc_criticality=true;
        auto trace = model->trace_VLE_isobar_binary(p, T0, rhovecL, rhovecV);
        traces.push_back(trace);

    }
    auto VLLEsoln = VLLE::find_VLLE_p_binary(*model, traces);
    CHECK(VLLEsoln.size() == 1);
    CHECK(VLLEsoln[0].at("polished")[3].get<double>() == Approx(125.14).margin(0.1));
}

TEST_CASE("Test VLLE tracing", "[VLLE]")
{
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    std::vector<std::string> names = {"Nitrogen", "Ethane"};
    using namespace teqp::cppinterface;
    std::string root = FLUIDDATAPATH;
    auto model = make_multifluid_model(names, root);
    std::vector<decltype(model)> pures;
    pures.emplace_back(make_multifluid_model({names[0]}, root));
    pures.emplace_back(make_multifluid_model({names[1]}, root));

    double T = 118.0;
    std::vector<nlohmann::json> traces;
    for (int ipure : {0, 1}){

        // Init at the pure fluid endpoint
        auto m0 = build_multifluid_model({names[ipure]}, root);
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
    
    auto get_array = [](const nlohmann::json& j){ Eigen::ArrayXd o(j.size()); for (auto i = 0; i < o.size(); ++i){ o[i] = j[i]; } return o; };
    
    {
        // Trace to the temperature from above
        auto rhovecV = get_array(VLLEsoln[0].at("polished")[0]),
             rhovecL1 = get_array(VLLEsoln[0].at("polished")[1]),
             rhovecL2 = get_array(VLLEsoln[0].at("polished")[2]);
        double Tincrement = 0.001;
        for (; T <= 120.3420; T += Tincrement){
            auto [drhovecV, drhovecL1, drhovecL2] = VLLE::get_drhovecdT_VLLE_binary(*model, T, rhovecV, rhovecL1, rhovecL2);
            rhovecV += drhovecV*Tincrement;
            rhovecL1 += drhovecL1*Tincrement;
            rhovecL2 += drhovecL2*Tincrement;
        }
        CHECK(rhovecV[0] == Approx(3669.84793));
        CHECK(rhovecL2[0] == Approx(5640.76015).margin(0.5));
        CHECK(rhovecL1[0] == Approx(19890.1584));
    }
    
    teqp::VLLE::VLLETracerOptions flags; flags.verbosity = 0; flags.init_dT = 0.01; flags.T_limit = 140;
    auto rhovecV = get_array(VLLEsoln[0].at("polished")[0]),
         rhovecL1 = get_array(VLLEsoln[0].at("polished")[1]),
         rhovecL2 = get_array(VLLEsoln[0].at("polished")[2]);
    auto trace = trace_VLLE_binary(*model, 118.0, rhovecV, rhovecL1, rhovecL2, flags);
//    std::cout << trace.dump(1) << std::endl;
}
