#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/phase_equil.hpp"
#include "teqp/ideal_eosterms.hpp"

using namespace teqp;
using namespace teqp::algorithms::phase_equil;

#include "test_common.in"

TEST_CASE("Test new VLE routines", "[VLEgen]")
{
    // As in the examples in https://doi.org/10.1021/acs.iecr.1c04703
    std::vector<std::string> names = {"Nitrogen", "Ethane"};
    using namespace teqp::cppinterface;
    std::string root = FLUIDDATAPATH;
    auto model = make_multifluid_model(names, root);
    std::vector<decltype(model)> pures;
    pures.emplace_back(make_multifluid_model({names[0]}, root));
    pures.emplace_back(make_multifluid_model({names[1]}, root));
    
    nlohmann::json jaig = nlohmann::json::array();
    for (auto name : names){
        auto jig = convert_CoolProp_idealgas(root+"/dev/fluids/"+name+".json", 0 /* index of EOS */);
        jaig.push_back(jig);
    }
    CHECK(jaig.is_array());
    
    std::cout << jaig.dump() << std::endl;
    
    // Check that converted structures can be loaded
    auto aig = make_model(nlohmann::json{{"kind", "IdealHelmholtz"}, {"model",jaig}});
    std::shared_ptr<AbstractModel> aig_shared(std::move(aig));

    double T = 118.0;
    std::vector<nlohmann::json> traces;
    for (int ipure : {0}){

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
        TVLEOptions opt; opt.p_termination = 1e8; opt.crit_termination=1e-4; opt.calc_criticality=true; opt.polish=true;
        auto trace = model->trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt);
        traces.push_back(trace);
        
        // Now check the phase equilibrium with the new solving class
        auto el = traces[0][30 ];
        auto jsonarray2Eigen = [](const nlohmann::json& j) -> Eigen::ArrayXd{ auto x = j.get<std::vector<double>>(); return Eigen::Map<Eigen::ArrayXd>(&(x[0]), x.size()); };
        
        std::cout << el.at("pL / Pa") << " Pa" << std::endl;
        std::cout << el.at("pV / Pa") << " Pa" << std::endl;
        Eigen::ArrayXd zbulk = jsonarray2Eigen(el.at("rhoL / mol/m^3")); zbulk /= zbulk.sum();
        std::vector<Eigen::ArrayXd> rhovecs = {jsonarray2Eigen(el.at("rhoL / mol/m^3")), jsonarray2Eigen(el.at("rhoV / mol/m^3"))};
        auto betas = (Eigen::ArrayXd(2) << 1.0, 0.0).finished();
        GeneralizedPhaseEquilibrium::UnpackedVariables init{T, rhovecs, betas};
        
        std::vector<std::shared_ptr<AbstractSpecification>> specs;
        
//        specs.push_back(std::make_shared<TSpecification>(init.T));
//        specs.push_back(std::make_shared<BetaSpecification>(0.99, 0));
        specs.push_back(std::make_shared<PSpecification>(el.at("pL / Pa").get<double>()/1.01));
//        specs.push_back(std::make_shared<MolarVolumeSpecification>(1/rhovecs[0].sum()));
        specs.push_back(std::make_shared<MolarEnthalpySpecification>(0));

        GeneralizedPhaseEquilibrium gpe(*model, zbulk, init, specs);
        gpe.attach_ideal_gas(aig_shared);
        
        Eigen::ArrayXd x = init.pack();
        gpe.call(x);
        std::cout << "x:" << x << std::endl;
        
        std::cout << "r:" << gpe.res.r << std::endl;
        std::cout << "Jana:" << gpe.res.J << std::endl;
        std::cout << "Jnum:" << gpe.num_Jacobian(init.pack(), init.pack()*1e-4) << std::endl;
        
        for (auto i = 0; i < 10; ++i){
            gpe.call(x);
            Eigen::ArrayXd dx = gpe.res.J.matrix().colPivHouseholderQr().solve(-gpe.res.r.matrix()).array().eval();
            x += dx;
            std::cout << "dx:" << dx << std::endl;
        }
        std::cout << "x:" << x << std::endl;
    }
}
