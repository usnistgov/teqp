#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

using Catch::Approx;

#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinAbsMatcher;
using Catch::Matchers::WithinRelMatcher;
using Catch::Matchers::WithinRel;

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/VLE.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/filesystem.hpp"
#include "teqp/constants.hpp"
#include "teqp/ideal_eosterms.hpp"
#include "teqp/math/finite_derivs.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;

#include "test_common.in"

using namespace teqp;
using multifluid_t = decltype(build_multifluid_model({""}, ""));

TEST_CASE("Test infinite dilution critical locus derivatives for multifluid", "[crit]")
{
    std::string root = FLUIDDATAPATH;

    const auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, root);
    using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;

    for (int i = 0; i < 2; ++i) {
        auto rhoc0 = 1/model.redfunc.vc[i];
        double T0 = model.redfunc.Tc[i];
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0;

        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(model, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(model, T0, rhovec0);

        // Just slightly not infinite dilution, values should be very similar
        Eigen::ArrayXd rhovec0almost = rhovec0; rhovec0almost[1 - i] = 1e-6;
        auto dil = ct::get_drhovec_dT_crit(model, T0, rhovec0almost);
        auto epdil = ct::eigen_problem(model, T0, rhovec0almost);

    }
}

TEST_CASE("Benchmark CO2 with Span and Wagner model", "[CO2bench]"){
    auto contents = R"(
    {
      "kind": "multifluid",
      "model": {
        "components": ["CarbonDioxide"],
        "root": "???"
      }
    }
    )"_json;
    contents["model"]["root"] = FLUIDDATAPATH;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
}

TEST_CASE("Benchmark Propane with Lemmon model", "[propanebench]"){
    auto contents = R"(
    {
      "kind": "multifluid",
      "model": {
        "components": ["n-Propane"],
        "root": "???"
      }
    }
    )"_json;
    contents["model"]["root"] = FLUIDDATAPATH;
    auto model = teqp::cppinterface::make_model(contents);
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    BENCHMARK("alphar"){
        return model->get_Ar00(300, 10000, z);
    };
    BENCHMARK("Ar11"){
        return model->get_Ar11(300, 10000, z);
    };
    BENCHMARK("Ar02"){
        return model->get_Ar02(300, 10000, z);
    };
    BENCHMARK("Ar20"){
        return model->get_Ar20(300, 10000, z);
    };
    
    using multifluid_t = decltype(multifluidfactory(std::string("")));
    const auto& rmodel = teqp::cppinterface::adapter::get_model_cref<multifluid_t>(model.get());
    
    BENCHMARK("Tr(z)"){
        return rmodel.redfunc.get_Tr(z);
    };
    BENCHMARK("alphar_taudelta(tau,delta,z)"){
        return rmodel.alphar_taudelta(0.8, 1.3, z);
    };
    BENCHMARK("alphar_taudelta0(tau,delta)"){
        return rmodel.alphar_taudeltai(0.8, 1.3, 0U);
    };
    BENCHMARK("allocate, fill, and index"){
        Eigen::ArrayXd x(20); x[0] = 1.0; for(auto i = 1; i < x.size(); ++i){ x[i] = 3.3*x[i-1]; } return x[5];
    };
    
    BENCHMARK("pow, double exponent"){
        return std::pow(3.7, 17.0);
    };
    BENCHMARK("pow, int exponent"){
        return powi(3.7, 17);
    };
}

TEST_CASE("Test infinite dilution critical locus derivatives for multifluid with both orders", "[crit]")
{
    std::string root = FLUIDDATAPATH;

    auto pure_endpoint = [&](const std::vector < std::string> &fluids, int i) {
        const auto model = build_multifluid_model(fluids, root);
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        auto rhoc0 = 1 / model.redfunc.vc[i];
        double T0 = model.redfunc.Tc[i];
        Eigen::ArrayXd rhovec0(2); rhovec0.setZero(); rhovec0[i] = rhoc0;
        // Values for infinite dilution
        auto infdil = ct::get_drhovec_dT_crit(model, T0, rhovec0);
        auto epinfdil = ct::eigen_problem(model, T0, rhovec0);
        auto der = ct::get_derivs(model, T0, rhovec0);
        auto z = (rhovec0 / rhovec0.sum()).eval();
        auto alphar = model.alphar(T0, rhoc0, z);
        return std::make_tuple(T0, rhoc0, alphar, infdil, epinfdil, der);
    };

    auto [T0, rho0, alphar0, infdil0, eig0, der0] = pure_endpoint({ "Nitrogen", "Ethane" }, 0);
    auto [T1, rho1, alphar1, infdil1, eig1, der1] = pure_endpoint({ "Ethane", "Nitrogen" }, 1);
    CHECK(T0 == T1);
    CHECK(rho0 == rho1);
    CHECK(alphar0 == alphar1);
    CHECK(infdil0(1) == Approx(infdil1(0)));
    CHECK(infdil0(0) == Approx(infdil1(1)));

    auto [Ta, rhoa, alphara, infdila, eiga, dera] = pure_endpoint({ "Ethane", "Nitrogen" }, 0);
    auto [Tb, rhob, alpharb, infdilb, eigb, derb] = pure_endpoint({ "Nitrogen", "Ethane" }, 1);
    CHECK(Ta == Tb);
    CHECK(rhoa == rhob);
    CHECK(alphara == alpharb);
    CHECK(infdila(1) == Approx(infdilb(0)));
    CHECK(infdila(0) == Approx(infdilb(1)));
}


TEST_CASE("Confirm failure for missing files","[multifluid]") {
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "IMPOSSIBLE PATH.json"));
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", FLUIDDATAPATH+"/dev/mixtures/mixture_binary_pairs.json"));
    CHECK_THROWS(build_multifluid_model({ "Ethane" }, "IMPOSSIBLE PATH"));
}

TEST_CASE("Trace critical locus for nitrogen + ethane", "[crit],[multifluid]")
{
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, root);

    for (auto ifluid = 0; ifluid < 2; ++ifluid) {
        double T0 = model.redfunc.Tc[ifluid];
        Eigen::ArrayXd rhovec0(2); rhovec0 = 0.0; rhovec0[ifluid] = 1.0 / model.redfunc.vc[ifluid];

        auto tic0 = std::chrono::steady_clock::now();
        std::string filename = "";
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        TCABOptions opt; opt.init_dt = 100; opt.integration_order = 1;
        auto j = ct::trace_critical_arclength_binary(model, T0, rhovec0, filename, opt);
        CHECK(j.size() > 3);
        auto tic1 = std::chrono::steady_clock::now();
    }
    
    for (auto ifluid = 0; ifluid < 2; ++ifluid) {
        double T0 = model.redfunc.Tc[ifluid];
        Eigen::ArrayXd rhovec0(2); rhovec0 = 0.0; rhovec0[ifluid] = 1.0 / model.redfunc.vc[ifluid];

        auto tic0 = std::chrono::steady_clock::now();
        std::string filename = "";
        using ct = CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        TCABOptions opt; opt.max_dt = 10000; opt.init_dt = 10; opt.abs_err = 1e-8; opt.rel_err = 1e-6; opt.small_T_count = 100;
        auto j = ct::trace_critical_arclength_binary(model, T0, rhovec0, filename, opt);
        CHECK(j.size() > 3);
        auto tic1 = std::chrono::steady_clock::now();
    }
}

TEST_CASE("Check that all pure fluid models can be instantiated", "[multifluid],[all]"){
    std::string root = FLUIDDATAPATH;
    SECTION("With absolute paths to json file") {
        int counter = 0;
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            if (path.filename().stem() == "Methanol") { continue; }
            CAPTURE(path.string());
            auto abspath = std::filesystem::absolute(path).string();
            auto model = build_multifluid_model({ abspath }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
            std::valarray<double> z(0.0, 1);
            model.alphar(300, 1.0, z);
            counter += 1;
        }
        CHECK(counter > 100);
    }
    SECTION("With filename stems") {
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            auto stem = path.filename().stem().string(); // filename without the .json
            if (stem == "Methanol") { continue; }
            auto model = build_multifluid_model({ stem }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
            std::valarray<double> z(0.0, 1);
            model.alphar(300, 1.0, z);
        }
    }    
}

TEST_CASE("Check that all ancillaries can be instantiated and work properly", "[multifluid],[all]") {
    std::string root = FLUIDDATAPATH;
    SECTION("With absolute paths to json file") {
        int counter = 0;
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            if (path.filename().stem() == "Methanol") { continue; }
            CAPTURE(path.string());
            auto abspath = std::filesystem::absolute(path).string();
            auto model = build_multifluid_model({ abspath }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
            auto pure0 = nlohmann::json::parse(model.get_meta()).at("pures")[0];
            // Skip pseudo-pure fluids, where ancillary checking is irrelevant
            if (pure0.at("EOS")[0].at("pseudo_pure")){
                counter += 1;
                continue;
            }
            auto jancillaries = pure0.at("ANCILLARIES");
            auto anc = teqp::MultiFluidVLEAncillaries(jancillaries);
            double T = 0.8*anc.rhoL.T_r;
            auto rhoV = anc.rhoV(T), rhoL = anc.rhoL(T);
            auto rhovec = teqp::pure_VLE_T(model, T, rhoL, rhoV, 10);
            CAPTURE(rhoL);
            CAPTURE(rhoV);
            CAPTURE(rhovec);
            CHECK_THROWS(anc.rhoV(1.1*anc.rhoL.T_r));
            CHECK_THROWS(anc.rhoL(1.1*anc.rhoL.T_r));
            
            auto rhoLerr = std::abs(rhovec[0]/rhoL-1);
            auto rhoVerr = std::abs(rhovec[1]/rhoV-1);
            CHECK(rhoLerr < 0.02);
            CHECK(rhoVerr < 0.02);
            
            counter += 1;
        }
        CHECK(counter > 100);
    }
}

TEST_CASE("Check that mixtures can also do absolute paths", "[multifluid],[abspath]") {
    std::string root = FLUIDDATAPATH;
    SECTION("With absolute paths to json file") {
        std::vector<std::filesystem::path> paths = { root + "/dev/fluids/Methane.json", root + "/dev/fluids/Ethane.json" };
        std::vector<std::string> abspaths;
        for (auto p : paths) {
            abspaths.emplace_back(std::filesystem::absolute(p).string());
        }
        auto model = build_multifluid_model(abspaths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
        auto model2 = build_multifluid_model(abspaths, root); // default path for BIP
    }
}

TEST_CASE("Check mixing absolute and relative paths and fluid names", "[multifluid],[abspath]") {
    std::string root = FLUIDDATAPATH;
    SECTION("With correct name of fluid") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Methane.json").string(), "Ethane" };
        auto model = build_multifluid_model(paths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
    SECTION("Needing a reverse lookup for one fluid") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Methane.json").string(), "PROPANE" };
        auto model = build_multifluid_model(paths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
}

TEST_CASE("Check specifying some different kinds of sources of BIP", "[multifluidBIP]") {
    std::string root = FLUIDDATAPATH;
    SECTION("Not JSON, should throw") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Nitrogen.json").string(), "Ethane" };
        CHECK_THROWS(build_multifluid_model(paths, root, "I am not a JSON formatted string"));
    }
    SECTION("The normal approach") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Nitrogen.json").string(), "Ethane" };
        auto model = build_multifluid_model(paths, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
    SECTION("Sending the contents in JSON format") {
        std::vector<std::string> paths = { std::filesystem::absolute(root + "/dev/fluids/Nitrogen.json").string(), "PROPANE" };
        auto BIP = load_a_JSON_file(root + "/dev/mixtures/mixture_binary_pairs.json");
        auto model = build_multifluid_model(paths, root, BIP.dump());
    }
}

TEST_CASE("Check that all binary pairs specified in the binary pair file can be instantiated", "[multifluid],[binaries]") {
    std::string root = FLUIDDATAPATH;
    REQUIRE_NOTHROW(build_alias_map(root));
    auto amap = build_alias_map(root);
    for (auto el : load_a_JSON_file(root + "/dev/mixtures/mixture_binary_pairs.json")) {
        auto is_unsupported = [](const auto& s) {
            return (s == "METHANOL" || s == "R1216" || s == "C14" || s == "IOCTANE" || s == "C4F10" || s == "C5F12" || s == "C1CC6" || s == "C3CC6" || s == "CHLORINE" || s == "RE347MCC");
        };
        if (is_unsupported(el["Name1"]) || is_unsupported(el["Name2"])) {
            continue;
        }
        CAPTURE(el["Name1"]);
        CAPTURE(el["Name2"]);
        CHECK_NOTHROW(build_multifluid_model({ amap[el["Name1"]], amap[el["Name2"]] }, root)); // default path for BIP
    }
}

TEST_CASE("Check that all pure fluid models can be evaluated at zero density", "[multifluid],[all],[virial]") {
    std::string root = FLUIDDATAPATH;
    SECTION("With filename stems") {
        for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
            auto stem = path.filename().stem().string(); // filename without the .json
            if (stem == "Methanol") { continue; }
            auto model = build_multifluid_model({ stem }, root);
            std::valarray<double> z(1.0, 1); 
            using tdx = TDXDerivatives<decltype(model), double, decltype(z) >;
            auto ders = tdx::template get_Ar0n<4>(model, model.redfunc.Tc[0], 0.0, z);
            CAPTURE(stem);
            CHECK(std::isfinite(ders[1]));

            using vd = VirialDerivatives<decltype(model),double, decltype(z)>;
            auto Bn = vd::get_Bnvir<4>(model, model.redfunc.Tc[0], z);

            CAPTURE(stem);
            CHECK(std::isfinite(Bn[2]));
        }
    }
}

TEST_CASE("Check that virial coefficients can be calculated with multiple derivative methods", "[multifluid],[virial]") {
    std::string root = FLUIDDATAPATH;
    std::string stem = "Argon";
    CAPTURE(stem); 
    
    auto model = build_multifluid_model({ stem }, root);
    std::valarray<double> z(1.0, 1);

    using vd = VirialDerivatives<decltype(model), double, decltype(z)>;

    auto BnAD = vd::get_Bnvir<4, ADBackends::autodiff>(model, 298.15, z);
    auto Bnmcx = vd::get_Bnvir<4, ADBackends::multicomplex>(model, 298.15, z);
    CHECK(BnAD[2] == Approx(Bnmcx[2]));
    CHECK(BnAD[3] == Approx(Bnmcx[3]));
    CHECK(BnAD[4] == Approx(Bnmcx[4]));
    
//    auto derBnAD100 = vd::get_dmBnvirdTm<2, 1, ADBackends::autodiff>(model, 100.0, z);
    auto derBnAD = vd::get_dmBnvirdTm<2, 1, ADBackends::autodiff>(model, 298.15, z);
    auto derBnMCX = vd::get_dmBnvirdTm<2, 1, ADBackends::multicomplex>(model, 298.15, z);
    CHECK(derBnAD == Approx(derBnMCX));
}

TEST_CASE("dpsat/dTsat", "[dpdTsat]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "Methane", "Ethane" }, root);
    using id = IsochoricDerivatives<decltype(model)>;
    double T = 200;
    auto rhovecL = (Eigen::ArrayXd(2) << 5431.76157173312, 12674.110334043948).finished();
    auto rhovecV = (Eigen::ArrayXd(2) << 1035.298519871195, 162.03291757734976).finished();
    
    // Concentration derivatives w.r.t. T along the isopleth
    auto [drhovecdTL, drhovecdTV] = get_drhovecdT_xsat(model, T, rhovecL, rhovecV);
    
    auto dpdT = get_dpsat_dTsat_isopleth(model, T, rhovecL, rhovecV);
    CHECK(dpdT == Approx(39348.33949198946).margin(0.01));
}

TEST_CASE("Trace a VLE isotherm for CO2 + water", "[isothermCO2water]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "CarbonDioxide", "Water" }, root);
    double T = 308.15;
    auto rhovecL = (Eigen::ArrayXd(2) << 0.0, 55174.92375117).finished();
    auto rhovecV = (Eigen::ArrayXd(2) << 0.0, 2.20225704).finished();

    auto o = trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV);
}

TEST_CASE("Trace a VLE isotherm for acetone + benzene", "[isothermacetonebenzene]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "Acetone", "Benzene" }, root);
    double T = 348.05;
    auto rhovecL = (Eigen::ArrayXd(2) << 12502.86504072, 0.0).finished();
    auto rhovecV = (Eigen::ArrayXd(2) << 69.20719534,  0.0).finished();
    auto o = trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV);
}

TEST_CASE("Calculate water at critical point", "[WATERcrit]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "Water" }, root);
    
    using tdx = TDXDerivatives<decltype(model)>;
    auto Tc = model.redfunc.Tc[0];
    auto rhoc = 1/model.redfunc.vc[0];
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto a1 = tdx::get_Ar0n<1>(model, Tc, rhoc, z);
    CHECK(std::isfinite(a1[1]));
    auto a2 = tdx::get_Ar0n<2>(model, Tc, rhoc, z);
    auto a3 = tdx::get_Ar0n<3>(model, Tc, rhoc, z);
    auto a4 = tdx::get_Ar0n<3>(model, Tc, rhoc, z);
    CHECK(a3[1] == a1[1]);
    auto R = model.R(z);
    auto dpdrho = R*Tc*(1 + 2*a4[1] + a4[2]);
    auto d2pdrho2 = R*Tc/(rhoc)*(2*a4[1] + 4*a4[2] + a4[3]);
    CHECK(dpdrho == Approx(0).margin(1e-9));
    CHECK(d2pdrho2 == Approx(0).margin(1e-9));
}

TEST_CASE("Calculate partial molar volume for a CO2 containing mixture", "[partial_molar_volume]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "CarbonDioxide", "Heptane" }, root);
    using id = IsochoricDerivatives<decltype(model), double, Eigen::ArrayXd>;
    
    double T = 343.0;
    Eigen::ArrayXd rhovec = (Eigen::ArrayXd(2) << 0.99999, 1.0-0.99999).finished();
    rhovec *= 6690.19673875373;

    std::valarray<double> expected = { 0.000149479684800994, -0.000575458122621522 };
    auto der = id::get_partial_molar_volumes(model, T, rhovec);
    for (auto i = 0U; i < expected.size(); ++i){
        CHECK(expected[i] == Approx(der[i]));
    }
}

TEST_CASE("Check that all pure fluid ideal-gas terms can be converted", "[multifluid],[all],[alphaig]") {
    std::string root = FLUIDDATAPATH;
    auto paths = get_files_in_folder(root + "/dev/fluids", ".json");
    auto p = GENERATE_REF(from_range(paths));
    CHECK(std::filesystem::is_regular_file(p));
    CAPTURE(p);
    // Check can be loaded from both path and string contents
    auto jig = convert_CoolProp_idealgas(p.string(), 0 /* index of EOS */);
    auto jig2 = convert_CoolProp_idealgas(load_a_JSON_file(p.string()).dump(), 0 /* index of EOS */);
    // Convert to json array
    nlohmann::json jaig = nlohmann::json::array(); jaig.push_back(jig);
    CHECK(jaig.is_array());
    
//    std::cout << jaig.dump() << std::endl;
    
    // Check that converted structures can be loaded
    auto aig = IdealHelmholtz(jaig);
}

TEST_CASE("Check that BIP can be set in a string", "[multifluida]") {
    std::string root = FLUIDDATAPATH;
    double T = 300, rhomolar = 300;
    auto z = (Eigen::ArrayXd(2) << 0.4, 0.6).finished();
    auto def = build_multifluid_model({"Nitrogen","Ethane"}, root); // default parameters
    CHECK(TDXDerivatives<decltype(def)>::get_Ar01(def, T, rhomolar, z) == Approx(-0.026028104905899584));
    std::string s = R"([{"BibTeX": "Kunz-JCED-2012", "CAS1": "7727-37-9", "CAS2": "74-84-0", "F": 1.0, "Name1": "Nitrogen", "Name2": "Ethane", "betaT": 1.01774814228, "betaV": 0.978880168, "function": "Nitrogen-Ethane", "gammaT": 1.0877732316831683, "gammaV": 1.042352891}])";
    nlohmann::json model = {
        {"components",{"Nitrogen","Ethane"}},
        {"root", root},
        {"BIP", s},
        {"departure", ""}
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", model}
    };
    auto model_ = cppinterface::make_model(j);
    CHECK(model_->get_Ar01(T, rhomolar, z) != Approx(-0.026028104905899584));
}

TEST_CASE("Check ammonia+argon", "[multifluidArNH3]") {
    std::string root = FLUIDDATAPATH;
    
    // Check that default model (no departure function) prints the right
    auto def = build_multifluid_model({"AMMONIA","ARGON"}, root); // default parameters
    nlohmann::json mixdef = nlohmann::json::parse(def.get_meta())["mix"];
//    std::cout << mix.dump(1) << std::endl;
    CHECK(!mixdef.empty());
    CAPTURE(mixdef.dump(1));
    
    std::string sBIP = R"([ {"BibTeX": "?", "CAS1": "7440-37-1", "CAS2": "7664-41-7", "F": 1.0, "Name1": "ARGON", "Name2": "AMMONIA", "betaT": 1.146326, "betaV": 0.756526, "function": "BAA", "gammaT": 0.998353, "gammaV": 1.041113}])";
    std::string sdep = R"([{"BibTeX": "??", "Name": "BAA", "Npower": 1, "aliases": [], "beta": [0.0, 0.6, 0.5], "d": [3.0, 1.0, 1.0], "epsilon": [0.0, 0.31, 0.39], "eta": [0.0, 1.3, 1.5], "gamma": [0.0, 0.9, 1.5], "l": [1.0, 0.0, 0.0], "n": [0.02350785, -1.913776, 1.624062], "t": [2.3, 1.65, 0.42], "type": "Gaussian+Exponential"}])";
    nlohmann::json jmodel = {
        {"components", {"AMMONIA", "ARGON"}},
        {"root", root},
        {"BIP", sBIP},
        {"departure", sdep}
    };
    nlohmann::json j = {
        {"kind", "multifluid"},
        {"model", jmodel}
    };
    auto model_ = cppinterface::make_model(j);
    double T = 293.15, rhomolar = 40000;
    auto z = (Eigen::ArrayXd(2) << 0.95, 0.05).finished();
    double p_MPa = (model_->get_pr(T, rhomolar*z) + rhomolar*model_->get_R(z)*T)/1e6;
    
    const auto& mref = teqp::cppinterface::adapter::get_model_cref<multifluid_t>(model_.get());
    nlohmann::json mix = nlohmann::json::parse(mref.get_meta())["mix"];
//    std::cout << mix.dump(1) << std::endl;
    CHECK(!mix.empty());
    CAPTURE(mix.dump(1));
    CHECK(p_MPa == Approx(129.07019029846455).margin(1e-3));
}


TEST_CASE("Check pure fluid throws with composition array of wrong length", "[virial]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "CarbonDioxide" }, root);
    double T = 300;
    auto z = (Eigen::ArrayXd(2) << 0.3, 0.9).finished();
    using vir = VirialDerivatives<decltype(model)>;
    CHECK_THROWS(vir::get_dmBnvirdTm<2,1>(model, T, z));
    CHECK_THROWS(vir::get_Bnvir<2>(model, T, z));
}
TEST_CASE("Test ECS for pure fluids", "[ECS]"){
    auto contents = R"({
        "kind": "multifluid-ECS-HuberEly1994",
        "model": {
          "reference_fluid": {
                "name": "?",
                "acentric": 0.25253,
                "Z_crit": 0.280191,
                "T_crit / K": 487.21,
                "rhomolar_crit / mol/m^3": 2988.659106070714
          },
          "fluid": {
                "name": "C4F10",
                "acentric": 0.371,
                "f_T_coeffs": [ 0.00776042865, -0.641975631],
                "h_T_coeffs": [ 0.00278313281, -0.593657910],
                "rhomolar_crit / mol/m^3": 2520.0,
                "T_crit / K": 386.326,
                "Z_crit": 0.28703530765310314
          }
        }
    })"_json;
    contents["model"]["reference_fluid"]["name"] = FLUIDDATAPATH + "/dev/fluids/R113.json";
    auto model = teqp::cppinterface::make_model(contents);
    double T = 400, rho = 2700;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    model->get_Ar00(T, rho, z);
}

TEST_CASE("Check models for R", "[multifluidR]") {
    std::string root = FLUIDDATAPATH;
    
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    SECTION("Default, mole fraction weighted"){
        nlohmann::json model = {
            {"components", {"Water"}},
            {"root", root},
            {"BIP", ""},
            {"departure", ""}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        auto model_ = cppinterface::make_model(j);
        CHECK(model_->get_R(z) == 8.314371357587);
    }
    
    SECTION("CODATA"){
        nlohmann::json model = {
            {"components", {"Water"}},
            {"root", root},
            {"BIP", ""},
            {"departure", ""},
            {"flags", {{"Rmodel", "CODATA"}}}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        auto model_ = cppinterface::make_model(j);
        CHECK(model_->get_R(z) == 8.31446261815324);
    }
}

TEST_CASE("Ar20 for CO2", "[Ar20CO2]"){
    double rho = 10624.9063; // mol/m^3
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    
    using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>; // Overkill: 200 digits of working precision!
    auto model = build_multifluid_model({"CO2"}, FLUIDDATAPATH);
    
    auto f = [&rho, &z, &model](const auto Trecip){ return model.alphar(1.0/Trecip, rho, z); };
    
    std::cout << std::setprecision(20);
//    std::cout << "T / K,rho / mol/m^3,multiprecision,autodiff,err2MP" << std::endl;
    for (double T = 304.2; T < 340; T += 0.05){
        my_float Trecip = 1.0/T;
        my_float h = 1e-20;
        auto mp = -teqp::centered_diff<2,6>(f, Trecip, h)*Trecip*Trecip; // cvr/R
        
        using tdx = TDXDerivatives<decltype(model), double>;
        auto ad = -tdx::get_Ar20(model, T, rho, z);
//        std::cout << T << "," << rho << "," << mp << "," << ad << "," << mp/ad-1 << std::endl;
    }
}

TEST_CASE("Check composition derivatives for ternary with all one component", "[ternary]") {
    std::string root = FLUIDDATAPATH;
    const auto model = build_multifluid_model({ "Methane","Ethane","n-Propane" }, root);
    double T = 300, rho = 10.0;
    
    auto z = (Eigen::ArrayXd(3) << 1, 0, 0).finished();
    using tdx = TDXDerivatives<decltype(model)>;
    using iso = IsochoricDerivatives<decltype(model)>;
    double valdil = tdx::get_Arxy<0, 1>(model, T, rho, z);
    CAPTURE(valdil);
    CHECK(std::isfinite(valdil));
    auto graddil = iso::build_Psir_gradient_autodiff(model, T, rho*z);
    CHECK_THROWS(iso::build_Psir_Hessian_autodiff(model, T, rho*z));
    
    double dx = 1e-13;
    auto zalmost = (Eigen::ArrayXd(3) << 1-2*dx, dx, dx).finished();
    double valalmost = tdx::get_Arxy<0, 1>(model, T, rho, zalmost);
    CAPTURE(valalmost);
    CHECK(std::isfinite(valalmost));
    auto gradalmost = iso::build_Psir_gradient_autodiff(model, T, rho*zalmost);
    auto Hessalmost = iso::build_Psir_Hessian_autodiff(model, T, rho*zalmost);
    auto deltagrad = graddil - gradalmost;
    CAPTURE(deltagrad);
    
    auto fT = [&](const auto &molefracs){ return model.redfunc.get_Tr(molefracs); };
    ArrayXdual2nd zz = z.cast<dual2nd>();
    
    dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
    ArrayXdual g;
    CHECK_THROWS(autodiff::hessian(fT, wrt(zz), at(zz), u, g));
    
    ArrayXdual2nd zzalmost = zalmost.cast<dual2nd>();
    dual2nd ualmost; // the output scalar u = f(x), evaluated together with Hessian below
    ArrayXdual galmost;
    auto HessTralmost = autodiff::hessian(fT, wrt(zzalmost), at(zzalmost), ualmost, galmost).eval();
    CAPTURE(HessTralmost);
    
    CHECK_THAT(valdil, WithinRel(valalmost, 1e-10));
    CHECK_THAT(graddil[0], WithinRel(gradalmost[0], 1e-10));
    CHECK_THAT(graddil[1], WithinRel(gradalmost[1], 1e-10));
    CHECK_THAT(graddil[2], WithinRel(gradalmost[2], 1e-10));
}


TEST_CASE("Check adding simpler EOS as pure fluid contribution in multifluid approach", "[multifluidpuremodels]") {
    std::string root = FLUIDDATAPATH;
    
    SECTION("cubic EOS"){
        
        // Load some existing data from the JSON structure to avoid repeating ourselves
        auto f = load_a_JSON_file(root+"/dev/fluids/CarbonDioxide.json");
        
        // Flush existing contributions
        f["EOS"][0]["alphar"].clear();
        
        // Overwrite the residual portion with a cubic EOS
        // In this case SRK which defines the values for
        // OmegaA, OmegaB, Delta1 and Delta2, all other values taken
        // from the Span&Wagner EOS
        auto reducing = f["EOS"][0]["STATES"]["reducing"];
        double Tc_K = reducing.at("T");
        double pc_Pa = reducing.at("p");
        double rhoc_molm3 = reducing.at("rhomolar");
        f["EOS"][0]["alphar"].push_back({
            {"R / J/mol/K", 8.31446261815324},
            {"OmegaA", 0.42748023354034140439},
            {"OmegaB", 0.086640349964957721589},
            {"Delta1", 1.0},
            {"Delta2", 0.0},
            {"Tcrit / K", Tc_K},
            {"pcrit / Pa", pc_Pa},
            // Reducing state variables are taken from critical point
            {"Tred / K", Tc_K},
            {"rhored / mol/m^3", rhoc_molm3},
            {"alpha", {
                {{"type", "Twu"}, {"c", {1, 2, 3}}} // Dummy values for coefficients
            }},
            {"type", "ResidualHelmholtzGenericCubic"}
        });
        //    std::cout << f["EOS"][0]["alphar"].dump(1) << std::endl;
        
        nlohmann::json model = {
            {"components", {f}}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        auto model_ = cppinterface::make_model(j);
    }
    SECTION("PC-SAFT"){
        // Load some existing data from the JSON structure to avoid repeating ourselves
        auto f = load_a_JSON_file(root+"/dev/fluids/CarbonDioxide.json");
        
        // Flush existing contributions
        f["EOS"][0]["alphar"].clear();
        
        // Overwrite the residual portion with the PC-SAFT EOS
        auto reducing = f["EOS"][0]["STATES"]["reducing"];
        double Tc_K = reducing.at("T");
        double pc_Pa = reducing.at("p");
        double rhoc_molm3 = reducing.at("rhomolar");
        f["EOS"][0]["alphar"].push_back({
            // Reducing state variables are taken from critical point
            {"Tred / K", Tc_K},
            {"rhored / mol/m^3", rhoc_molm3},
            {"m", 1.593}, // placeholder value for testing
            {"sigma / A", 3.445}, // placeholder value for testing
            {"epsilon_over_k", 176.47}, // placeholder value for testing
            {"type", "ResidualHelmholtzPCSAFTGrossSadowski2001"}
        });
        //    std::cout << f["EOS"][0]["alphar"].dump(1) << std::endl;
        
        nlohmann::json model = {
            {"components", {f}}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        auto model_ = cppinterface::make_model(j);
    }
    SECTION("PC-SAFT cyclopentane+water explicit"){
        
        double Tred_K = 511.72, rhored_molm3=3920;
        // Build the residual portion with the PC-SAFT EOS
        nlohmann::json alphar = {
            // Reducing state variables are taken from critical point
            {"Tred / K", Tred_K},
            {"rhored / mol/m^3", rhored_molm3},
            {"m", 2.3655}, // placeholder value for testing
            {"sigma / A", 3.7114}, // placeholder value for testing
            {"epsilon_over_k", 265.83}, // placeholder value for testing
            {"type", "ResidualHelmholtzPCSAFTGrossSadowski2001"}
        };
        
        nlohmann::json alphars; alphars.push_back(alphar);
        
        // The reducing state is needed for mixture models
        nlohmann::json reducing = {
            {"T", Tred_K},
            {"rhomolar", rhored_molm3},
        };
        nlohmann::json states = {{"reducing", reducing}};
        nlohmann::json EOS = {
            {"alphar", alphars},
            {"STATES", states},
            {"gas_constant", constants::R_CODATA2017}
        };
        
        // And we need to store some identifiers for use in mixtures
        nlohmann::json info = {
            {"NAME", "Cyclopentane"},
            {"CAS", "287-92-3"},
            {"REFPROP_NAME", "CYCLOPEN"},
            {"HASH", "43ab1810"}
        };
        nlohmann::json EOSlist; EOSlist = nlohmann::json::array(); EOSlist.push_back(EOS);
        nlohmann::json f = {{"EOS", EOSlist}, {"INFO", info}};
        
        nlohmann::json model = {
            {"components", {f, "Water"}},
            {"BIP", ""},
            {"departure", ""},
            {"flags", {{"force-estimate", "yes"},{"estimate","Lorentz-Berthelot"}}}
        };
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", model}
        };
        
        j["model"]["root"] = FLUIDDATAPATH;
        std::string as_str = j.dump(2);
        CAPTURE(as_str);
        
        auto model_ = cppinterface::make_model(j);
//        CHECK(0==1);
    }
}
