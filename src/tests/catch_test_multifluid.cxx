#include "catch/catch.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/filesystem.hpp"


TEST_CASE("Confirm failure for missing files","[multifluid]") {
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "IMPOSSIBLE PATH.json"));
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "../mycp/dev/mixtures/mixture_binary_pairs.json"));
    CHECK_THROWS(build_multifluid_model({ "Ethane" }, "IMPOSSIBLE PATH"));
}

TEST_CASE("Trace critical locus for nitrogen + ethane", "[crit],[multifluid]")
{
    std::string root = "../mycp";
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
    std::string root = "../mycp";
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

TEST_CASE("Check that mixtures can also do absolute paths", "[multifluid],[abspath]") {
    std::string root = "../mycp";
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