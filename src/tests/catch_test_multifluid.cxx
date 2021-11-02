#include "catch/catch.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/filesystem.hpp"


TEST_CASE("Confirm failure for missing files","[multifluid]") {
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "IMPOSSIBLE PATH.json"));
    CHECK_THROWS(build_multifluid_model({ "BADFLUID" }, "IMPOSSIBLE PATH", "../mycp/dev/mixtures/mixture_binary_pairs.json"));
}

TEST_CASE("Trace critical locus for nitrogen + ethane", "[crit],[multifluid]")
{
    std::string root = "../mycp";
    const auto model = build_multifluid_model({ "Nitrogen", "Ethane" }, root, root+"/dev/mixtures/mixture_binary_pairs.json");

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

TEST_CASE("Check that all models can be instantiated", "[multifluid],[all]"){
    std::string root = "../mycp";
    for (auto filename : get_files_in_folder(root + "/dev/fluids", ".json")) {
        auto stem = filename.stem().string(); // filename without the .json
        if (stem == "Methanol") { continue; }
        auto model = build_multifluid_model({ stem }, root, root + "/dev/mixtures/mixture_binary_pairs.json");
    }
}