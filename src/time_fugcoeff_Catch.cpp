#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/generators/catch_generators.hpp>

// Only this file gets the implementation
#define REFPROP_IMPLEMENTATION
#define REFPROP_FUNCTION_MODIFIER
#include "REFPROP_lib.h"
#undef REFPROP_FUNCTION_MODIFIER
#undef REFPROP_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <valarray>
#include <random>
#include <numeric>

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"

using namespace teqp;

static bool loaded_refprop = false;

#include "tests/test_common.in"

void init_REFPROP() {

    if (loaded_refprop) { return; }

    // you may need to change this path to suit your installation
    // note: forward-slashes are recommended.
    std::string path = std::getenv("RPPREFIX");
    std::string dll_name = "";

    // load the shared library and set up the fluid
    std::string err;
    loaded_refprop = load_REFPROP(err, path, dll_name);
    printf("loaded refprop: %s @ address %zu\n", loaded_refprop ? "true" : "false", REFPROP_address());
    {
        char hpath[256] = " ";
        strcpy(hpath, const_cast<char*>(path.c_str()));
        SETPATHdll(hpath, 255);
    }

    // Try to disable caching in REFPROP
    {
        int ierr = 0; char herr[256];
        char hflag[256] = "Cache                                                ";
        int jFlag = 3, kFlag = -1;
        FLAGSdll(hflag, jFlag, kFlag, ierr, herr, 255, 255);
        //std::cout << kFlag << std::endl;
    }    
}

TEST_CASE("Time fugacity coefficient"){
    
    init_REFPROP();

    // Implicit sections are built for each number of Ncomp
    auto Ncomp = GENERATE(1,2,3,4,5,6);

    double T = 300, D_moldm3 = 3, D_molm3 = D_moldm3*1e3;

    // Setup teqp model
    std::vector<std::string> component_list = { "Methane","Ethane","n-Propane","n-Butane","n-Pentane","n-Hexane" };
    std::vector<std::string> fluid_set(component_list.begin(), component_list.begin() + Ncomp);
    auto model = build_multifluid_model(fluid_set, FLUIDDATAPATH, FLUIDDATAPATH + "/dev/mixtures/mixture_binary_pairs.json");
    using id = IsochoricDerivatives<decltype(model), double>;
    auto rhovec = (D_molm3*Eigen::ArrayXd::Ones(Ncomp)/Ncomp).eval();

    auto setup_REFPROP = [&]() {
        // Initialize the model
        {
            std::string name = fluid_set[0];
            for (auto j = 1; j < fluid_set.size(); ++j) {
                name += "*" + fluid_set[j];
            }
            int ierr = 0, nc = Ncomp;
            char herr[256] = " ", hfld[10001] = " ", hhmx[256] = "HMX.BNC", href[4] = "DEF";
#if defined(USE_TEQP_HMX)
            std::string rhs = std::string("./teqpHMX.BNC") + "\0";
            strncpy(hhmx, rhs.c_str(), rhs.size());
#endif
            strcpy(hfld, (name + "\0").c_str());
            SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
            if (ierr != 0) {
                printf("This ierr: %d herr: %s\n", ierr, herr);
            }
        }
    };
    setup_REFPROP();

    std::valarray<double> z(20); z = 0.0; z[std::slice(0, Ncomp, 1)] = 1.0 / Ncomp;
    std::valarray<double> u(20); u = 0.0;
    int ierr = 0; char herr[256];

    SECTION("check are the same") {
        FUGCOFdll(T, D_moldm3, &(z[0]), &(u[0]), ierr, herr, 255);
        auto phiteqp = id::template get_fugacity_coefficients(model, T, rhovec);
        auto diff = (phiteqp - Eigen::Map<const Eigen::ArrayXd>(&(u[0]), phiteqp.size())).eval();
        REQUIRE(diff.abs().minCoeff() < 1e-13);
    };
    BENCHMARK(std::string("teqp") + std::to_string(Ncomp)) {
        return id::template get_fugacity_coefficients(model, T, rhovec);
    };
    BENCHMARK(std::string("REFPROP") + std::to_string(Ncomp)) {
        FUGCOFdll(T, D_moldm3, &(z[0]), &(u[0]), ierr, herr, 255);
        return u;
    };
}