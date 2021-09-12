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

struct OneTiming {
    double value, sec_per_call;
};

template<typename Taus, typename Deltas>
auto some_REFPROP(int itau, int idelta, Taus &taus, Deltas &deltas) {
    std::vector<OneTiming> o;
    double z[20] = { 1.0 };
    for (auto repeat = 0; repeat < 100; ++repeat) {
        std::valarray<double> ps = 0.0 * taus;
        double Arterm = -10000;
        auto tic = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < taus.size(); ++i) {
            PHIXdll(itau, idelta, taus[i], deltas[i], z, Arterm); ps[i] = Arterm;
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double elap_us = std::chrono::duration<double>(toc - tic).count() / taus.size() * 1e6;
        double val = std::accumulate(std::begin(ps), std::end(ps), 0.0) / ps.size();
        OneTiming result = { val, elap_us };
        o.emplace_back(result);
    }
    return o;
}

template<int itau, int idelta, typename Taus, typename Deltas, typename TT, typename RHO, typename Model>
auto some_teqp(const Taus& taus, const Deltas& deltas, const Model &model, const TT &Ts, const RHO &rhos) {
    std::vector<OneTiming> out;

    // And the same example with teqp
    auto N = taus.size();
    auto c = (Eigen::ArrayXd(1) << 1.0).finished();

    using tdx = TDXDerivatives<Model, double, decltype(c)>;

    for (auto counter = 0; counter < 100; ++counter)
    {
        double o = 0.0;
        auto tic = std::chrono::high_resolution_clock::now();
        for (auto j = 0; j < N; ++j) {
            if constexpr (itau == 0 && idelta == 0) {
                o += tdx::get_Ar00(model, Ts[j], rhos[j], c);
            }
            else if constexpr (itau == 0 && idelta == 1) {
                o += tdx::get_Ar01(model, Ts[j], rhos[j], c);
            }
            else if constexpr (itau == 0 && idelta == 2) {
                o += tdx::get_Ar02(model, Ts[j], rhos[j], c);
            }
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double elap_us = std::chrono::duration<double>(toc - tic).count() / taus.size() * 1e6;
        double val = o / N;
        OneTiming result = { val, elap_us };
        out.emplace_back(result);
    }
    return out;
}

int main()
{
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = "C:/Program Files (x86)/REFPROP";
    std::string DLL_name = "REFPRP64.dll";

    // Load the shared library and set up the fluid
    std::string err;
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop: %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP){return EXIT_FAILURE; }
    SETPATHdll(const_cast<char*>(path.c_str()), 400);
    int ierr = 0, nc = 1;
    char herr[255], hfld[10000] = "PROPANE", hhmx[255] = "HMX.BNC", href[4] = "DEF";
    SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
    {
        char hflag[256] = "Cache                                                ";
        int jFlag = 3, kFlag = -1;
        FLAGSdll(hflag, jFlag, kFlag, ierr, herr, 255, 255);
        std::cout << kFlag << std::endl;
    }

    if (ierr != 0) printf("This ierr: %d herr: %s\n", ierr, herr);
    {
        double d = 1.0;

        auto model = build_multifluid_model({ "n-Propane" }, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");

        double rhoc = 1/model.redfunc.vc[0];
        double Tc = model.redfunc.Tc[0];

        //
        std::default_random_engine re;        
        std::valarray<double> taus(100000);
        {
            std::uniform_real_distribution<double> unif(2.0941098901098902, 2.1941098901098902);
            std::transform(std::begin(taus), std::end(taus), std::begin(taus), [&unif, &re](double x) { return unif(re); });
        }
        std::valarray<double> deltas(taus.size()); {
            std::uniform_real_distribution<double> unif(0.0015981745536338204, 0.0016981745536338204);
            std::transform(std::begin(deltas), std::end(deltas), std::begin(deltas), [&unif, &re](double x) { return unif(re); });
        }

        auto check_values = [](auto res) {
            Eigen::ArrayXd vals(res.size());
            for (auto i = 0; i < res.size(); ++i) { vals[i] = res[i].value; }
            if (std::abs(vals.maxCoeff() - vals.minCoeff()) > 1e-15 * std::abs(vals.minCoeff())) {
                throw std::invalid_argument("Didn't get the same value for all inputs");
            }
            return vals.mean();
        };

        constexpr int itau = 0, idelta = 0;

        auto timingREFPROP = some_REFPROP(itau, idelta, taus, deltas);

        auto Ts = Tc / taus;
        auto rhos = deltas * rhoc;
        auto timingteqp = some_teqp<itau, idelta>(taus, deltas, model, Ts, rhos);
        
        std::cout << "Values:" << check_values(timingREFPROP) << ", " <<  check_values(timingteqp) << std::endl;

        for (auto i = 0; i < timingREFPROP.size(); ++i) {
            std::cout << timingteqp[i].sec_per_call << ", " << timingREFPROP[i].sec_per_call << std::endl;
        }
    }
    return EXIT_SUCCESS;
}