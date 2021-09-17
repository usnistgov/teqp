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
#include "teqp/models/eos.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/derivs.hpp"

struct OneTiming {
    double value, sec_per_call;
};

constexpr int repeatmax = 100;
enum class obtainablethings { PHIX, CHEMPOT };

template<typename Taus, typename Deltas>
auto some_REFPROP(obtainablethings thing, int itau, int idelta, Taus& taus, Deltas& deltas) {
    std::vector<OneTiming> o;
    
    if (thing == obtainablethings::PHIX) {
        double z[20] = { 1.0 };
        for (auto repeat = 0; repeat < repeatmax; ++repeat) {
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
    }
    else {

    }
    return o;
}

template<int itau, int idelta, typename Taus, typename Deltas, typename TT, typename RHO, typename Model>
auto some_teqp(obtainablethings thing, const Taus& taus, const Deltas& deltas, const Model &model, const TT &Ts, const RHO &rhos) {
    std::vector<OneTiming> out;

    // And the same example with teqp
    auto N = taus.size();
    auto c = (Eigen::ArrayXd(1) << 1.0).finished();

    using tdx = TDXDerivatives<Model, double, decltype(c)>;

    if (thing == obtainablethings::PHIX) {
        for (auto repeat = 0; repeat < repeatmax; ++repeat)
        {
            double o = 0.0;
            auto tic = std::chrono::high_resolution_clock::now();
            for (auto j = 0; j < N; ++j) {
                if constexpr (itau == 0 && idelta == 0) {
                    o += tdx::get_Ar00(model, Ts[j], rhos[j], c);
                }
                else if constexpr (itau == 0 && idelta == 1) {
                    o += tdx::get_Ar01(model, Ts[j], rhos[j], c);
                    //o += tdx::get_Ar0n<1>(model, Ts[j], rhos[j], c)[1];
                }
                else if constexpr (itau == 0 && idelta == 2) {
                    o += tdx::get_Ar02(model, Ts[j], rhos[j], c);
                }
                else if constexpr (itau == 0 && idelta > 2) {
                    o += tdx::template get_Ar0n<idelta>(model, Ts[j], rhos[j], c)[idelta];
                }
            }
            auto toc = std::chrono::high_resolution_clock::now();
            double elap_us = std::chrono::duration<double>(toc - tic).count() / taus.size() * 1e6;
            double val = o / N;
            OneTiming result = { val, elap_us };
            out.emplace_back(result);
        }
    }
    else {
    }
    return out;
}

template<int itau, int idelta, typename Taus, typename Deltas, typename TT, typename RHO, typename Model>
auto one_deriv(obtainablethings thing, Taus& taus, Deltas& deltas, const Model& model, const std::string &modelname, TT& Ts, RHO& rhos) {

    auto check_values = [](auto res) {
        Eigen::ArrayXd vals(res.size());
        for (auto i = 0; i < res.size(); ++i) { vals[i] = res[i].value; }
        if (std::abs(vals.maxCoeff() - vals.minCoeff()) > 1e-15 * std::abs(vals.minCoeff())) {
            throw std::invalid_argument("Didn't get the same value for all inputs");
        }
        return vals.mean();
    };

    std::cout << "Ar_{" << itau << "," << idelta << "}" << std::endl;

    auto timingREFPROP = some_REFPROP(thing, itau, idelta, taus, deltas);
    auto timingteqp = some_teqp<itau, idelta>(thing, taus, deltas, model, Ts, rhos);

    std::cout << "Values:" << check_values(timingREFPROP) << ", " << check_values(timingteqp) << std::endl;

    auto N = timingREFPROP.size();
    std::vector<double> timesteqp, timesREFPROP, valsteqp, valsREFPROP;
    for (auto i = 0; i < N; ++i) {
        timesteqp.push_back(timingteqp[i].sec_per_call);
        timesREFPROP.push_back(timingREFPROP[i].sec_per_call);
        valsteqp.push_back(timingteqp[i].value);
        valsREFPROP.push_back(timingREFPROP[i].value);
    }
    for (auto i = 1; i < 6; ++i) {
        std::cout << timingteqp[N-i].sec_per_call << ", " << timingREFPROP[N-i].sec_per_call << std::endl;
    }
    nlohmann::json j = {
        {"timeteqp",timesteqp},
        {"timeREFPROP",timesREFPROP},
        {"valteqp",valsteqp},
        {"valREFPROP",valsREFPROP},
        {"model", modelname},
        {"itau", itau},
        {"idelta", idelta}
    };
    return j;
}

int main()
{
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = std::getenv("RPPREFIX");
    std::string DLL_name = "";

    // Load the shared library and set up the fluid
    std::string err;
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop: %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP) { return EXIT_FAILURE; }
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
        auto model = build_multifluid_model({ "n-Propane" }, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");

        double rhoc = 1/model.redfunc.vc[0];
        double Tc = model.redfunc.Tc[0];

        //
        std::default_random_engine re;
        std::valarray<double> taus(10000);
        {
            std::uniform_real_distribution<double> unif(2.0941098901098902, 2.1941098901098902);
            std::transform(std::begin(taus), std::end(taus), std::begin(taus), [&unif, &re](double x) { return unif(re); });
        }
        std::valarray<double> deltas(taus.size()); {
            std::uniform_real_distribution<double> unif(0.0015981745536338204, 0.0016981745536338204);
            std::transform(std::begin(deltas), std::end(deltas), std::begin(deltas), [&unif, &re](double x) { return unif(re); });
        }

        auto Ts = Tc / taus;
        auto rhos = deltas * rhoc;

        obtainablethings thing = obtainablethings::PHIX;

        nlohmann::json outputs = nlohmann::json::array();

        auto build_vdW = [](auto Ncomp) {
            std::valarray<double> Tc_K(Ncomp), pc_Pa(Ncomp);
            for (int i = 0; i < Ncomp; ++i) {
                Tc_K[i] = 100.0 + 10.0 * i;
                pc_Pa[i] = 1e6 + 0.1e6 * i;
            }
            return vdWEOS(Tc_K, pc_Pa);
        };
        auto vdW = build_vdW(1);
        outputs.push_back(one_deriv<0, 0>(thing, taus, deltas, vdW, "vdW", Ts, rhos));
        outputs.push_back(one_deriv<0, 1>(thing, taus, deltas, vdW, "vdW", Ts, rhos));
        outputs.push_back(one_deriv<0, 2>(thing, taus, deltas, vdW, "vdW", Ts, rhos));
        outputs.push_back(one_deriv<0, 3>(thing, taus, deltas, vdW, "vdW", Ts, rhos));

        auto build_PCSAFT = [](auto Ncomp) {
            SAFTCoeffs coeff;
            std::vector<SAFTCoeffs> coeffs(1);
            auto &c = coeffs[0];
            c.m = 2.0020;
            c.sigma_Angstrom = 3.6184;
            c.epsilon_over_k = 208.11;
            c.name = "propane";
            c.BibTeXKey = "Gross-IECR-2001";
            return PCSAFTMixture(coeffs);
        };
        auto SAFT = build_PCSAFT(1);
        outputs.push_back(one_deriv<0, 0>(thing, taus, deltas, SAFT, "PCSAFT", Ts, rhos));
        outputs.push_back(one_deriv<0, 1>(thing, taus, deltas, SAFT, "PCSAFT", Ts, rhos));
        outputs.push_back(one_deriv<0, 2>(thing, taus, deltas, SAFT, "PCSAFT", Ts, rhos));
        outputs.push_back(one_deriv<0, 3>(thing, taus, deltas, SAFT, "PCSAFT", Ts, rhos));

        outputs.push_back(one_deriv<0, 0>(thing, taus, deltas, model, "multifluid", Ts, rhos));
        outputs.push_back(one_deriv<0, 1>(thing, taus, deltas, model, "multifluid", Ts, rhos));
        outputs.push_back(one_deriv<0, 2>(thing, taus, deltas, model, "multifluid", Ts, rhos));
        outputs.push_back(one_deriv<0, 3>(thing, taus, deltas, model, "multifluid", Ts, rhos));

        std::ofstream file("Ar0n_timings.json");
        file << outputs;
    }
    return EXIT_SUCCESS;
}