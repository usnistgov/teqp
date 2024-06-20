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

// On windows, the small macro is defined in a header.  Sigh...
#if defined(small)
#undef small
#endif

#include "teqp/models/multifluid.hpp"
#include "teqp/models/vdW.hpp"
#include "teqp/models/cubics.hpp"
#include "teqp/models/pcsaft.hpp"
using namespace teqp::PCSAFT;
#include "teqp/derivs.hpp"

#include "tests/test_common.in"

using namespace teqp;

struct OneTiming {
    double value, sec_per_call;
};

constexpr int repeatmax = 10000;
enum class obtainablethings { PHIX, CHEMPOT };

template<typename Taus, typename Deltas, typename TT, typename RHO>
auto some_REFPROP(obtainablethings thing, int Ncomp, int itau, int idelta, Taus& taus, Deltas& deltas, const TT &Ts, const RHO &rhos) {
    std::vector<OneTiming> o;
    
    if (thing == obtainablethings::PHIX) {
        std::valarray<double> z(20); z = 0;
        for (auto i = 0; i < Ncomp; ++i) {
            z[i] = 1.0 / static_cast<double>(Ncomp);
        }
        for (auto repeat = 0; repeat < repeatmax; ++repeat) {
            std::valarray<double> ps = 0.0 * taus;
            double Arterm = -10000;
            auto tic = std::chrono::high_resolution_clock::now();
            double Tr=-1, Dr=-1;
            for (auto i = 0; i < taus.size(); ++i) {
                REDXdll(&(z[0]), Tr, Dr);
                // rhos are in mol/m^3, but REFPROP gives Dr in mol/L
                double tau = Tr / Ts[i], delta = rhos[i]/1000.0/Dr;
                PHIXdll(itau, idelta, taus[i], deltas[i], &(z[0]), Arterm); ps[i] = Arterm;
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

template<int itau, int idelta, ADBackends backend, typename Taus, typename Deltas, typename TT, typename RHO, typename Model>
auto some_teqp(obtainablethings thing, int Ncomp, const Taus& taus, const Deltas& deltas, const Model &model, const TT &Ts, const RHO &rhos) {
    std::vector<OneTiming> out;

    // And the same example with teqp
    auto N = taus.size();
    auto c = (Eigen::ArrayXd::Ones(Ncomp) / static_cast<double>(Ncomp)).eval();

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
                    o += tdx::template get_Ar01<backend>(model, Ts[j], rhos[j], c);
                    //o += tdx::get_Ar0n<1>(model, Ts[j], rhos[j], c)[1];
                }
                else if constexpr (itau == 0 && idelta == 2) {
                    o += tdx::template get_Ar02<backend>(model, Ts[j], rhos[j], c);
                }
                else if constexpr (itau == 0 && idelta > 2) {
                    o += tdx::template get_Ar0n<idelta, backend>(model, Ts[j], rhos[j], c)[idelta];
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
auto one_deriv(obtainablethings thing, int Ncomp, Taus& taus, Deltas& deltas, const Model& model, const std::string &modelname, TT& Ts, RHO& rhos) {

    auto check_values = [](auto res) {
        Eigen::ArrayXd vals(res.size());
        for (auto i = 0; i < res.size(); ++i) { vals[i] = res[i].value; }
        if (std::abs(vals.maxCoeff() - vals.minCoeff()) > 1e-15 * std::abs(vals.minCoeff())) {
            throw std::invalid_argument("Didn't get the same value for all inputs");
        }
        return vals.mean();
    };

    std::cout << modelname << std::endl;
    std::cout << "Ar_{" << itau << "," << idelta << "}" << std::endl;

    auto timingREFPROP = some_REFPROP(thing, Ncomp, itau, idelta, taus, deltas, Ts, rhos);
    auto timingteqpad = some_teqp<itau, idelta, ADBackends::autodiff>(thing, Ncomp, taus, deltas, model, Ts, rhos);
    //auto timingteqpmcx = some_teqp<itau, idelta, ADBackends::multicomplex>(thing, Ncomp, taus, deltas, model, Ts, rhos);

    std::cout << "Values:" << check_values(timingREFPROP) << ", " << check_values(timingteqpad) << std::endl;

    auto N = timingREFPROP.size();
    std::vector<double> timesteqpad, timesteqpmcx, timesREFPROP, 
                        valsteqpad,   valsteqpmcx,  valsREFPROP;
    for (auto i = 0; i < N; ++i) {
        timesteqpad.push_back(timingteqpad[i].sec_per_call);
        //timesteqpmcx.push_back(timingteqpmcx[i].sec_per_call);
        timesREFPROP.push_back(timingREFPROP[i].sec_per_call);
        valsteqpad.push_back(timingteqpad[i].value);
        //valsteqpmcx.push_back(timingteqpmcx[i].value);
        valsREFPROP.push_back(timingREFPROP[i].value);
    }
    for (auto i = 1; i < 6; ++i) {
        std::cout << timingteqpad[N-i].sec_per_call << ", " << timingREFPROP[N-i].sec_per_call << std::endl;
    }
    nlohmann::json j = {
        {"timeteqp",timesteqpad},
        {"timeteqp(autodiff)",timesteqpad},
        //{"timeteqp(multicomplex)",timesteqpmcx},
        {"timeREFPROP",timesREFPROP},
        {"valteqp(autodiff)",valsteqpad},
        //{"valteqp(multicomplex)",valsteqpmcx},
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
    char hpath[256] = " ";
    strcpy(hpath, const_cast<char*>(path.c_str()));
    SETPATHdll(hpath, 255);
    
    {
        int ierr = 0; 
        char hflag[256] = "Cache                                                ", herr[256] = " ";
        int jFlag = 3, kFlag = -1;
        FLAGSdll(hflag, jFlag, kFlag, ierr, herr, 255, 255);
        //std::cout << kFlag << std::endl;
    }
    {
        // Prepare some input values. It doesn't matter what the values of tau and delta are,
        // so long as they are not the same since we are not doing a phase equilibrium calculation, just
        // non-iterative calculations
        auto dummymodel = build_multifluid_model({ "n-Propane" }, FLUIDDATAPATH, FLUIDDATAPATH + "/dev/mixtures/mixture_binary_pairs.json");
        double rhoc = 1/dummymodel.redfunc.vc[0];
        double Tc = dummymodel.redfunc.Tc[0];
        std::default_random_engine re;
        std::valarray<double> taus(100);
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

        std::vector<std::string> component_list = { "n-Propane","Ethane","Methane","n-Butane","n-Pentane","n-Hexane" };
        for (int Ncomp : {1, 2, 3, 4, 5, 6}) {
            std::vector<std::string> fluid_set(component_list.begin(), component_list.begin() + Ncomp);

            // Initialize the model
            {
                std::string name = fluid_set[0];
                for (auto j = 1; j < fluid_set.size(); ++j) {
                    name += "*" + fluid_set[j];
                }
                int ierr = 0, nc = Ncomp;
                char herr[255], hfld[10000] = " ", hhmx[255] = "HMX.BNC", href[4] = "DEF";
#if defined(USE_TEQP_HMX)
                std::string rhs = std::string("./teqpHMX.BNC") + "\0";
                strncpy(hhmx, rhs.c_str(), rhs.size());
#endif
                strcpy(hfld, (name + "\0").c_str());
                SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
                if (ierr != 0) printf("This ierr: %d herr: %s\n", ierr, herr);
            }

            auto append_Ncomp = [&outputs, &Ncomp]() { outputs.back()["Ncomp"] = Ncomp; };
            
            auto model = build_multifluid_model(fluid_set, FLUIDDATAPATH, FLUIDDATAPATH + "/dev/mixtures/mixture_binary_pairs.json");

            auto build_vdW = [](auto Ncomp) {
                std::valarray<double> Tc_K(Ncomp), pc_Pa(Ncomp);
                for (int i = 0; i < Ncomp; ++i) {
                    Tc_K[i] = 100.0 + 10.0 * i;
                    pc_Pa[i] = 1e6 + 0.1e6 * i;
                }
                return vdWEOS(Tc_K, pc_Pa);
            };
            auto vdW = build_vdW(Ncomp);

            auto build_PCSAFT = [](auto Ncomp) {
                std::vector<SAFTCoeffs> coeffs;
                for (auto i = 0; i < Ncomp; ++i) {
                    // Values don't matter to the computer, just make them all the same...
                    SAFTCoeffs c;
                    c.m = 2.0020;
                    c.sigma_Angstrom = 3.6184;
                    c.epsilon_over_k = 208.11;
                    c.name = "propane";
                    c.BibTeXKey = "Gross-IECR-2001";
                    coeffs.push_back(c);
                }
                return PCSAFTMixture(coeffs);
            };
            auto SAFT = build_PCSAFT(Ncomp);

            std::valarray<double> Tc_K(369.89, Ncomp), pc_Pa(4251200.0, Ncomp), acentric(0.1521, Ncomp);
            auto PR = canonical_PR(Tc_K, pc_Pa, acentric);

            outputs.push_back(one_deriv<0, 0>(thing, Ncomp, taus, deltas, vdW, "vdW", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 1>(thing, Ncomp, taus, deltas, vdW, "vdW", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 2>(thing, Ncomp, taus, deltas, vdW, "vdW", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 3>(thing, Ncomp, taus, deltas, vdW, "vdW", Ts, rhos)); append_Ncomp();

            outputs.push_back(one_deriv<0, 0>(thing, Ncomp, taus, deltas, PR, "PR", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 1>(thing, Ncomp, taus, deltas, PR, "PR", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 2>(thing, Ncomp, taus, deltas, PR, "PR", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 3>(thing, Ncomp, taus, deltas, PR, "PR", Ts, rhos)); append_Ncomp();

            outputs.push_back(one_deriv<0, 0>(thing, Ncomp, taus, deltas, SAFT, "PCSAFT", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 1>(thing, Ncomp, taus, deltas, SAFT, "PCSAFT", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 2>(thing, Ncomp, taus, deltas, SAFT, "PCSAFT", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 3>(thing, Ncomp, taus, deltas, SAFT, "PCSAFT", Ts, rhos)); append_Ncomp();

            outputs.push_back(one_deriv<0, 0>(thing, Ncomp, taus, deltas, model, "multifluid", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 1>(thing, Ncomp, taus, deltas, model, "multifluid", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 2>(thing, Ncomp, taus, deltas, model, "multifluid", Ts, rhos)); append_Ncomp();
            outputs.push_back(one_deriv<0, 3>(thing, Ncomp, taus, deltas, model, "multifluid", Ts, rhos)); append_Ncomp();
        }

        std::ofstream file("Ar0n_timings.json");
        file << outputs;
    }
    return EXIT_SUCCESS;
}
