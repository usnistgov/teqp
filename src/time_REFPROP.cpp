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
    char herr[255], hfld[10000] = "KRYPTON", hhmx[255] = "HMX.BNC", href[4] = "DEF";
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
        double z[20] = {1.0}, p = -1;

        // Random speed testing
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
        int itau = 0, idelta = 0;
        for (auto repeat = 0; repeat < 100; ++repeat) {
            std::valarray<double> ps = 0.0 * taus;
            double rhoc = 10139.128;
            double Tc = 190.564;
            auto tic = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < taus.size(); ++i) {
                double Ar01 = -10000;
                PHIXdll(itau, idelta, taus[i], deltas[i], z, Ar01); ps[i] = Ar01; //ps[i] = (Ar01 + 1)*deltas[i]*rhoc*8.31451*Tc/taus[i];
                /*idelta = 1;
                PHIXdll(itau, idelta, Ts[i], ds[i], z, p); ps[i] += p;
                idelta = 2;
                PHIXdll(itau, idelta, Ts[i], ds[i], z, p); ps[i] += p;
                idelta = 3;
                PHIXdll(itau, idelta, Ts[i], ds[i], z, p); ps[i] += p;*/
                //PRESSdll(Ts[i], ds[i], z, p); ps[i] = p;
            }
            auto toc = std::chrono::high_resolution_clock::now();
            double elap_us = std::chrono::duration<double>(toc - tic).count() / taus.size() * 1e6;
            std::cout << "Repeat " << std::to_string(repeat) << ": " << elap_us << " us/call for PRESSdll; " << std::accumulate(std::begin(ps), std::end(ps), 0.0)/ps.size() << std::endl;
        }

        // And the same example with teqp
        auto model = build_multifluid_model({ "Krypton" }, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");
        auto N = taus.size();
        auto c = (Eigen::ArrayXd(1) << 1.0).finished();

        // Get reference to pure fluid model
        const auto f = model.corr.get_EOS(0);
        for (auto counter = 0; counter < 100; ++counter)
        {
            double o = 0.0;
            Timer t(N);
            for (auto j = 0; j < N; ++j) {
                o += f.alphar(taus[j], deltas[j]);
            }
            std::cout << o/N << std::endl;
        }
    }
    return EXIT_SUCCESS;
}