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
    char herr[255], hfld[10000] = "METHANE", hhmx[255] = "HMX.BNC", href[4] = "DEF";
    SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);

    if (ierr != 0) printf("This ierr: %d herr: %s\n", ierr, herr);
    {
        double d = 1.0;
        double z[20] = {1.0}, p = -1;

        // Random speed testing
        std::default_random_engine re;        
        std::valarray<double> Ts(100000);
        {
            std::uniform_real_distribution<double> unif(89, 360);
            std::transform(std::begin(Ts), std::end(Ts), std::begin(Ts), [&unif, &re](double x) { return unif(re); });
        }
        std::valarray<double> ds(Ts.size()); {
            std::uniform_real_distribution<double> unif(1, 1.001);
            std::transform(std::begin(ds), std::end(ds), std::begin(ds), [&unif, &re](double x) { return unif(re); }); 
        }
        int itau = 0, idelta = 0;
        for (auto repeat = 0; repeat < 10; ++repeat) {
            std::valarray<double> ps = 0.0 * Ts;
            auto tic = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < Ts.size(); ++i) {
                PHIXdll(itau, idelta, Ts[i], ds[i], z, p); ps[i] = p;
            }
            auto toc = std::chrono::high_resolution_clock::now();
            double elap_us = std::chrono::duration<double>(toc - tic).count() / Ts.size() * 1e6;
            std::cout << elap_us << " us/call for PRESSdll; " << std::accumulate(std::begin(ps), std::end(ps), 0.0) << std::endl;
        }
    }
    return EXIT_SUCCESS;
}