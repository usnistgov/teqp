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

    nlohmann::json outputs = nlohmann::json::array();

    double T = 300, D_moldm3 = 3, D_molm3 = D_moldm3*1e3;
    int N = 1000000;
    std::vector<std::string> component_list = { "Methane","Ethane","n-Propane","n-Butane","n-Pentane","n-Hexane" };
    {
        for (int Ncomp : {1, 2, 3, 4, 5, 6}) {
            
            std::vector<std::string> fluid_set(component_list.begin(), component_list.begin() + Ncomp);
            auto one_teqp = [&]()
            {
                // teqp!!
                auto model = build_multifluid_model(fluid_set, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");
                using id = IsochoricDerivatives<decltype(model), double>;
                auto rhovec = (D_molm3*Eigen::ArrayXd::Ones(Ncomp)/Ncomp).eval();
                auto tic = std::chrono::high_resolution_clock::now(); 
                double usummer = 0.0;
                for (auto j = 0; j < N; ++j) {
                    auto val = id::template get_fugacity_coefficients(model, T, rhovec);
                    usummer += val.sum();
                }
                auto toc = std::chrono::high_resolution_clock::now();
                double elap_us = std::chrono::duration<double>(toc - tic).count() / N * 1e6;
                std::cout << elap_us << " us/call for fugacity coefficient w/ " << Ncomp << " component(s) with value " << usummer << std::endl;
                return nlohmann::json{ {"val",usummer/N},{"time",elap_us},{"model","teqp"}, {"Ncomp",Ncomp} };
            };
            auto one_REFPROP = [&](){
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
                        return nlohmann::json{ {"err", herr} };
                    }
                }
                std::valarray<double> z(20); z = 0.0; z[std::slice(0, Ncomp, 1)] = 1.0 / Ncomp;
                std::valarray<double> u(20); u = 0.0;
                auto usummer = 0.0;
                int ierr = 0; char herr[256];
                auto tic = std::chrono::high_resolution_clock::now();
                for (auto j = 0; j < N; ++j) {
                    FUGCOFdll(T, D_moldm3, &(z[0]), &(u[0]), ierr, herr, 255);
                    //if (ierr != 0) { std::cout << ierr << ": " << herr << std::endl; }
                    usummer += std::valarray<double>(u[std::slice(0, Ncomp, 1)]).sum();
                }
                auto toc = std::chrono::high_resolution_clock::now();
                double elap_us = std::chrono::duration<double>(toc - tic).count() / N * 1e6;
                std::cout << elap_us << " us/call for FUGCOF w/ " << Ncomp << " component(s) with value " << usummer << std::endl;
                return nlohmann::json{ {"val",usummer/N},{"time",elap_us},{"model","REFPROP"}, {"Ncomp",Ncomp} };
            };

            outputs.push_back(one_teqp());
            outputs.push_back(one_REFPROP());
        }
        std::ofstream file("fugcoeff_timings.json");
        file << outputs;
    }
    return EXIT_SUCCESS;
}