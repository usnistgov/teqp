#include <tuple>
#include <valarray>
#include <iostream>

#include "teqp/core.hpp"

void trace() { 
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    vdWEOS<double> vdW(Tc_K, pc_Pa);
    auto Zc = 3.0/8.0;
    auto rhoc0 = pc_Pa[0]/(vdW.R*Tc_K[0]) / Zc; 
    auto T = Tc_K[0];
    const auto dT = 1;
    std::valarray<double> rhovec = { rhoc0, 0.0 };
    auto tic0 = std::chrono::steady_clock::now();
    for (auto iter = 0; iter < 1000; ++iter){
        auto drhovecdT = get_drhovec_dT_crit(vdW, T, rhovec);
        rhovec += drhovecdT*dT;
        T += dT;
        int rr = 0;
        auto z0 = rhovec[0] / rhovec.sum();
        std::cout << z0 <<  "," << rhovec[0] << "," << T << "," << get_splus(vdW, T, rhovec) << std::endl;
        if (z0 < 0) {
            break;
        }
    }
    auto tic1 = std::chrono::steady_clock::now();
    std::cout << T << " " << rhovec[1] << std::endl;
    std::cout << std::chrono::duration<double>(tic1 - tic0).count() << " s to trace" << std::endl;

    int rr = 0;
}
int main() {   
    trace();
    return EXIT_SUCCESS;
}