#include <Eigen/Dense>
#include <tuple>
#include <valarray>
#include <iostream>

#include "teqp/core.hpp"
#include "teqp/critical_tracing.hpp"

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
    int rr = 0;
}
int main() {   
    trace();
    return EXIT_SUCCESS;
}