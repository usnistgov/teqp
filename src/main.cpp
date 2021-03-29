#include "teqp/core.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <chrono>
#include <iomanip>

void test_vdwMix() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    vdWEOS<double> vdW(Tc_K, pc_Pa);

    volatile double T = 298.15;
    auto rho = 3.0;
    auto R = get_R_gas<double>();
    auto rhotot = rho;

    const std::valarray<double> rhovec = { rho/2, rho/2 };

    auto fPsir = [&vdW](const auto& T, const auto& rhovec) {
        using container = decltype(rhovec);
        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        auto molefrac = rhovec/rhotot_;
        return vdW.alphar(T, rhotot_, molefrac)*vdW.R*T*rhotot_;
    };
    auto Psir = fPsir(T, rhovec);
    auto dPsirdrho0 = rhovec[0]*derivrhoi(fPsir, T, rhovec, 0);
    auto dPsirdrho1 = rhovec[1]*derivrhoi(fPsir, T, rhovec, 1);
    auto pfromderiv = rho*R*T - Psir + dPsirdrho0 + dPsirdrho1;
    auto sr = get_splus(vdW, T, rhovec);

    auto t2 = std::chrono::steady_clock::now();
    auto pfromderiv3 = rhotot*R*T + get_pr(vdW, T, rhovec);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t3 - t2).count() << " from isochoric (mix) " << std::endl;

}

int main(){
    test_vdwMix();
    return EXIT_SUCCESS;
}