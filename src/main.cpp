#include "teqp/core.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <chrono>
#include <iomanip>

void test_vdW() {
    volatile double T = 298.15;
    auto rho = 3.0;
    auto R = get_R_gas<double>();

    double Omega_b = 1.0 / 8, Omega_a = 27.0 / 64;
    double Tcrit = 150.687, pcrit = 4863000.0; // Argon
    double b = Omega_b * R * Tcrit / pcrit;
    double ba = Omega_b / Omega_a / Tcrit / R;
    double a = b / ba;

    auto vdW = vdWEOS1(a, b);

    auto t2 = std::chrono::steady_clock::now();
    volatile auto pp = vdW.p(T, 1 / rho);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t3 - t2).count() << " from p(T,v)" << std::endl;

    const std::valarray<double> rhovec = { rho, 0.0 };

    auto t21 = std::chrono::steady_clock::now();
    auto pfromderiv = rho*R*T + get_pr(vdW, T, rhovec);
    auto t31 = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration<double>(t31 - t21).count() << " from isochoric" << std::endl;
    auto err = pfromderiv / pp - 1.0;
    std::cout << "Error (fractional): " << err << std::endl;

    std::valarray<double> molefrac = {1.0};
    double B2 = get_B2vir(vdW, T, molefrac);
    double B2exact = b-a/(R*T);

    auto Nvir = 8;
    auto Bn = get_Bnvir(vdW, Nvir, T, molefrac);
    // Exact solutions for virial coefficients for van der Waals 
    auto get_vdW_exacts = [a,b,R,T](int Nmax){
        std::map<int, double> o = {{2, b - a / (R * T)}};
        for (auto i = 3; i <= Nmax; ++i) {
            o[i] = pow(b, i-1);
        }
        return o;
    };
    auto Bnexact = get_vdW_exacts(Nvir);
    for (auto i = 2; i <= Nvir; ++i){
        std::cout << std::scientific << i << ", " << Bnexact[i] << ", " << Bn[i] << ", " << std::abs(Bnexact[i]-Bn[i]) << std::endl;
    }
    int rr = 0;

}

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

    {
    auto t1 = std::chrono::steady_clock::now();
    volatile auto dT1 = derivT(fPsir, T, rhovec);
    auto t2 = std::chrono::steady_clock::now();
    volatile auto dT2 = derivTmcx(fPsir, T, rhovec);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count() << " with complex step" << std::endl; 
    std::cout << std::chrono::duration<double>(t3 - t2).count() << " with multicomplex " << std::endl;
    }
}

int main(){
    test_vdW();
    test_vdwMix();
    return EXIT_SUCCESS;
}