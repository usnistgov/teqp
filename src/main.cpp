#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <chrono>
#include <iomanip>

#include "teqp/derivs.hpp"
#include "teqp/models/eos.hpp"
#include "teqp/models/pcsaft.hpp"
//
//void test_vdwMix() {
//    // Argon + Xenon
//    std::valarray<double> Tc_K = { 150.687, 289.733 };
//    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
//    vdWEOS<double> vdW(Tc_K, pc_Pa);
//
//    double T = 298.15;
//    auto rho = 3.0;
//    auto R = get_R_gas<double>();
//    auto rhotot = rho;
//
//    const auto rhovec = (Eigen::ArrayXd(2) << rho/2, rho/2).finished();
//
//    auto fPsir = [&vdW](const auto& T, const auto& rhovec) {
//        using container = decltype(rhovec);
//        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
//        auto molefrac = rhovec/rhotot_;
//        return vdW.alphar(T, rhotot_, molefrac)*vdW.R*T*rhotot_;
//    };
//    auto Psir = fPsir(T, rhovec);
//    auto dPsirdrho0 = rhovec[0]*derivrhoi(fPsir, T, rhovec, 0);
//    auto dPsirdrho1 = rhovec[1]*derivrhoi(fPsir, T, rhovec, 1);
//    auto pfromderiv = rho*R*T - Psir + dPsirdrho0 + dPsirdrho1;
//    using id = IsochoricDerivatives<decltype(vdW)>;
//    auto sr = id::get_splus(vdW, T, rhovec);
//
//    auto t2 = std::chrono::steady_clock::now();
//    auto pfromderiv3 = rhotot*R*T + id::get_pr(vdW, T, rhovec);
//    auto t3 = std::chrono::steady_clock::now();
//    std::cout << std::chrono::duration<double>(t3 - t2).count() << " from isochoric (mix) " << std::endl;
//
//}

template<typename T1, typename T2, typename T3>
void f(const T1& v1, const T2& v2, const T3& v3) {
    using t = decltype(forceeval(v1* v2* v3[0]));
    std::cout << "Hi";

}

int main(){
    //test_vdwMix();   
    
    //autodiff::dual1st x;
    //double y;
    //Eigen::VectorX<double> z;
    //f(x, y, z);
    ////using t = decltype(forceeval(x * y * z[0]));
    ////t f = "";
    
    std::vector<std::string> names = { "Methane", "Ethane" };
    PCSAFTMixture mix(names);
    mix.print_info();
    using id = IsochoricDerivatives<decltype(mix)>;
    using vd = VirialDerivatives<decltype(mix)>;
    double T = 300;
    const auto rhovec = (Eigen::ArrayXd(2) << 1.0, 2.0).finished();
    const Eigen::ArrayXd molefrac = (rhovec/rhovec.sum()).eval();
    const double rho = rhovec.sum();
    double a00csd = get_Ar01<ADBackends::complex_step>(mix, T, rho, molefrac);
    double a00cx = get_Ar01<ADBackends::multicomplex>(mix, T, rho, molefrac);
    double a00ad = get_Ar01<ADBackends::autodiff>(mix, T, rho, molefrac);
    double a00iso = id::get_Ar01(mix, T, rhovec);
    double apriso = id::get_pr(mix, T, rhovec);
    double B2 = vd::get_B2vir(mix, T, molefrac);
    double B12 = vd::get_B12vir(mix, T, molefrac);
    

    return EXIT_SUCCESS;
}