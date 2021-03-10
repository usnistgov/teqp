#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <complex>

#include "autodiff/forward.hpp"
#include "autodiff/reverse.hpp"

/* A (very) simple implementation of the van der Waals EOS*/
class vdWEOSSimple {
private:
    double a, b;
public:
    vdWEOSSimple(double a, double b) : a(a), b(b) {};

    const double R = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A

    template<typename TType, typename RhoType>
    auto alphar(const TType &T, const RhoType& rho) const -> TType{
        auto rhotot = std::accumulate(std::begin(rho), std::end(rho), static_cast<typename RhoType::value_type>(0.0));
        auto Psiminus = -log(1.0 - b * rhotot);
        auto Psiplus = rhotot;
        return Psiminus - a / (R * T) * Psiplus;
    }
};

void test_vdW_autodiff() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    
    double T = 298.15;
    auto rho = 3.0;
    auto R = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A
    auto rhotot = rho;
    const std::valarray<double> rhovec = { rhotot / 2, rhotot / 2 };
    
    int i = 0;
    double ai = 27.0/64.0*pow(R*Tc_K[i], 2)/pc_Pa[i];
    double bi = 1.0/8.0*R*Tc_K[i]/pc_Pa[i];
    vdWEOSSimple vdW(ai, bi);

    autodiff::dual varT {T};
    auto u = vdW.alphar(varT, rhovec);
    int rr = 0;
    auto dalphardT = derivative([&vdW, &rhovec](auto& T) {return vdW.alphar(T, rhovec); }, wrt(varT), at(varT));

    double h = 1e-100;
    auto dalphardT_comstep = vdW.alphar(std::complex<double>(T,h), rhovec).imag()/h;
    std::cout << dalphardT-dalphardT_comstep << " diff, absolute" << std::endl;

}

int main() {
    test_vdW_autodiff();
    return EXIT_SUCCESS;
}