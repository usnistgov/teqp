#include <iostream>
#include <algorithm>
#include <numeric>
#include <valarray>
#include <complex>
#include "teqp/models/eos.hpp"

#include "MultiComplex/MultiComplex.hpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

template<typename Model>
void test_autodiff(Model model) {
    
    double T = 298.15;
    double rho = 3.0;
    auto rhotot = rho;
    const std::valarray<double> rhovec = { rhotot / 2, rhotot / 2 };
    const std::valarray<double> molefrac = { 0.5, 0.5 };
    double v1, v2, v3;

    int Nrep = 10000;

    auto ticn1 = std::chrono::steady_clock::now();
    for (int i = 0; i < Nrep; ++i) {
        volatile double rr = model.alphar(T+i*1e-16, rho, molefrac);
    }
    auto tic0 = std::chrono::steady_clock::now();

    // autodiff derivatives
    for (int i = 0; i < Nrep; ++i) {
        autodiff::dual4th varT = static_cast<double>(T);
        auto f = [&model, &rho, &molefrac](auto& T) {return eval(model.alphar(T, rho, molefrac)); };
        auto [alphar, dalphardT,d2,d3,d4] = derivatives(f, wrt(varT), at(varT));
        v1 = dalphardT;
    }
    auto tic1 = std::chrono::steady_clock::now();

    // complex step (first) derivative
    constexpr double h = 1e-100;
    for (int i = 0; i < Nrep; ++i){
        volatile auto dalphardT_comstep = model.alphar(std::complex<double>(T,h), rho, molefrac).imag()/h;
        v2 = dalphardT_comstep;
    }
    auto tic2 = std::chrono::steady_clock::now();

    // Multicomplex derivatives
    for (int i = 0; i < Nrep; ++i) {
        volatile auto diffs = mcx::diff_mcx1<double>([&model, &rho, &molefrac](auto& T) {return model.alphar(T, rho, molefrac); }, T, 1, true)[1];
        v3 = diffs;
    }
    auto tic3 = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration<double>(tic0 - ticn1).count()/Nrep*1e6 << " us (function evaluation in double)" << std::endl; 
    std::cout << std::chrono::duration<double>(tic1 - tic0).count()/Nrep*1e6 << " us (autodiff)" << std::endl;
    std::cout << std::chrono::duration<double>(tic2 - tic1).count()/Nrep*1e6 << " us (CSD)" << std::endl;
    std::cout << std::chrono::duration<double>(tic3 - tic2).count()/Nrep*1e6 << " us (MCX)" << std::endl;

    std::cout << v1 << "," << v2 << "," << v3 << std::endl;

    std::cout << build_Psir_Hessian_mcx(model, T, rhovec) << std::endl; 
    std::cout << build_Psir_Hessian_autodiff(model, T, rhovec) << std::endl;

    auto ffff = 0;
}

int main() {
    test_autodiff(build_simple());
    test_autodiff(build_vdW());
    return EXIT_SUCCESS;
}