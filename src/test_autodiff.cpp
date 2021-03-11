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

auto build_simple() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    auto R = 1.380649e-23 * 6.02214076e23; ///< Exact value, given by k_B*N_A
    int i = 0;
    double ai = 27.0 / 64.0 * pow(R * Tc_K[i], 2) / pc_Pa[i];
    double bi = 1.0 / 8.0 * R * Tc_K[i] / pc_Pa[i];
    return vdWEOS1(ai, bi);
}
auto build_vdW() {
    // Argon + Xenon
    std::valarray<double> Tc_K = { 150.687, 289.733 };
    std::valarray<double> pc_Pa = { 4863000.0, 5842000.0 };
    return vdWEOS(Tc_K, pc_Pa);
}

template<typename Model>
void test_autodiff(Model model) {
    
    double T = 298.15;
    auto rho = 3.0;
    auto rhotot = rho;
    const std::valarray<double> rhovec = { rhotot / 2, rhotot / 2 };

    int Nrep = 10000;

    auto ticn1 = std::chrono::steady_clock::now();
    for (int i = 0; i < Nrep; ++i) {
        volatile double rr = model.alphar(T+i*1e-16, rhovec);
    }

    auto tic0 = std::chrono::steady_clock::now();

    // autodiff derivatives
    for (int i = 0; i < Nrep; ++i) {
        autodiff::dual4th varT = T + i*1e-16;
        auto f = [&model, &rhovec](auto& T) {return eval(model.alphar(T, rhovec)); };
        auto [alphar, dalphardT, d2alphardT2, d3, d4] = derivatives(f, wrt(varT, varT, varT, varT), at(varT));
    }
    auto tic1 = std::chrono::steady_clock::now();

    // complex step (first) derivative
    constexpr double h = 1e-100;
    for (int i = 0; i < Nrep; ++i){
        volatile auto dalphardT_comstep = model.alphar(std::complex<double>(T+i*1e-16,h), rhovec).imag()/h;
    }
    auto tic2 = std::chrono::steady_clock::now();

    // Multicomplex derivatives
    for (int i = 0; i < Nrep; ++i) {
    auto diffs = diff_mcx1<double>([&model, &rhovec](auto& T) {return model.alphar(T, rhovec); }, T + i * 1e-16, 4, true);
    }
    auto tic3 = std::chrono::steady_clock::now();

    /*std::cout << (dalphardT-dalphardT_comstep)/dalphardT << " diff, rel (first deriv)" << std::endl;
    std::cout << (d2alphardT2 - diffs[2])/diffs[2] << " diff, rel (second deriv)" << std::endl;*/

    std::cout << std::chrono::duration<double>(tic0 - ticn1).count()/Nrep*1e6 << " us (function evaluation in double)" << std::endl; 
    std::cout << std::chrono::duration<double>(tic1 - tic0).count()/Nrep*1e6 << " us (autodiff)" << std::endl;
    std::cout << std::chrono::duration<double>(tic2 - tic1).count()/Nrep*1e6 << " us (CSD)" << std::endl;
    std::cout << std::chrono::duration<double>(tic3 - tic2).count()/Nrep*1e6 << " us (MCX)" << std::endl;

    // Test evaluation of Hessian of Psir
    dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
    VectorXdual2nd g;
    VectorXdual2nd rhovecc(2); rhovecc << rhovec[0], rhovec[1];
    auto hfunc = [&model, &T](const VectorXdual2nd& rho_) {
        auto rhotot_ = std::accumulate(std::begin(rho_), std::end(rho_), (decltype(rho_[0]))0.0);
        return eval(model.alphar(T, rho_)*model.R*T*rhotot_);
    };
    Eigen::MatrixXd H = autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g); // evaluate the function value u, its gradient, and its Hessian matrix H
    std::cout << H;

    auto ffff = 0;
}

int main() {
    test_autodiff(build_simple());
    test_autodiff(build_vdW());
    return EXIT_SUCCESS;
}