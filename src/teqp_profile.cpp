#include <vector>
#include <iostream>
#include <complex>
#include <variant>
#include <chrono>

#include "teqp/types.hpp"
#include "nlohmann/json.hpp"
#include "teqp/models/eos.hpp"
//#include "teqp/models/CPA.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/containers.hpp"
#include <Eigen/Dense>
#include "teqp/derivs.hpp"

class Timer {
private:
    int N;
    decltype(std::chrono::steady_clock::now()) tic;
public:
    Timer(int N) : N(N), tic(std::chrono::steady_clock::now()) {}
    ~Timer() {
        auto elap = std::chrono::duration<double>(std::chrono::steady_clock::now() - tic).count();
        std::cout << elap / N * 1e6 << " us/call" << std::endl;
    }
};

template <typename ModelContainer, typename T1, typename T2, typename T3>
auto get_f(const ModelContainer& modcon, const T1& x1, const T2& x2, const T3& x3) {
    // Call the function with T, rho, molefrac arguments
    return std::visit([&](auto&& model) { return model.alphar(x1, x2, x3); }, modcon);
}

int main() {
    // Here, all models that can be stored in this container are defined. Types may
    // be obtained from factory functions without explicit definition
    using MFType = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));
    ModelContainer<vdWEOS1, vdWEOS<double>, MFType, PCSAFTMixture> mc;
    std::vector<int> indices;

    auto adder = [&](auto&&m) { indices.push_back(mc.add_model(m)); };

    adder(vdWEOS1(1, 2));
    adder(vdWEOS<double>({ 150.687 }, { 4863000.0 })); 
    adder(PCSAFTMixture({ "Methane" }));
    adder(build_multifluid_model({ "Methane" }, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json"));

    double x1 = 3.0;
    double x2 = 2.0; 
    auto c = (Eigen::ArrayXd(1) << 1.0).finished();
    int N = 100000;

    auto f_alphar = [&x1, &x2, &c](auto& model) { return model.alphar(x1, x2, c); };
    //auto alphar = mc.caller(1, f_alphar);
    //auto meta = mc.caller(1, [&x1, &x2, &c](auto& model) { return model.get_meta(); })

    auto m = mc.get_ref<MFType>(4);
    m.alphar(x2, x2, c);
    {
        Timer t(N);
        for (auto j = 0; j < N; ++j) {
            auto v = m.alphar(x1, x2, c);
            //std::cout << v << std::endl;
        }
    }
    for (auto counter = 0; counter < 10; ++counter){
        Timer t(N);
        for (auto j = 0; j < N; ++j) {
            x1 += j * 1e-10;
            //auto f_Ar01 = [&x1, &x2, &c](auto& model) { using tdx = TDXDerivatives<decltype(model), decltype(x1), decltype(c)>;  return tdx::get_Ar02(model, x1, x2, c); };
            auto v = mc.caller(4, [&x1, &x2, &c](auto& model) { using tdx = TDXDerivatives<decltype(model), decltype(x1), decltype(c)>;  return tdx::get_Ar0n<3>(model, x1, x2, c); });
            //std::cout << v << std::endl;
        }
    }

    ////std::complex<double> x2(3.0, 1.0);
    //std::cout << "----" << std::endl;
    //for (auto i : indices) {
    //    const auto& m = mc.get_model(i);
    //    Timer t(N);
    //    for (auto j = 0; j < N; ++j) {
    //        auto v = get_f(m, x1, x2, c);
    //        //std::cout << v << std::endl;
    //    }
    //}
    //std::cout << "----" << std::endl;
    //for (auto i : indices) {
    //    Timer t(N);
    //    for (auto j = 0; j < N; ++j) {
    //        auto v = mc.caller(i, f_alphar);
    //        //std::cout << v << std::endl;
    //    }
    //    auto f_alphar = [&x1, &x2, &c](auto& model) { return model(x1, x2, c); };
    //}
}