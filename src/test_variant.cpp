#include <vector>
#include <iostream>
#include <complex>
#include <variant>
#include <chrono>
#include <map>
#include <valarray>

#include "teqp/types.hpp"
#include "nlohmann/json.hpp"
#include "teqp/models/eos.hpp"
//#include "teqp/models/CPA.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/containers.hpp"
#include <Eigen/Dense>

class Timer {
private:
    int N;
    decltype(std::chrono::steady_clock::now()) tic;
public:
    Timer(int N) : N(N), tic(std::chrono::steady_clock::now()){}
    ~Timer() {
        auto elap = std::chrono::duration<double>(std::chrono::steady_clock::now()-tic).count();
        std::cout << elap/N*1e6 << " us/call" << std::endl;
    }
};

template <typename ModelContainer, typename T1, typename T2, typename T3> 
auto get_f(const ModelContainer &modcon, const T1& x1, const T2& x2, const T3& x3){
    // The output type is the type promotion of T, rho, and molefrac
    std::common_type_t<T1, T2, decltype(x3[0])> result = -1;
    // Call the function with T, rho, molefrac arguments
    std::visit([&](auto&& model) { result = model.alphar(x1, x2, x3); }, modcon);
    return result;
}

int main(){
    // Here, all models that can be stored in this container are defined. Types may
    // be obtained from factory functions without explicit definition
    using MFType = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));
    ModelContainer<MFType, vdWEOS<double>, vdWEOS1> mc;
    nlohmann::json vdWargs = { { "a",3 }, { "b", 4 } };
    std::string vdWs = vdWargs.dump();
    for (auto i = 0; i < 10; ++i) {
        mc.add_model(vdWEOS1(1, 2));
    }
    const auto& v = mc.get_ref<vdWEOS1>(1);

    auto c = (Eigen::ArrayXd(2) << 3.0, 3.0).finished();
    int N = 1000;
    volatile double x1 = 3.0;
    std::complex<double> x2(3.0, 1.0);
    vdWEOS1 b1(3, 4);
    for (auto i = 0; i < mc.size(); ++i) {
        Timer t(N);
        for (auto j = 0; j < N; ++j) {
            volatile auto v = b1.alphar(x1, x2, c);
            //std::cout << v << std::endl;
        }
    }
    std::cout << "----" << std::endl;
    for (auto i = 0; i < mc.size(); ++i) {
        const auto& m = mc.get_model(1);
        Timer t(N); 
        for (auto j = 0; j < N; ++j) {
            auto v = get_f(m, x1, x2, c);
            //std::cout << v << std::endl;
        }
    }
    std::cout << "----" << std::endl;
    for (auto i = 1; i <= mc.size(); ++i) {
        Timer t(N);
        for (auto j = 0; j < N; ++j) {
            auto v = mc.get_alphar(i, x1, x2, c);
            //std::cout << v << std::endl;
        }
    }
}