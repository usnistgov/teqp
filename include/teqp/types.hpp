#pragma once

#include "nlohmann/json.hpp"

#include "MultiComplex/MultiComplex.hpp"

#include <vector>
#include <valarray>
#include <set>
#include <chrono>

#include "boost/multiprecision/cpp_bin_float.hpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include "Eigen/Dense"

namespace teqp {

    // Registration of types that are considered to be containers
    // See https://stackoverflow.com/a/12045843
    template <typename Container>
    struct is_container : std::false_type { };
    template <typename... Ts> struct is_container<std::vector<Ts...> > : std::true_type { };
    template <typename... Ts> struct is_container<std::valarray<Ts...> > : std::true_type { };
    // Missing Eigen::Array and Eigen::Matrix types here

    template<typename T>
    auto forceeval(T&& expr)
    {
        using namespace autodiff::detail;
        if constexpr (isDual<T> || isExpr<T>) {
            return eval(expr);
        }
        else {
            return expr;
        }
    }

    // See https://stackoverflow.com/a/41438758
    template<typename T> struct is_complex_t : public std::false_type {};
    template<typename T> struct is_complex_t<std::complex<T>> : public std::true_type {};

    // See https://stackoverflow.com/a/41438758
    template<typename T> struct is_mcx_t : public std::false_type {};
    template<typename T> struct is_mcx_t<mcx::MultiComplex<T>> : public std::true_type {};

    template<typename T>
    auto getbaseval(const T& expr)
    {
        using namespace autodiff::detail;
        if constexpr (isDual<T> || isExpr<T>) {
            return val(expr);
        }
        else if constexpr (is_complex_t<T>()) {
            return expr.real();
        }
        else if constexpr (is_mcx_t<T>()) {
            return expr.real();
        }
        else if constexpr (boost::multiprecision::is_number<T>()) {
            return static_cast<double>(expr);
        }
        else {
            return expr;
        }
    }

    inline auto toeig (const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); }

    inline auto all_same_length(const nlohmann::json& j, const std::vector<std::string>& ks) {
        std::set<decltype(j[0].size())> lengths;
        for (auto k : ks) { lengths.insert(j[k].size()); }
        return lengths.size() == 1;
    }

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

    /// From Ulrich Deiters
    template <typename T>                             // arbitrary integer power
    T powi(const T& x, int n) {
        switch (n) {
        case 0:
            return static_cast<T>(1.0);                       // x^0 = 1 even for x == 0
        case 1:
            return static_cast<T>(x);
        case 2:
            return static_cast<T>(x * x);
        case 3:
            return static_cast<T>(x * x * x);
        case 4:
            auto x2 = x * x;
            return static_cast<T>(x2 * x2);
        }
        if (n < 0) {
            using namespace autodiff::detail;
            if constexpr (isDual<T> || isExpr<T>) {
                return eval(powi(eval(1.0 / x), -n));
            }
            else {
                return powi(static_cast<T>(1.0) / x, -n);
            }
        }
        else {
            T y(x), xpwr(x);
            n--;
            while (n > 0) {
                if (n % 2 == 1) {
                    y = y * xpwr;
                    n--;
                }
                xpwr = xpwr * xpwr;
                n /= 2;
            }
            return y;
        }
    }

    template<typename T>
    inline auto powIVi(const T& x, const Eigen::ArrayXi& e) {
        //return e.binaryExpr(e.cast<T>(), [&x](const auto&& a_, const auto& e_) {return static_cast<T>(powi(x, a_)); });
        static Eigen::Array<T, Eigen::Dynamic, 1> o;
        o.resize(e.size());
        for (auto i = 0; i < e.size(); ++i) {
            o[i] = powi(x, e[i]);
        }
        return o;
        //return e.cast<T>().unaryExpr([&x](const auto& e_) {return powi(x, e_); }).eval();
    }

    //template<typename T>
    //auto powIV(const T& x, const Eigen::ArrayXd& e) {
    //    Eigen::Array<T, Eigen::Dynamic, 1> o = e.cast<T>();
    //    return o.unaryExpr([&x](const auto& e_) {return powi(x, e_); } ).eval();
    //}

    inline auto pow(const double& x, const double& e) {
        return std::pow(x, e);
    }

    inline auto pow(const double& x, const int& e) {
        return powi(x, e);
    }

    template<typename T>
    auto pow(const std::complex<T>& x, const Eigen::ArrayXd& e) {
        Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> o(e.size());
        for (auto i = 0; i < e.size(); ++i) {
            o[i] = pow(x, e[i]);
        }
        return o;
    }

    template<typename T>
    auto pow(const mcx::MultiComplex<T>& x, const Eigen::ArrayXd& e) {
        Eigen::Array<mcx::MultiComplex<T>, Eigen::Dynamic, 1> o(e.size());
        for (auto i = 0; i < e.size(); ++i) {
            o[i] = pow(x, e[i]);
        }
        return o;
    }

}; // namespace teqp