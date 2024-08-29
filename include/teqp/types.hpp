#pragma once

#include <vector>
#include "Eigen/Dense"

namespace teqp {
    template<typename T> inline T pow2(const T& x) { return x * x; }
    template<typename T> inline T pow3(const T& x) { return x * x * x; }

    inline auto toeig(const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); }
}

// Everything inside this NO_TYPES_HEADER section involves slow and large headers
// so if you are just defining types (but not their implementations), you can avoid the 
// compilation speed hit invoked by these headers
#if !defined(NO_TYPES_HEADER)

#if defined(TEQP_MULTICOMPLEX_ENABLED)
#include "MultiComplex/MultiComplex.hpp"
#endif

#include <valarray>
#include <chrono>

#if defined(TEQP_MULTIPRECISION_ENABLED)
#include "boost/multiprecision/cpp_bin_float.hpp"
#endif

#include "teqp/exceptions.hpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

namespace teqp {

    // Registration of types that are considered to be containers
    // See https://stackoverflow.com/a/12045843
    template <typename Container>
    struct is_container : std::false_type { };
    template <typename... Ts> struct is_container<std::vector<Ts...> > : std::true_type { };
    template <typename... Ts> struct is_container<std::valarray<Ts...> > : std::true_type { };
    // Missing Eigen::Array and Eigen::Matrix types here

    template <typename T> struct is_eigen_impl : std::false_type {};
    template <typename T, int... Is> struct is_eigen_impl<Eigen::Matrix<T, Is...>> : std::true_type {};
    template <typename T, int... Is> struct is_eigen_impl<Eigen::Array<T, Is...>> : std::true_type {};

    template<typename T>
    auto forceeval(T&& expr)
    {
        using namespace autodiff::detail;
        if constexpr (isReal<T>) {
            return expr;
        }
        else if constexpr (isDual<T> || isExpr<T>) {
            return autodiff::detail::eval(expr);
        }
        else {
            return expr;
        }
    }
    
    /// A constexpr function for ensuring that an argument to a function is NOT an expr,
    /// which can have surprising behavior
    template<typename T>
    void error_if_expr(T&& /*expr*/)
    {
        using namespace autodiff::detail;
        if constexpr (isExpr<T>) {
            static_assert(true, "Argument to function is an expression, but should not be");
        }
    }

    // See https://stackoverflow.com/a/41438758
    template<typename T> struct is_complex_t : public std::false_type {};
    template<typename T> struct is_complex_t<std::complex<T>> : public std::true_type {};

    // See https://stackoverflow.com/a/41438758
    template<typename T> struct is_mcx_t : public std::false_type {};
#if defined(TEQP_MULTICOMPLEX_ENABLED)    
    template<typename T> struct is_mcx_t<mcx::MultiComplex<T>> : public std::true_type {};
#endif

    // Extract the underlying value from more complicated numerical types, like complex step types with
    // a tiny increment in the imaginary direction
    template<typename T>
    auto getbaseval(const T& expr)
    {
        using namespace autodiff::detail;
        if constexpr (isDual<T> || isExpr<T> || isReal<T>) {
            return autodiff::detail::val(expr);
        }
        else if constexpr (is_complex_t<T>()) {
            return expr.real();
        }
        else if constexpr (is_mcx_t<T>()) {
#if defined(TEQP_MULTIPRECISION_ENABLED)
            // Argument is a multicomplex of a boost multiprecision
            using contained_type = decltype(expr.real());
            if constexpr (boost::multiprecision::is_number<contained_type>()) {
                return static_cast<double>(expr.real());
            }
            else {
                return expr.real();
            }
#else
            return expr.real();
#endif
        }
#if defined(TEQP_MULTIPRECISION_ENABLED)
        else if constexpr (boost::multiprecision::is_number<T>()) {
            return static_cast<double>(expr);
        }
#endif
        else {
            return expr;
        }
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
            T xrecip = 1.0/x;
            return powi(xrecip, -n);
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

    template<typename T>
    inline auto powIVd(const T& x, const Eigen::ArrayXd& e) {
        return e.cast<T>().unaryExpr([&x](const auto& e_) {return forceeval(pow(x, e_)); }).eval();
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
#if defined(TEQP_MULTICOMPLEX_ENABLED)
    template<typename T>
    auto pow(const mcx::MultiComplex<T>& x, const Eigen::ArrayXd& e) {
        Eigen::Array<mcx::MultiComplex<T>, Eigen::Dynamic, 1> o(e.size());
        for (auto i = 0; i < e.size(); ++i) {
            o[i] = pow(x, e[i]);
        }
        return o;
    }
    template<typename T>
    auto pow(const Eigen::ArrayX<mcx::MultiComplex<T>>& x, const int& e) {
        auto y = x;
        for (auto i = 0; i < x.size(); ++i) {
            y[i] = powi(x[i], e);
        }
        return y;
    }
#endif

    /**
     \brief Take the dot-product of two vector-like objects that have contiguous memory and support the .size() method
     
     This allows to mix and match 1D Eigen arrays and vectors and STL vectors
     */
    auto contiguous_dotproduct(const auto& x, const auto&y){
        using x_t = std::decay_t<decltype(x[0])>;
        using y_t = std::decay_t<decltype(y[0])>;
        using ret_t = std::common_type_t<x_t, y_t>;
        if (static_cast<long>(x.size()) != y.size()){
            throw teqp::InvalidArgument("Arguments to contiguous_dotproduct are not the same size");
        }
        ret_t summer = 0.0;
        for (auto i = 0U; i < x.size(); ++i){
            summer += x[i]*y[i];
        }
        return summer;
//        auto get_ptr = [](auto& x){
//            using x_t = std::decay_t<decltype(x[0])>;
//            const x_t* x_0_ptr;
//            if constexpr (is_eigen_impl<decltype(x)>::value){
//                x_0_ptr = x.data();
//            }
//            else{
//                x_0_ptr = &(x[0]);
//            }
//            return x_0_ptr;
//        };
//        // element-wise product and then a sum
//        return (
//            Eigen::Map<const Eigen::ArrayX<x_t>>(get_ptr(x), x.size()).template cast<ret_t>()
//            * Eigen::Map<const Eigen::ArrayX<y_t>>(get_ptr(y), y.size()).template cast<ret_t>()
//        ).sum();
    }

}; // namespace teqp

#endif
