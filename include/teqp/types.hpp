#pragma once

#include "nlohmann/json.hpp"

#include <vector>
#include <valarray>
#include <set>

// Registration of types that are considered to be containers
// See https://stackoverflow.com/a/12045843
template <typename Container>
struct is_container : std::false_type { };
template <typename... Ts> struct is_container<std::vector<Ts...> > : std::true_type { };
template <typename... Ts> struct is_container<std::valarray<Ts...> > : std::true_type { };
// Missing Eigen::Array and Eigen::Matrix types here

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

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
    else {
        return expr;
    }
}

auto toeig = [](const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); };

auto all_same_length = [](const nlohmann::json& j, const std::vector<std::string>& ks) {
    std::set<int> lengths;
    for (auto k : ks) { lengths.insert(j[k].size()); }
    return lengths.size() == 1;
};
