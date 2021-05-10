#pragma once

#include <vector>
#include <valarray>

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