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