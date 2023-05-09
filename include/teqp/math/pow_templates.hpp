#pragma once
#include "teqp/types.hpp"

namespace teqp{
template<typename A> auto POW2(const A& x) { return forceeval(x*x); }
template<typename A> auto POW3(const A& x) { return forceeval(POW2(x)*x); }
template<typename A> auto POW4(const A& x) { return forceeval(POW2(x)*POW2(x)); }
template<typename A> auto POW5(const A& x) { return forceeval(POW2(x)*POW3(x)); }
template<typename A> auto POW7(const A& x) { return forceeval(POW2(x)*POW5(x)); }
template<typename A> auto POW8(const A& x) { return forceeval(POW4(x)*POW4(x)); }
template<typename A> auto POW10(const A& x) { return forceeval(POW2(x)*POW8(x)); }
template<typename A> auto POW12(const A& x) { return forceeval(POW4(x)*POW8(x)); }
}
