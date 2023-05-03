#include "teqp/derivs.hpp"

#include "teqpcpp.cpp"
using MI = teqp::cppinterface::ModelImplementer;

#include "teqp/cpp/deriv_adapter.hpp"
using namespace teqp::cppinterface;

// Derivatives from isochoric thermodynamics (all have the same signature)
#define X(f) \
double MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        return DerivativeAdapter(model).f(T, rhovec); \
    }, m_model); \
}
ISOCHORIC_double_args
#undef X

#define X(f) \
EArrayd MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        return DerivativeAdapter(model).f(T, rhovec); \
    }, m_model); \
}
ISOCHORIC_array_args
#undef X

#define X(f) \
EMatrixd MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        return DerivativeAdapter(model).f(T, rhovec); \
    }, m_model); \
}
ISOCHORIC_matrix_args
#undef X

