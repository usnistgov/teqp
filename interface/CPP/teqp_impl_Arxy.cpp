#include "teqp/derivs.hpp"

#include "teqpcpp.cpp"
using MI = teqp::cppinterface::ModelImplementer;

// Derivatives from isochoric thermodynamics (all have the same signature)
#define X(f) \
double MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        using id = IsochoricDerivatives<decltype(model), double, EArrayd>; \
        return id::f(model, T, rhovec); \
    }, m_model); \
}
ISOCHORIC_double_args
#undef X

#define X(f) \
EArrayd MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        using id = IsochoricDerivatives<decltype(model), double, EArrayd>; \
        return id::f(model, T, rhovec); \
    }, m_model); \
}
ISOCHORIC_array_args
#undef X

#define X(f) \
EMatrixd MI::f(const double T, const EArrayd& rhovec) const  { \
    return std::visit([&](const auto& model) { \
        using id = IsochoricDerivatives<decltype(model), double, EArrayd>; \
        return id::f(model, T, rhovec); \
    }, m_model); \
}
ISOCHORIC_matrix_args
#undef X

double MI::get_Arxy(const int NT, const int ND, const double T, const double rho, const EArrayd& molefracs) const {
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, EArrayd>;
        return tdx::template get_Ar(NT, ND, model, T, rho, molefracs);
    }, m_model);
}

double MI::get_neff(const double T, const double rho, const EArrayd& molefracs) const {
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, EArrayd>;
        return tdx::template get_neff(model, T, rho, molefracs);
    }, m_model);
}

EArray33d MI::get_deriv_mat2(const double T, double rho, const EArrayd& z) const {
    return std::visit([&](const auto& model) {
        // Although the template argument suggests that only residual terms
        // are returned, also the ideal-gas ones are returned because the
        // ideal-gas term is required to implement alphar which just redirects
        // to alphaig
        return DerivativeHolderSquare<2, AlphaWrapperOption::residual>(model, T, rho, z).derivs;
    }, m_model);
}
