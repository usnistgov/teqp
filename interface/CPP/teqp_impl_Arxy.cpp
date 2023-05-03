#include "teqp/derivs.hpp"

#include "teqpcpp.cpp"
using MI = teqp::cppinterface::ModelImplementer;

#include "teqp/cpp/deriv_adapter.hpp"
using namespace teqp::cppinterface;

// Here XMacros are used to create functions like get_Ar00, get_Ar01, ....
#define X(i,j) \
double MI::get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefracs) const { \
    return std::visit([&](const auto& model) { \
        return DerivativeAdapter(model).get_Ar ## i ## j(T, rho, molefracs); \
    }, m_model); \
}
ARXY_args
#undef X

// Here XMacros are used to create functions like get_Ar01n, get_Ar02n, ....
#define X(i) \
EArrayd MI::get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefracs) const { \
    return std::visit([&](const auto& model) -> EArrayd { \
        return DerivativeAdapter(model).get_Ar0 ## i ## n(T, rho, molefracs); \
    }, m_model); \
}
AR0N_args
#undef X

double MI::get_Arxy(const int NT, const int ND, const double T, const double rho, const EArrayd& molefracs) const {
    return std::visit([&](const auto& model) {
        return DerivativeAdapter(model).get_Arxy(NT, ND, T, rho, molefracs);
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
