#include "teqp/derivs.hpp"

#include "teqpcpp.cpp"
using MI = teqp::cppinterface::ModelImplementer;

// Here XMacros are used to create functions like get_Ar00, get_Ar01, ....
#define X(i,j) \
double MI::get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefracs) const { \
    return std::visit([&](const auto& model) { \
        using tdx = teqp::TDXDerivatives<decltype(model), double, REArrayd>; \
        return tdx::template get_Arxy<i,j>(model, T, rho, molefracs); \
    }, m_model); \
}
ARXY_args
#undef X

// Here XMacros are used to create functions like get_Ar01n, get_Ar02n, ....
#define X(i) \
EArrayd MI::get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefracs) const { \
    return std::visit([&](const auto& model) -> EArrayd { \
        using tdx = teqp::TDXDerivatives<decltype(model), double, REArrayd>; \
        auto vals = tdx::template get_Ar0n<i>(model, T, rho, molefracs); \
        return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size()); \
    }, m_model); \
}
AR0N_args
#undef X
