#include "teqp/models/fwd.hpp"

#include "teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

double teqp::cppinterface::get_Arxy(const teqp::AllowedModels& model, const int NT, const int ND, const double T, const double rho, const Eigen::ArrayXd &molefracs){
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, std::decay_t<decltype(molefracs)>>;
        return tdx::template get_Ar(NT, ND, model, T, rho, molefracs);
    }, model);
}

nlohmann::json teqp::cppinterface::trace_critical_arclength_binary(const teqp::AllowedModels& model, const double T0, const Eigen::ArrayXd &rhovec0) {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec0)>>;
        return crit::trace_critical_arclength_binary(model, T0, rhovec0, "");
    }, model);
}