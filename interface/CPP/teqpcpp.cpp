// Pulls in the AllowedModels variant
#include "teqp/models/fwd.hpp"

#include "teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

double teqp::cppinterface::get_Ar01(const void* modelptr, const double T, const double rho, const Eigen::ArrayXd& molefracs) {
    const teqp::AllowedModels& model = *static_cast<const teqp::AllowedModels*>(modelptr);
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, std::decay_t<decltype(molefracs)>>;
        return tdx::get_Arxy<0, 1, ADBackends::autodiff>(model, T, rho, molefracs);
        }, model);
}

double teqp::cppinterface::get_Arxy(const void* modelptr, const int NT, const int ND, const double T, const double rho, const Eigen::ArrayXd &molefracs){
    const teqp::AllowedModels& model = *static_cast<const teqp::AllowedModels*>(modelptr);
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, std::decay_t<decltype(molefracs)>>;
        return tdx::get_Ar(NT, ND, model, T, rho, molefracs);
    }, model);
}

nlohmann::json teqp::cppinterface::trace_critical_arclength_binary(const void* modelptr, const double T0, const Eigen::ArrayXd &rhovec0) {
    const teqp::AllowedModels& model = *static_cast<const teqp::AllowedModels*>(modelptr);
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec0)>>;
        return crit::trace_critical_arclength_binary(model, T0, rhovec0, "");
    }, model);
}