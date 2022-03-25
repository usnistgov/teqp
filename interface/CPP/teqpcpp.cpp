#include "teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

double teqp::cppinterface::get_Arxy(const teqp::AllowedModels&model, const int NT, const int ND, const double T, const double rho, const std::valarray<double> &molefracs){
	// Make an Eigen view of the double buffer
    Eigen::Map<const Eigen::ArrayXd> molefrac_(&(molefracs[0]), molefracs.size());

    // Now call the visitor function to get the value
    return std::visit([&](const auto& model) {
        using tdx = teqp::TDXDerivatives<decltype(model), double, decltype(molefrac_)>;
        return tdx::get_Ar(NT, ND, model, T, rho, molefrac_);
    }, model);
}

nlohmann::json teqp::cppinterface::trace_critical_arclength_binary(const teqp::AllowedModels& model, const double T0, const std::valarray<double> &rhovec0) {
    // Make an Eigen view of the double buffer
    Eigen::Map<const Eigen::ArrayXd> rhovec0_(&(rhovec0[0]), rhovec0.size());

    // Now call the visitor function to get the value
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, Eigen::ArrayXd>;
        return crit::trace_critical_arclength_binary(model, T0, rhovec0_, "");
    }, model);
}