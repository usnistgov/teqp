#include "teqp/algorithms/critical_tracing.hpp"
#include "teqpcpp.cpp"

using namespace teqp;
using MI = teqp::cppinterface::ModelImplementer;

nlohmann::json MI::trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>& filename, const std::optional<TCABOptions> &options) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec0)>>;
        return crit::trace_critical_arclength_binary(model, T0, rhovec0, filename , options);
    }, m_model);
}
EArrayd MI::get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_drhovec_dT_crit(model, T, rhovec);
    }, m_model);
}
double MI::get_dp_dT_crit(const double T, const REArrayd& rhovec) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_dp_dT_crit(model, T, rhovec);
    }, m_model);
}
EArray2 MI::get_criticality_conditions(const double T, const REArrayd& rhovec) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_criticality_conditions(model, T, rhovec);
    }, m_model);
}
EigenData MI::eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>& alignment_v0) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec)>>;
        return crit::eigen_problem(model, T, rhovec, alignment_v0.value_or(Eigen::ArrayXd()));
    }, m_model);
}
double MI::get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const {
    return std::visit([&](const auto& model) {
        using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec)>>;
        return crit::get_minimum_eigenvalue_Psi_Hessian(model, T, rhovec);
    }, m_model);
}
