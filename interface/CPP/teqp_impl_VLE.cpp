#include "teqpcpp.cpp"
#include "teqp/algorithms/VLE.hpp"

using namespace teqp;
using MI = teqp::cppinterface::ModelImplementer;

std::tuple<EArrayd, EArrayd> MI::get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
    return std::visit([&](const auto& model) {
        return teqp::get_drhovecdp_Tsat(model, T, rhovecL, rhovecV);
    }, m_model);
}
std::tuple<EArrayd, EArrayd> MI::get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
    return std::visit([&](const auto& model) {
        return teqp::get_drhovecdT_psat(model, T, rhovecL, rhovecV);
    }, m_model);
}
double MI::get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const {
    return std::visit([&](const auto& model) {
        return teqp::get_dpsat_dTsat_isopleth(model, T, rhovecL, rhovecV);
    }, m_model);
}

nlohmann::json MI::trace_VLE_isotherm_binary(const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<TVLEOptions> &options) const{
    return std::visit([&](const auto& model) {
        return teqp::trace_VLE_isotherm_binary(model, T0, rhovecL0, rhovecV0, options);
    }, m_model);
}
nlohmann::json MI::trace_VLE_isobar_binary(const double p, const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<PVLEOptions> &options) const{
    return std::visit([&](const auto& model) {
        return teqp::trace_VLE_isobar_binary(model, p, T0, rhovecL0, rhovecV0, options);
    }, m_model);
}
std::tuple<VLE_return_code,EArrayd,EArrayd> MI::mix_VLE_Tx(const double T, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const REArrayd& xspec, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const{
    return std::visit([&](const auto& model) {
        return teqp::mix_VLE_Tx(model, T, rhovecL0, rhovecV0, xspec, atol, reltol, axtol, relxtol, maxiter);
    }, m_model);
}
MixVLEReturn MI::mix_VLE_Tp(const double T, const double pgiven, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLETpFlags> &flags) const{
    return std::visit([&](const auto& model) {
        return teqp::mix_VLE_Tp(model, T, pgiven, rhovecL0, rhovecV0, flags);
    }, m_model);
}
std::tuple<VLE_return_code,double,EArrayd,EArrayd> MI::mixture_VLE_px(const double p_spec, const REArrayd& xmolar_spec, const double T0, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLEpxFlags>& flags ) const{
    return std::visit([&](const auto& model) {
        return teqp::mixture_VLE_px(model, p_spec, xmolar_spec, T0, rhovecL0, rhovecV0, flags);
    }, m_model);
}

