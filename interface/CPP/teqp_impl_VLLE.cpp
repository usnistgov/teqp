#include "teqpcpp.cpp"
#include "teqp/algorithms/VLLE.hpp"

using namespace teqp;
using MI = teqp::cppinterface::ModelImplementer;

std::tuple<VLLE::VLLE_return_code,EArrayd,EArrayd,EArrayd> MI::mix_VLLE_T(const double T, const REArrayd& rhovecVinit, const REArrayd& rhovecL1init, const REArrayd& rhovecL2init, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const{
    return std::visit([&](const auto& model) {
        return VLLE::mix_VLLE_T(model, T, rhovecVinit, rhovecL1init, rhovecL2init, atol, reltol, axtol, relxtol, maxiter);
    }, m_model);
}

std::vector<nlohmann::json> MI::find_VLLE_T_binary(const std::vector<nlohmann::json>& traces, const std::optional<VLLE::VLLEFinderOptions> options) const{
    return std::visit([&](const auto& model) {
        return VLLE::find_VLLE_T_binary(model, traces, options);
    }, m_model);
}
