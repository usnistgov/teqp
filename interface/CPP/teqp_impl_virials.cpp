#include "teqp/derivs.hpp"
#include "teqpcpp.cpp"

using MI = teqp::cppinterface::ModelImplementer;

double MI::get_B2vir(const double T, const EArrayd& molefrac) const  {
    return std::visit([&](const auto& model) {
        using vd = VirialDerivatives<decltype(model), double, RAX>;
        return vd::get_B2vir(model, T, molefrac);
    }, m_model);
}
double MI::get_B12vir(const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        using vd = VirialDerivatives<decltype(model), double, RAX>;
        return vd::get_B12vir(model, T, molefrac);
    }, m_model);
}
std::map<int, double> MI::get_Bnvir(const int Nderiv, const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        using vd = VirialDerivatives<decltype(model), double, RAX>;
        return vd::get_Bnvir_runtime(Nderiv, model, T, molefrac);
    }, m_model);
}
double MI::get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        using vd = VirialDerivatives<decltype(model), double, RAX>;
        return vd::get_dmBnvirdTm_runtime(Nderiv, NTderiv, model, T, molefrac);
    }, m_model);
}
