#include "teqp/derivs.hpp"
#include "teqpcpp.cpp"
#include "teqp/cpp/deriv_adapter.hpp"

using MI = teqp::cppinterface::ModelImplementer;
using namespace teqp::cppinterface;

double MI::get_B2vir(const double T, const EArrayd& molefrac) const  {
    return std::visit([&](const auto& model) {
        return DerivativeAdapter(model).get_B2vir(T, molefrac);
    }, m_model);
}
double MI::get_B12vir(const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        return DerivativeAdapter(model).get_B12vir(T, molefrac);
    }, m_model);
}
std::map<int, double> MI::get_Bnvir(const int Nderiv, const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        return DerivativeAdapter(model).get_Bnvir(Nderiv, T, molefrac);
    }, m_model);
}
double MI::get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const {
    return std::visit([&](const auto& model) {
        return DerivativeAdapter(model).get_dmBnvirdTm(Nderiv, NTderiv, T, molefrac);
    }, m_model);
}
