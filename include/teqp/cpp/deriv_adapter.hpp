#pragma once

#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"

namespace teqp{
namespace cppinterface{

/**
 This class holds a const reference to a class, and exposes an interface that matches that used in AbstractModel
 
 The exposed methods cover all the derivative methods that are obtained by derivatives of the model
 */
template<typename ModelType>
class DerivativeAdapter : public teqp::cppinterface::AbstractModel{
public:
    const ModelType&  model;
    DerivativeAdapter(const ModelType& model): model(model) {} ;
    const AllowedModels& get_model() const override { throw teqp::NotImplementedError(""); };
    AllowedModels& get_mutable_model() override { throw teqp::NotImplementedError(""); };
    
    virtual double get_R(const EArrayd& molefrac) const override {
        return model.R(molefrac);
    };
    
    virtual double get_Arxy(const int NT, const int ND, const double T, const double rhomolar, const EArrayd& molefrac) const override{
        return TDXDerivatives<ModelType, double, EArrayd>::get_Ar(NT, ND, model, T, rhomolar, molefrac);
    };
    
    // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
    #define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefrac) const  override { return TDXDerivatives<ModelType, double, EArrayd>::template get_Arxy<i,j>(model, T, rho, molefrac); };
        ARXY_args
    #undef X
    // And like get_Ar01n, get_Ar02n, ....
    #define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefrac) const  override { auto vals = TDXDerivatives<ModelType, double, EArrayd>::template get_Ar0n<i>(model, T, rho, molefrac); return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size()); };
        AR0N_args
    #undef X
    
    // Virial derivatives
    virtual double get_B2vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<ModelType, double, EArrayd>::get_B2vir(model, T, z);
    };
    virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const override {
        return VirialDerivatives<ModelType, double, EArrayd>::get_Bnvir_runtime(Nderiv, model, T, z);
    };
    virtual double get_B12vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<ModelType, double, EArrayd>::get_B12vir(model, T, z);
    };
    virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const override {
        return VirialDerivatives<decltype(model), double, EArrayd>::get_dmBnvirdTm_runtime(Nderiv, NTderiv, model, T, molefrac);
    };
    
    // Derivatives from isochoric thermodynamics (all have the same signature within each block), and they differ by their output argument
    #define X(f) virtual double f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(model), double, EArrayd>::f(model, T, rhovec); };
        ISOCHORIC_double_args
    #undef X
    #define X(f) virtual EArrayd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(model), double, EArrayd>::f(model, T, rhovec); };
        ISOCHORIC_array_args
    #undef X
    #define X(f) virtual EMatrixd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(model), double, EArrayd>::f(model, T, rhovec); };
        ISOCHORIC_matrix_args
    #undef X
    #define X(f) virtual std::tuple<double, Eigen::ArrayXd, Eigen::MatrixXd> f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(model), double, EArrayd>::f(model, T, rhovec); };
        ISOCHORIC_multimatrix_args
    #undef X
    
    virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z ) const override { throw teqp::NotImplementedError("Not available"); };
    
    virtual std::tuple<EArrayd, EArrayd> get_drhovecdp_Tsat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override { throw teqp::NotImplementedError("Not available"); };
    virtual std::tuple<EArrayd, EArrayd> get_drhovecdT_psat(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override { throw teqp::NotImplementedError("Not available"); };
    virtual double get_dpsat_dTsat_isopleth(const double T, const REArrayd& rhovecL, const REArrayd& rhovecV) const override { throw teqp::NotImplementedError("Not available"); };
    virtual nlohmann::json trace_VLE_isotherm_binary(const double T0, const EArrayd& rhovec0, const EArrayd& rhovecV0, const std::optional<TVLEOptions> &) const override { throw teqp::NotImplementedError("Not available"); };
    virtual nlohmann::json trace_VLE_isobar_binary(const double p, const double T0, const EArrayd& rhovecL0, const EArrayd& rhovecV0, const std::optional<PVLEOptions> &) const override { throw teqp::NotImplementedError("Not available"); };
    virtual std::tuple<VLE_return_code,EArrayd,EArrayd> mix_VLE_Tx(const double T, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const REArrayd& xspec, const double atol, const double reltol, const double axtol, const double relxtol, const int maxiter) const override { throw teqp::NotImplementedError("Not available"); };
    virtual MixVLEReturn mix_VLE_Tp(const double T, const double pgiven, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLETpFlags> &flags) const override { throw teqp::NotImplementedError("Not available"); };
    virtual std::tuple<VLE_return_code,double,EArrayd,EArrayd> mixture_VLE_px(const double p_spec, const REArrayd& xmolar_spec, const double T0, const REArrayd& rhovecL0, const REArrayd& rhovecV0, const std::optional<MixVLEpxFlags>& flags) const override { throw teqp::NotImplementedError("Not available"); };
    
    virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>&, const std::optional<TCABOptions> & = std::nullopt) const override { throw teqp::NotImplementedError("Not available"); };
    virtual EArrayd get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const  override { throw teqp::NotImplementedError("Not available"); };
    virtual double get_dp_dT_crit(const double T, const REArrayd& rhovec) const override { throw teqp::NotImplementedError("Not available"); };
    virtual EArray2 get_criticality_conditions(const double T, const REArrayd& rhovec) const  override { throw teqp::NotImplementedError("Not available"); };
    virtual EigenData eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>&) const override { throw teqp::NotImplementedError("Not available"); };
    virtual double get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const override { throw teqp::NotImplementedError("Not available"); };
    
};

}
}
