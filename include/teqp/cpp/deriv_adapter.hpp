#pragma once

#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"

namespace teqp{
namespace cppinterface{
namespace adapter{

template<typename ModelType>
struct Owner{
    const ModelType model;
    Owner(ModelType&& m) : model(m) {};
};

template<typename ModelType>
struct ConstViewer{
    const ModelType& model;
    ConstViewer(ModelType& m) : model(m) {};
};

namespace internal{
    template<class T>struct tag{using type=T;};
}

/**
 This class holds a const reference to a class, and exposes an interface that matches that used in AbstractModel
 
 The exposed methods cover all the derivative methods that are obtained by derivatives of the model
 */
template<typename ModelPack>
class DerivativeAdapter : public teqp::cppinterface::AbstractModel{
public:
    const ModelPack mp;
    
    template<typename T>
    DerivativeAdapter(internal::tag<T> tag_, const T&& mp): mp(mp) {} ;
    
//    template<typename T>
//    DerivativeAdapter(const Owner<T>&& mp): mp(mp) {} ;
//
//    template<typename T>
//    DerivativeAdapter(const ConstViewer<T>&& mp): mp(mp) {} ;
    
    const AllowedModels& get_model() const override { throw teqp::NotImplementedError(""); };
    AllowedModels& get_mutable_model() override { throw teqp::NotImplementedError(""); };
    
    virtual double get_R(const EArrayd& molefrac) const override {
        return mp.model.R(molefrac);
    };
    
    virtual double get_Arxy(const int NT, const int ND, const double T, const double rhomolar, const EArrayd& molefrac) const override{
        return TDXDerivatives<decltype(mp.model), double, EArrayd>::get_Ar(NT, ND, mp.model, T, rhomolar, molefrac);
    };
    
    // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
#define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefrac) const  override { return TDXDerivatives<decltype(mp.model), double, EArrayd>::template get_Arxy<i,j>(mp.model, T, rho, molefrac); };
    ARXY_args
#undef X
    // And like get_Ar01n, get_Ar02n, ....
#define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefrac) const  override { auto vals = TDXDerivatives<decltype(mp.model), double, EArrayd>::template get_Ar0n<i>(mp.model, T, rho, molefrac); return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size()); };
    AR0N_args
#undef X
    
    // Virial derivatives
    virtual double get_B2vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.model), double, EArrayd>::get_B2vir(mp.model, T, z);
    };
    virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.model), double, EArrayd>::get_Bnvir_runtime(Nderiv, mp.model, T, z);
    };
    virtual double get_B12vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.model), double, EArrayd>::get_B12vir(mp.model, T, z);
    };
    virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const override {
        return VirialDerivatives<decltype(mp.model), double, EArrayd>::get_dmBnvirdTm_runtime(Nderiv, NTderiv, mp.model, T, molefrac);
    };
    
    // Derivatives from isochoric thermodynamics (all have the same signature within each block), and they differ by their output argument
#define X(f) virtual double f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.model), double, EArrayd>::f(mp.model, T, rhovec); };
    ISOCHORIC_double_args
#undef X
#define X(f) virtual EArrayd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.model), double, EArrayd>::f(mp.model, T, rhovec); };
    ISOCHORIC_array_args
#undef X
#define X(f) virtual EMatrixd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.model), double, EArrayd>::f(mp.model, T, rhovec); };
    ISOCHORIC_matrix_args
#undef X
#define X(f) virtual std::tuple<double, Eigen::ArrayXd, Eigen::MatrixXd> f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.model), double, EArrayd>::f(mp.model, T, rhovec); };
    ISOCHORIC_multimatrix_args
#undef X
    virtual Eigen::ArrayXd get_Psir_sigma_derivs(const double T, const EArrayd& rhovec, const EArrayd& v) const override{
        return IsochoricDerivatives<decltype(mp.model), double, EArrayd>::get_Psir_sigma_derivs(mp.model, T, rhovec, v);
    };
    
    virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z ) const override { throw teqp::NotImplementedError("Not available"); };
    
//    virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>&, const std::optional<TCABOptions> & = std::nullopt) const override { throw teqp::NotImplementedError("Not available"); };
//    virtual EArrayd get_drhovec_dT_crit(const double T, const REArrayd& rhovec) const  override { throw teqp::NotImplementedError("Not available"); };
//    virtual double get_dp_dT_crit(const double T, const REArrayd& rhovec) const override { throw teqp::NotImplementedError("Not available"); };
//    virtual EArray2 get_criticality_conditions(const double T, const REArrayd& rhovec) const  override { throw teqp::NotImplementedError("Not available"); };
//    virtual EigenData eigen_problem(const double T, const REArrayd& rhovec, const std::optional<REArrayd>&) const override { throw teqp::NotImplementedError("Not available"); };
//    virtual double get_minimum_eigenvalue_Psi_Hessian(const double T, const REArrayd& rhovec) const override { throw teqp::NotImplementedError("Not available"); };
    
};

template<typename TemplatedModel> auto view(const TemplatedModel& tp){
    ConstViewer cv{tp};
    return new DerivativeAdapter<decltype(cv)>(internal::tag<decltype(cv)>{}, std::move(cv));
}
template<typename TemplatedModel> auto own(const TemplatedModel&& tp){
    Owner o(std::move(tp));
    return new DerivativeAdapter<decltype(o)>(internal::tag<decltype(o)>{}, std::move(o));
}

template<typename TemplatedModel> auto make_owned(const TemplatedModel& tmodel){
    using namespace teqp::cppinterface;
    return std::unique_ptr<AbstractModel>(own(std::move(tmodel)));
};

}
}
}
