#pragma once

#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"

namespace teqp{
namespace cppinterface{
namespace adapter{

/// The ownership wrapper of a model
/// \note Takes ownership, so the argument passed to the constructor is invalidated
template<typename ModelType>
struct Owner{
private:
    ModelType model;
public:
    auto& get_ref(){ return model; };
    const auto& get_cref() const { return model; };
    const std::type_index index;
    Owner(ModelType&& m) : model(m), index(std::type_index(typeid(ModelType))) {};
};

/// The ownership wrapper of a model
/// \note Takes ownership, so the argument passed to the constructor is invalidated
template<typename ModelType>
struct ConstViewer{
private:
    const ModelType& model;
public:
    auto& get_ref(){ return model; };
    const auto& get_cref() const { return model; };
    const std::type_index index;
    ConstViewer(ModelType& m) : model(m), index(std::type_index(typeid(ModelType))) {};
};

namespace internal{
    template<class T>struct tag{using type=T;};
}

/**
 This class holds a const reference to a class, and exposes an interface that matches that used in AbstractModel
 
 The exposed methods cover all the derivative methods that are obtained by derivatives of the model, other methods in the AbstractModel implementation can then call methods implemented in this class
 
 \note This technique is known as type-erasure in C++
 */
template<typename ModelPack>
class DerivativeAdapter : public teqp::cppinterface::AbstractModel{
private:
    ModelPack mp;
public:
    auto& get_ModelPack_ref(){ return mp; }
    const auto& get_ModelPack_cref() const { return mp; }
    
    template<typename T>
    DerivativeAdapter(internal::tag<T> /*tag_*/, const T&& mp): mp(mp) {} ;
    
    const std::type_index& get_type_index() const override {
        return mp.index;
    };
    
//    template<typename T>
//    DerivativeAdapter(const Owner<T>&& mp): mp(mp) {} ;
//
//    template<typename T>
//    DerivativeAdapter(const ConstViewer<T>&& mp): mp(mp) {} ;
    
    virtual double get_R(const EArrayd& molefrac) const override {
        return mp.get_cref().R(molefrac);
    };
    
    virtual double get_Arxy(const int NT, const int ND, const double T, const double rhomolar, const EArrayd& molefrac) const override{
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_Ar(NT, ND, mp.get_cref(), T, rhomolar, molefrac);
    };
    
    // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
#define X(i,j) virtual double get_Ar ## i ## j(const double T, const double rho, const REArrayd& molefrac) const  override { return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::template get_Arxy<i,j>(mp.get_cref(), T, rho, molefrac); };
    ARXY_args
#undef X
    // And like get_Ar01n, get_Ar02n, ....
#define X(i) virtual EArrayd get_Ar0 ## i ## n(const double T, const double rho, const REArrayd& molefrac) const  override { auto vals = TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::template get_Ar0n<i>(mp.get_cref(), T, rho, molefrac); return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size()); };
    AR0N_args
#undef X
    
    // Virial derivatives
    virtual double get_B2vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_B2vir(mp.get_cref(), T, z);
    };
    virtual std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_Bnvir_runtime(Nderiv, mp.get_cref(), T, z);
    };
    virtual double get_B12vir(const double T, const EArrayd& z) const override {
        return VirialDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_B12vir(mp.get_cref(), T, z);
    };
    virtual double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const override {
        return VirialDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_dmBnvirdTm_runtime(Nderiv, NTderiv, mp.get_cref(), T, molefrac);
    };
    
    // Composition derivatives with temperature and density as the working variables
    virtual double get_ATrhoXi(const double T, const int NT, const double rhomolar, const int ND, const EArrayd& molefrac, const int i, const int NXi) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_ATrhoXi_runtime(mp.get_cref(), T, NT, rhomolar, ND, molefrac, i, NXi);
    };
    virtual double get_ATrhoXiXj(const double T, const int NT, const double rhomolar, const int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_ATrhoXiXj_runtime(mp.get_cref(), T, NT, rhomolar, ND, molefrac, i, NXi, j, NXj);
    };
    virtual double get_ATrhoXiXjXk(const double T, const int NT, const double rhomolar, const int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_ATrhoXiXjXk_runtime(mp.get_cref(), T, NT, rhomolar, ND, molefrac, i, NXi, j, NXj, k, NXk);
    };
    
    // Derivatives from isochoric thermodynamics (all have the same signature within each block), and they differ by their output argument
#define X(f) virtual double f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.get_cref()), double, EArrayd>::f(mp.get_cref(), T, rhovec); };
    ISOCHORIC_double_args
#undef X
#define X(f) virtual EArrayd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.get_cref()), double, EArrayd>::f(mp.get_cref(), T, rhovec); };
    ISOCHORIC_array_args
#undef X
#define X(f) virtual EMatrixd f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.get_cref()), double, EArrayd>::f(mp.get_cref(), T, rhovec); };
    ISOCHORIC_matrix_args
#undef X
#define X(f) virtual std::tuple<double, Eigen::ArrayXd, Eigen::MatrixXd> f(const double T, const EArrayd& rhovec) const override { return IsochoricDerivatives<decltype(mp.get_cref()), double, EArrayd>::f(mp.get_cref(), T, rhovec); };
    ISOCHORIC_multimatrix_args
#undef X
    virtual Eigen::ArrayXd get_Psir_sigma_derivs(const double T, const EArrayd& rhovec, const EArrayd& v) const override{
        return IsochoricDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_Psir_sigma_derivs(mp.get_cref(), T, rhovec, v);
    };
    
    virtual EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z ) const override {
        return DerivativeHolderSquare<2>(mp.get_cref(), T, rho, z).derivs;
    };
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

template<typename TemplatedModel> auto make_cview(const TemplatedModel& tmodel){
    using namespace teqp::cppinterface;
    return std::unique_ptr<AbstractModel>(view(tmodel));
};

/**
 \brief Get a const reference to the model that is being held in a DerivativeAdapter instance
 
 \note Available for both ownership and const viewer holder types.
 */
template<typename ModelType>
const ModelType& get_model_cref(const AbstractModel *am)
{
    if (am == nullptr){
        throw teqp::InvalidArgument("Argument to get_model_cref is a nullptr");
    }
    const auto* mptr = dynamic_cast<const DerivativeAdapter<ConstViewer<const ModelType>>*>(am);
    const auto* mptr2 = dynamic_cast<const DerivativeAdapter<Owner<const ModelType>>*>(am);
    if (mptr != nullptr){
        return mptr->get_ModelPack_cref().get_cref();
    }
    else if (mptr2 != nullptr){
        return mptr2->get_ModelPack_cref().get_cref();
    }
    else{
        throw teqp::InvalidArgument("Unable to cast model to desired type");
    }
}

/**
 \brief Get a mutable reference to the model
 
 \note Only available when the holder type is ownership (not available for const viewer holder type)
 */
template<typename ModelType>
ModelType& get_model_ref(AbstractModel *am)
{
    if (am == nullptr){
        throw teqp::InvalidArgument("Argument to get_model_ref is a nullptr");
    }
    auto* mptr2 = dynamic_cast<DerivativeAdapter<Owner<ModelType>>*>(am);
    if (mptr2 != nullptr){
        return mptr2->get_ModelPack_ref().get_ref();
    }
    else{
        throw teqp::InvalidArgument("Unable to cast model to desired type; only the Owner ownership model is allowed");
    }
}

}
}
}
