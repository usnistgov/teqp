#pragma once

#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"

#if defined(TEQP_MULTIPRECISION_ENABLED)
// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;
#include "teqp/finite_derivs.hpp"
#endif

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

/// The viewer wrapper of a model
/// \note Does not take ownership, so the argument passed to the constructor must have a longer lifespan than this class
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

/**
 \brief A collection type that allows you to sum contributions from multiple EOS terms
 
  Each term in the summation must support the alphar generic interface that accepts T, rho, molefrac as arguments
 
 The gas constant method is used of the FIRST function in the instance
 
 \note Some information used from https://stackoverflow.com/a/40749139 regarding the tuple summation
 */
template<typename... Funcs>
class OwnershipSummer{
public:
    std::tuple<Funcs...> contributions;
    OwnershipSummer(Funcs && ...f) : contributions(std::forward<Funcs>(f)...){};
    
    auto& get_ref(){ return *this; };
    const auto& get_cref() const { return *this; };
    const std::type_index index;
    
    /// The gas constant, obtained from the first model in the tuple
    template<typename MoleFracType>
    auto R(const MoleFracType& molefrac){
        return std::get<0>(contributions).R(molefrac);
    }
    
    /// The generic alphar function, which sums the contributions coming from the individual models passed into the constructor
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T, const RhoType& rhomolar, const MoleFracType& molefrac) const {
        auto sum_func = [&T, &rhomolar, &molefrac](auto const&... e)->decltype(auto) {
            return (e.alphar(T, rhomolar, molefrac)+...);
        };
        return std::apply(sum_func, contributions);
    }
};

template <typename... Args>
OwnershipSummer<Args...> make_OwnershipSummer(Args&&... args)
{
    return OwnershipSummer<Args...>(std::forward<Args>(args)...);
}

namespace internal{
    template<class T>struct tag{using type=T;};
}

template<typename T, typename U>
concept CallableReducingDensity = requires(T t, U u) {
    { t.get_reducing_density(u) };
};

template<typename T, typename U>
concept CallableReducingTemperature = requires(T t, U u) {
    { t.get_reducing_temperature(u) };
};

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
    DerivativeAdapter(internal::tag<T> /*tag_*/, const T&& mp): mp(mp) {} 
    
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
    // And like get_Ar10n, get_Ar20n, ....
#define X(i) virtual EArrayd get_Ar ## i ## 0n(const double T, const double rho, const REArrayd& molefrac) const  override { auto vals = TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::template get_Arn0<i>(mp.get_cref(), T, rho, molefrac); return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size()); };
    ARN0_args
#undef X
    
    virtual double get_Ar01ep(const double T, const double rho, const EArrayd& molefrac) const  override {
        using namespace boost::multiprecision;
        using my_float_t = number<cpp_bin_float<100U>>;
        auto f = [&](const auto& rhoep){
            return mp.get_cref().alphar(T, rhoep, molefrac);
        };
        return rho*static_cast<double>(centered_diff<1,4>(f, static_cast<my_float_t>(rho), 1e-16*static_cast<my_float_t>(rho)));
    }
    virtual double get_Ar02ep(const double T, const double rho, const EArrayd& molefrac) const  override {
        using namespace boost::multiprecision;
        using my_float_t = number<cpp_bin_float<100U>>;
        auto f = [&](const auto& rhoep){
            return mp.get_cref().alphar(T, rhoep, molefrac);
        };
        return rho*rho*static_cast<double>(centered_diff<2,4>(f, static_cast<my_float_t>(rho), 1e-16*static_cast<my_float_t>(rho)));
    }
    virtual double get_Ar03ep(const double T, const double rho, const EArrayd& molefrac) const  override {
        using namespace boost::multiprecision;
        using my_float_t = number<cpp_bin_float<100U>>;
        auto f = [&](const auto& rhoep){
            return mp.get_cref().alphar(T, rhoep, molefrac);
        };
        return rho*rho*rho*static_cast<double>(centered_diff<3,4>(f, static_cast<my_float_t>(rho), 1e-16*static_cast<my_float_t>(rho)));
    }
    
    virtual double get_reducing_density(const EArrayd& molefrac) const  override {
        using Model = std::decay_t<decltype(mp.get_cref())>;
        if constexpr(CallableReducingDensity<Model, EArrayd>){
            return mp.get_cref().get_reducing_density(molefrac);
        }
        else{
            throw teqp::NotImplementedError("Cannot call get_reducing_density of a class that doesn't define it");
        }
    }
    virtual double get_reducing_temperature(const EArrayd& molefrac) const  override {
        using Model = std::decay_t<decltype(mp.get_cref())>;
        if constexpr(CallableReducingTemperature<Model, EArrayd>){
            return mp.get_cref().get_reducing_temperature(molefrac);
        }
        else{
            throw teqp::NotImplementedError("Cannot call get_reducing_temperature of a class that doesn't define it");
        }
    }
    
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
    
    // Composition derivatives with tau and delta as the working variables
    virtual double get_AtaudeltaXi(const double tau, const int NT, const double delta, const int ND, const EArrayd& molefrac, const int i, const int NXi) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_AtaudeltaXi_runtime(mp.get_cref(), tau, NT, delta, ND, molefrac, i, NXi);
    };
    virtual double get_AtaudeltaXiXj(const double tau, const int NT, const double delta, const int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_AtaudeltaXiXj_runtime(mp.get_cref(), tau, NT, delta, ND, molefrac, i, NXi, j, NXj);
    };
    virtual double get_AtaudeltaXiXjXk(const double tau, const int NT, const double delta, const int ND, const EArrayd& molefrac, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk) const override {
        return TDXDerivatives<decltype(mp.get_cref()), double, EArrayd>::get_AtaudeltaXiXjXk_runtime(mp.get_cref(), tau, NT, delta, ND, molefrac, i, NXi, j, NXj, k, NXk);
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
