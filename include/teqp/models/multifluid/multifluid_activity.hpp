#pragma once

#include "teqp/types.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/models/activity/activity_models.hpp"

namespace teqp::multifluid::multifluid_activity {

using namespace teqp::activity::activity_models;

template<typename M, typename T, typename U>
concept CallableLnGammaResid = requires(M m, T t, U u) {
    { m.calc_lngamma_resid(t,u) } -> std::same_as<Eigen::ArrayXd>;
};
template<typename M, typename T, typename U>
concept CallableLnGammaComb = requires(M m, T t, U u) {
    { m.calc_lngamma_comb(t,u) }  -> std::same_as<Eigen::ArrayXd>;
};

static_assert(!CallableLnGammaResid<NullResidualHelmholtzOverRT<double>, double, Eigen::ArrayXd>);
static_assert(!CallableLnGammaComb<NullResidualHelmholtzOverRT<double>, double, Eigen::ArrayXd>);

static_assert(CallableLnGammaComb<COSMOSAC::COSMO3, double, Eigen::ArrayXd>);
static_assert(CallableLnGammaResid<COSMOSAC::COSMO3, double, Eigen::ArrayXd>);

/**
 Implementing the general approach of:
 Jaeger et al.,
 A theoretically based departure function for multi-fluid mixture models,
 https://doi.org/10.1016/j.fluid.2018.04.015
*/
class MultifluidPlusActivity{
private:
    using multifluid_t = decltype(multifluidfactory(nlohmann::json{}));
    const multifluid_t m_multifluid;
    const ResidualHelmholtzOverRTVariant m_activity;
    const std::vector<double> b;
    const double u;
public:
    MultifluidPlusActivity(const nlohmann::json &spec) :
        m_multifluid(multifluidfactory(spec.at("multifluid"))),
        m_activity(ares_model_factory(spec.at("activity").at("aresmodel"))),
        b(spec.at("activity").at("options").at("b").get<std::vector<double>>()),
        u(spec.at("activity").at("options").at("u")){}
    
    /// Calculate the dimensionless value of \f$g_{\rm GE}^{\rm E,R}/RT\f$ from the AC model
    auto calc_gER_over_RT(double T, const Eigen::ArrayXd& molefrac) const {
        return std::visit([T, &molefrac](const auto& mod){return mod(T, molefrac); }, m_activity);
    }
    /// Calculate the value of array \f$\ln\gamma^{\rm R}\f$ from the AC model without any of the AD types
    auto calc_lngamma_resid(const double T, const Eigen::ArrayXd& molefrac) const {
        return std::visit([T, &molefrac](const auto& mod) -> Eigen::ArrayXd {
            if constexpr (CallableLnGammaResid<decltype(mod), decltype(T), decltype(molefrac)>){
                return mod.calc_lngamma_resid(T, molefrac);
            }
            else{
                throw teqp::NotImplementedError("this method is not implemented");
            }
        }, m_activity);
    }
    /// Calculate the value of array \f$\ln\gamma^{\rm comb}\f$ from the AC model without any of the AD types
    Eigen::ArrayXd calc_lngamma_comb(const double T, const Eigen::ArrayXd& molefrac) const {
        return std::visit([T, &molefrac](const auto& mod) -> Eigen::ArrayXd{
            if constexpr (CallableLnGammaComb<decltype(mod), decltype(T), decltype(molefrac)>){
                return mod.calc_lngamma_comb(T, molefrac);
            }
            else{
                throw teqp::NotImplementedError("this method is not implemented");
            }
        }, m_activity);
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }
    
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar_activity(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        auto gER_over_RT = std::visit([T, &molefrac](const auto& mod){return mod(T, molefrac); }, m_activity); // dimensionless
        if (static_cast<long>(b.size()) != molefrac.size()){
            throw teqp::InvalidArgument("Size of mole fractions is incorrect");
        }
        
        auto bm = contiguous_dotproduct(b, molefrac);
        
        const auto& Tcvec = m_multifluid.redfunc.Tc;
        const auto& vcvec = m_multifluid.redfunc.vc;
        
        auto rhor = m_multifluid.redfunc.get_rhor(molefrac);
        auto Tr = m_multifluid.redfunc.get_Tr(molefrac);
        auto tau = forceeval(Tr/T);
        auto delta_ref = forceeval(1.0/(u*bm*rhor));
        
        std::decay_t<std::common_type_t<TType, decltype(molefrac[0])>> summer = 0.0;
        for (auto i = 0; i < molefrac.size(); ++i){
             auto delta_i_ref = forceeval(1.0/(u*b[i]/vcvec(i)));
             auto tau_i = forceeval(Tcvec(i)/T);
             summer += molefrac(i)*(m_multifluid.alphar_taudeltai(tau, delta_ref, i) - m_multifluid.alphar_taudeltai(tau_i, delta_i_ref, i));
        }
        return forceeval(log(1.0+rho*bm)/log(1.0+1.0/u)*(gER_over_RT - summer));
    }
    
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        return forceeval(
            m_multifluid.alphar(T, rho, molefrac)
            + alphar_activity(T, rho, molefrac)
        );
    }
};

}; // namespace teqp
