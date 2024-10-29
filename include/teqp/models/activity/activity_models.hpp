#pragma once

#include "teqp/models/activity/COSMOSAC.hpp"

namespace teqp::activity::activity_models{

/**
 A residual Helmholtz term that returns nothing (the empty term)
 */
template<typename NumType>
class NullResidualHelmholtzOverRT {
public:
    template<typename TType, typename MoleFractions>
    auto operator () (const TType& /*T*/, const MoleFractions& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> val = 0.0;
        return val;
    }
};

/**
 
 \f[
 \frac{a^{E,\gamma}_{total}}{RT} = -sum_iz_i\ln\left(\sum_jz_jOmega_{ij}(T)\right)
 \f]
 
 \f[
 \frac{a^{E,\gamma}_{comb}}{RT} = -sum_iz_i\ln\left(\frac{\Omega_i}{z_i}\right)
  \f]
 
 \f[
 \frac{a^{E,\gamma}_{res}}{RT} = \frac{a^{E,\gamma}_{total}}{RT} - \frac{a^{E,\gamma}_{comb}}{RT}
 \f]
 
 Volume fraction of component \f$i\f$
\f[
 \phi_i = \frac{z_iv_i}{\sum_j z_j v_j}
 \f]
 with \f$v_i = b_i\f$
 */
template<typename NumType>
class WilsonResidualHelmholtzOverRT {
    
public:
    const std::vector<double> b;
    const Eigen::ArrayXXd m, n;
    WilsonResidualHelmholtzOverRT(const std::vector<double>& b, const Eigen::ArrayXXd& m, const Eigen::ArrayXXd& n) : b(b), m(m), n(n) {};
    
    template<typename TType, typename MoleFractions>
    auto combinatorial(const TType& /*T*/, const MoleFractions& molefracs) const {
        if (b.size() != static_cast<std::size_t>(molefracs.size())){
            throw teqp::InvalidArgument("Bad size of molefracs");
        }
        
        using TYPE = std::common_type_t<TType, decltype(molefracs[0])>;
        // The denominator in Phi
        TYPE Vtot = 0.0;
        for (auto i = 0U; i < molefracs.size(); ++i){
            auto v_i = b[i];
            Vtot += molefracs[i]*v_i;
        }
        
        TYPE summer = 0.0;
        for (auto i = 0U; i < molefracs.size(); ++i){
            auto v_i = b[i];
            // The ratio phi_i/z_i is expressed like this to better handle
            // the case of z_i = 0, which would otherwise be a divide by zero
            // in the case that the composition of one component is zero
            auto phi_i_over_z_i = v_i/Vtot;
            summer += molefracs[i]*log(phi_i_over_z_i);
        }
        return summer;
    }
    
    template<typename TType>
    auto get_Aij(std::size_t i, std::size_t j, const TType& T) const{
        return forceeval(m(i,j)*T + n(i,j));
    }
    
    template<typename TType, typename MoleFractions>
    auto total(const TType& T, const MoleFractions& molefracs) const {
        
        using TYPE = std::common_type_t<TType, decltype(molefracs[0])>;
        TYPE summer = 0.0;
        for (auto i = 0U; i < molefracs.size(); ++i){
            auto v_i = b[i];
            TYPE summerj = 0.0;
            for (auto j = 0U; j < molefracs.size(); ++j){
                auto v_j = b[j];
                auto Aij = get_Aij(i,j,T);
                auto Omega_ji = v_j/v_i*exp(-Aij/T);
                summerj += molefracs[j]*Omega_ji;
            }
            summer += molefracs[i]*log(summerj);
        }
        return forceeval(-summer);
    }
    
    // Returns ares/RT
    template<typename TType, typename MoleFractions>
    auto operator () (const TType& T, const MoleFractions& molefracs) const {
        return forceeval(total(T, molefracs) - combinatorial(T, molefracs));
    }
};

///**
// \note This approach works well except that... the derivatives at the pure fluid endpoints don't. So this is more a record as a failed idea
// */
//template<typename NumType>
//class BinaryBetaResidualHelmholtzOverRT {
//    
//public:
//    const std::vector<double> c;
//    BinaryBetaResidualHelmholtzOverRT(const std::vector<double>& c) : c(c) {};
//    
//    // Returns ares/RT
//    template<typename TType, typename MoleFractions>
//    auto operator () (const TType& /*T*/, const MoleFractions& molefracs) const {
//        if (molefracs.size() != 2){
//            throw teqp::InvalidArgument("Must be size of 2");
//        }
//        std::decay_t<std::common_type_t<TType, decltype(molefracs[0])>> out = c[0]*pow(molefracs[0], c[1])*pow(molefracs[1], c[2]);
//        return out;
//    }
//};

template<typename NumType>
class BinaryInvariantResidualHelmholtzOverRT {
    
public:
    const std::vector<double> c;
    BinaryInvariantResidualHelmholtzOverRT(const std::vector<double>& c) : c(c) {};
    
    // Returns ares/RT
    template<typename TType, typename MoleFractions>
    auto operator () (const TType& /*T*/, const MoleFractions& molefracs) const {
        if (molefracs.size() != 2){
            throw teqp::InvalidArgument("Must be size of 2");
        }
        std::decay_t<std::common_type_t<TType, decltype(molefracs[0])>> out = c[0]*molefracs[0]*molefracs[1]*(c[1] + c[2]*molefracs[1]);
        return out;
    }
};

using ResidualHelmholtzOverRTVariant = std::variant<NullResidualHelmholtzOverRT<double>, WilsonResidualHelmholtzOverRT<double>, BinaryInvariantResidualHelmholtzOverRT<double>, COSMOSAC::COSMO3>;

inline ResidualHelmholtzOverRTVariant ares_model_factory(const nlohmann::json& armodel) {
    
    std::string type = armodel.at("type");
    if (type == "Wilson"){
        std::vector<double> b = armodel.at("b");
        auto mWilson = build_square_matrix(armodel.at("m"));
        auto nWilson = build_square_matrix(armodel.at("n"));
        return WilsonResidualHelmholtzOverRT<double>(b, mWilson, nWilson);
    }
//    else if (type == "binaryBeta"){
//        std::vector<double> c = armodel.at("c");
//        return BinaryBetaResidualHelmholtzOverRT<double>(c);
//    }
    else if (type == "binaryInvariant"){
        std::vector<double> c = armodel.at("c");
        return BinaryInvariantResidualHelmholtzOverRT<double>(c);
    }
    else if (type == "COSMO-SAC-2010"){
        std::vector<double> A_COSMOSAC_A2 = armodel.at("A_COSMOSAC / A^2");
        std::vector<double> V_COSMOSAC_A3 = armodel.at("V_COSMOSAC / A^3");
        std::vector<COSMOSAC::FluidSigmaProfiles> profiles;
        for (auto& el : armodel.at("profiles")){
            COSMOSAC::FluidSigmaProfiles prof;
            auto get_ = [](const auto& j){
                return COSMOSAC::SigmaProfile{
                    j.at("sigma / e/A^2").template get<std::vector<double>>(),
                    j.at("p(sigma)*A / A^2").template get<std::vector<double>>()
                };
            };
            prof.nhb = get_(el.at("nhb"));
            prof.oh = get_(el.at("oh"));
            prof.ot = get_(el.at("ot"));
            profiles.push_back(prof);
        }
        COSMOSAC::COSMO3Constants constants;
        if (armodel.contains("constants")){
            const auto &jconstants = armodel.at("constants");
            constants.A_ES = jconstants.value("A_ES / kcal A^4 /(mol e^2)", constants.A_ES);
            constants.B_ES = jconstants.value("B_ES / kcal A^4 K^2/(mol e^2)", constants.B_ES);
            constants.fast_Gamma = jconstants.value("fast_Gamma", constants.fast_Gamma);
        }
        std::cout << constants.A_ES << std::endl;
        std::cout << constants.B_ES << std::endl;
        return COSMOSAC::COSMO3(A_COSMOSAC_A2, V_COSMOSAC_A3, profiles, constants);
    }
    else{
        throw teqp::InvalidArgument("bad type of ares model: " + type);
    }
};

}
