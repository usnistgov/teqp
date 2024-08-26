#pragma once


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

using ResidualHelmholtzOverRTVariant = std::variant<NullResidualHelmholtzOverRT<double>, WilsonResidualHelmholtzOverRT<double>>;

inline ResidualHelmholtzOverRTVariant ares_model_factory(const nlohmann::json& armodel) {
    
    std::string type = armodel.at("type");
    if (type == "Wilson"){
        std::vector<double> b = armodel.at("b");
        auto mWilson = build_square_matrix(armodel.at("m"));
        auto nWilson = build_square_matrix(armodel.at("n"));
        return WilsonResidualHelmholtzOverRT<double>(b, mWilson, nWilson);
    }
    else{
        throw teqp::InvalidArgument("bad type of ares model: " + type);
    }
};

}