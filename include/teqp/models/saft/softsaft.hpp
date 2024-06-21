#pragma once

#include "teqp/models/mie/lennardjones/johnson.hpp"
#include "teqp/types.hpp"
#include "teqp/constants.hpp"

namespace teqp::saft::softsaft{

namespace detail{

    const std::valarray<std::valarray<double>> aij_Johnson = {
        {},
        {0, 0.49304346593882,  2.1528349894745 ,-15.955682329017,  24.035999666294 , -8.6437958513990},
        {0,-0.47031983115362,  1.1471647487376 , 37.889828024211, -84.667121491179 , 39.643914108411},
        {0, 5.0325486243620 ,-25.915399226419  ,-18.862251310090, 107.63707381726  ,-66.602649735720},
        {0,-7.3633150434385 , 51.553565337453  ,-40.519369256098, -38.796692647218 , 44.605139198378},
        {0, 2.9043607296043 ,-24.478812869291  , 31.500186765040,  -5.3368920371407, -9.5183440180133}
    };

    template<typename TType, typename RhoType>
    auto g_LJ(const TType& Tstar, const RhoType& rhostar_monomer){
        std::common_type_t<TType, RhoType> summer = 1.0;
        for (auto i = 1; i < 6; ++i){
            for (auto j = 1; j < 6; ++j){
                summer += aij_Johnson[i][j]*powi(rhostar_monomer,i)*powi(Tstar, 1-j);
            }
        }
        return summer;
    }
}

class SoftSAFT{
public:
    Eigen::ArrayXd m, epsilon_over_k, sigma_m, sigma_m3;
    mie::lennardjones::Johnson::LJ126Johnson1993 Johnson;
    Eigen::ArrayXd toEig(const std::vector<double>&x){ return Eigen::Map<const Eigen::ArrayXd>(&x[0], x.size()); }
    
    SoftSAFT(const Eigen::ArrayXd&m, const Eigen::ArrayXd&epsilon_over_k, const Eigen::ArrayXd&sigma_m) : m(m), epsilon_over_k(epsilon_over_k), sigma_m(sigma_m), sigma_m3(sigma_m.pow(3)) {}
    
    SoftSAFT(const nlohmann::json& j) : SoftSAFT(toEig(j.at("m")), toEig(j.at("epsilon/kB / K")), toEig(j.at("sigma / m"))) {}
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }
 
    // The mixture is assumed to be formed of homosegmented constituents
    template <typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T, const RhoType& rhomolar, const MoleFracType& molefracs) const{
        using resulttype = std::decay_t<std::common_type_t<TType,RhoType,decltype(molefracs[0])>>;
        
        std::size_t N = molefracs.size();
        auto get_sigma3 = [this](std::size_t i, std::size_t j){
            double x = (sigma_m[i] + sigma_m[j])/2;
            return x*x*x;
        };
        auto get_epsilon_over_k = [this](std::size_t i, std::size_t j){
            return sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
        };
        
        // Use of vdW-1f rule to get the effective sigma and epsilon/k of the mixture
        std::decay_t<decltype(molefracs[0])> num1 = 0.0, num2 = 0.0, den = 0.0, m_mix = 0.0;
        for (auto i = 0U; i < N; ++i){
            m_mix += m[i]*molefracs[i];
            for (auto j = 0U; j < N; ++j){
                auto denentry = m[i]*m[j]*molefracs[i]*molefracs[j];
                auto num1entry = denentry*get_sigma3(i, j);
                auto num2entry = num1entry*get_epsilon_over_k(i, j);
                num1 += num1entry;
                num2 += num2entry;
                den += denentry;
            }
        }
        auto sigmamix3 = num1/den;
        auto epskmix = num2/den/sigmamix3;
        
        auto rhoN_monomer = rhomolar*m_mix*N_A; // Effective number density of segments (monomers per volume)
        
        auto Tstar_eff = forceeval(T/epskmix);
        auto rhostar_monomer_eff = forceeval(rhoN_monomer*sigmamix3);
        
        // Evaluate the contribution for the monomer based on
        // reduced temperature & density from vdW-1f mixing rules
        resulttype alphar_ref = m_mix*Johnson.alphar(Tstar_eff, rhostar_monomer_eff, Eigen::ArrayXd{0});
        
        // Evaluate the contribution for the chain
        resulttype alphar_chain = 0.0;
        
        // This is as in Blas (I think, some subtleties are maybe a bit different)
//        for (auto i = 0; i < N; ++i){
//            auto rhostar_monomer_i = rhomolar_monomer*sigma_m3[i];
//            alphar_chain += molefracs[i]*(1-m[i])*log(detail::g_LJ(Tstar_eff, rhostar_monomer_i));
//        }
        
        // This is as in Ghonasgi and Chapman, use the bulk effective monomer density
        alphar_chain = (1.0-m_mix)*log(detail::g_LJ(Tstar_eff, rhostar_monomer_eff));
        
        return forceeval(alphar_ref + alphar_chain);
    }
};

}
