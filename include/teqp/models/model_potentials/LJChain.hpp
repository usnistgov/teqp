#pragma once

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include <map>

namespace teqp {
    namespace LJChain {
    
    /**
     Chain contribution from Johnson, Muller and Gubbins, JPC 1994
     */
    template<typename Monomer>
    class LJChain {
    private:
        const Monomer monomer;
        double m;
        const std::valarray<std::valarray<double>> aij_Johnson = {
            {},
            {0, 0.49304346593882,  2.1528349894745 ,-15.955682329017,  24.035999666294 , -8.6437958513990},
            {0,-0.47031983115362,  1.1471647487376 , 37.889828024211, -84.667121491179 , 39.643914108411},
            {0, 5.0325486243620 ,-25.915399226419  ,-18.862251310090, 107.63707381726  ,-66.602649735720},
            {0,-7.3633150434385 , 51.553565337453  ,-40.519369256098, -38.796692647218 , 44.605139198378},
            {0, 2.9043607296043 ,-24.478812869291  , 31.500186765040,  -5.3368920371407, -9.5183440180133}
        };
        
    public:
        LJChain(Monomer &&monomer, double m) : monomer(monomer), m(m){};

        template<typename TType, typename RhoType>
        auto g_LJ(const TType& Tstar, const RhoType& rhostar_monomer) const{
            std::common_type_t<TType, RhoType> summer = 1.0;
            for (auto i = 1; i < 6; ++i){
                for (auto j = 1; j < 6; ++j){
                    summer += aij_Johnson[i][j]*powi(rhostar_monomer,i)*powi(Tstar, 1-j);
                }
            }
            return summer;
        }

        template<typename TType, typename RhoType>
        auto get_lnyR(const TType& Tstar, const RhoType& rhostar_monomer) const{
            return forceeval(log(g_LJ(Tstar, rhostar_monomer)));
        }

        template<typename TType, typename RhoType, typename MoleFracType>
        auto alphar(const TType& Tstar,
            const RhoType& rho_chain_star,
            const MoleFracType& molefrac) const
        {
            auto rhostar_monomer = forceeval(rho_chain_star*m);
            auto alphar_monomer = m*monomer.alphar(Tstar, rhostar_monomer, molefrac);
            auto alphar_chain = (1-m)*get_lnyR(Tstar, rhostar_monomer);
            return forceeval(alphar_chain + alphar_monomer);
        }

        template<class VecType>
        auto R(const VecType& /*molefrac*/) const {
            return 1.0;
        }
    };
    
    } // namespace LJChain
}; // namespace teqp
