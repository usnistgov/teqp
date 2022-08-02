#pragma once
#include <variant>

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

namespace teqp {

    /**
    \f$ \alpha^{\rm ig}= ln(\rho) + a_1 + a_2/T \f$

    which should be compared with the original form in GERG (and REFPROP and CoolProp)

    \f$ \alpha^{\rm ig}= ln(\delta) + a_1^* + a_2^*\tau \f$

    Note that a_1 contains an additive factor of -ln(rho_r) and a_2 contains a multiplicative factor of Tc
    relative to the former

    */
    class IdealHelmholtzLead {
    public:
        const double a_1, a_2;
        IdealHelmholtzLead(double a_1, double a_2) : a_1(a_1), a_2(a_2) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            return forceeval(log(rho) + a_1 + a_2/T);
        }
    };

    using IdealHelmholtzTerms = std::variant<IdealHelmholtzLead>;

    class PureIdealHelmholtz {
    public:
        std::vector<IdealHelmholtzTerms> contributions;
        PureIdealHelmholtz(const nlohmann::json& jpure) {
            for (auto& term : jpure) {
                if (term.at("type") == "Lead") {
                    contributions.emplace_back(IdealHelmholtzLead(term.at("a_1"), term.at("a_2")));
                }
                else {
                    throw InvalidArgument("Don't understand this type: " + term.at("type"));
                }
            }
        }
        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType &rho) const{
            std::common_type_t <TType, RhoType> ig = 0.0;
            for (const auto& term : contributions) {
                auto contrib = std::visit([&](auto& t) { return t.alphaig(T, rho); }, term);
                ig = ig + contrib;
            }
            return ig;
        }
    };

    /**
     * @brief Ideal-gas Helmholtz energy container
     * 
     * \f[ \alpha^{\rm ig} = \sum_i x_i[\alpha^{\rm ig}_{oi}(T,\rho) + x_i] \f]
     *
     * where x_i are mole fractions
     * 
     */
    class IdealHelmholtz {
        
        public:
        
        std::vector<PureIdealHelmholtz> pures;
        
        IdealHelmholtz(const nlohmann::json &jpures){
            for (auto &jpure : jpures){
                pures.emplace_back(jpure);
            }
        }

        template<typename TType, typename RhoType, typename MoleFrac>
        auto alphaig(const TType& T, const RhoType &rho, const MoleFrac &molefrac) const {
            std::common_type_t <TType, RhoType, decltype(molefrac[0])> ig = 0.0;
            if (molefrac.size() != pures.size()){
                throw teqp::InvalidArgument("molefrac and pures are not the same length");
            }
            std::size_t i = 0;
            for (auto &pure : pures){
                if (molefrac[i] != 0){
                    ig += molefrac[i]*(pure.alphaig(T, rho) + log(molefrac[i]));
                }
                else{
                    // lim_{x\to 0} x*ln(x) => 0
                }
                i++;
            }
            return ig;
        }
    };
}