#pragma once
#include <variant>

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

namespace teqp {

    /**
    \f$ \alpha^{\rm ig}= a \f$
    */
    class IdealHelmholtzConstant {
    public:
        const double a;
        IdealHelmholtzConstant(double a) : a(a) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            using otype = std::common_type_t <TType, RhoType>;
            return static_cast<otype>(a);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= a\times ln(T) \f$

    which should be compared with the original form in GERG (and REFPROP and CoolProp)

    \f$ \alpha^{\rm ig}= a^*\ln(\tau) \f$

    with \f$\tau=T_r/T \f$
    */
    class IdealHelmholtzLogT {
    public:
        const double a;
        IdealHelmholtzLogT(double a) : a(a) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(a * log(T)));
        }
    };

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
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(log(rho) + a_1 + a_2 / T));
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k\ln(1-\exp(-\theta_k/T)) \f$
    */
    class IdealHelmholtzPlanckEinstein {
    public:
        const std::valarray<double> n, theta;
        IdealHelmholtzPlanckEinstein(const std::valarray<double>& n, const std::valarray<double>& theta) : n(n), theta(theta) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0; i < n.size(); ++i) {
                summer = summer + n[i] * log(1 - exp(-theta[i] / T));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k ln(|cosh(theta_k/T)|) \f$

    See Table 7.6 in GERG-2004 monograph
    */
    class IdealHelmholtzGERG2004Cosh {
    public:
        const std::valarray<double> n, theta;
        IdealHelmholtzGERG2004Cosh(const std::valarray<double>& n, const std::valarray<double>& theta) : n(n), theta(theta) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0; i < n.size(); ++i) {
                summer = summer + n[i] * log(abs(cosh(theta[i] / T)));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k ln(|sinh(theta_k/T)|) \f$

    See Table 7.6 in GERG-2004 monograph
    */
    class IdealHelmholtzGERG2004Sinh {
    public:
        const std::valarray<double> n, theta;
        IdealHelmholtzGERG2004Sinh(const std::valarray<double>& n, const std::valarray<double>& theta) : n(n), theta(theta) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0; i < n.size(); ++i) {
                summer = summer + n[i] * log(abs(sinh(theta[i] / T)));
            }
            return forceeval(summer);
        }
    };

    // The collection of possible terms that could be part of the summation
    using IdealHelmholtzTerms = std::variant <
        IdealHelmholtzConstant, 
        IdealHelmholtzLead, 
        IdealHelmholtzLogT, 
        IdealHelmholtzPlanckEinstein,
        IdealHelmholtzGERG2004Cosh, 
        IdealHelmholtzGERG2004Sinh
    > ;

    class PureIdealHelmholtz {
    public:
        std::vector<IdealHelmholtzTerms> contributions;
        PureIdealHelmholtz(const nlohmann::json& jpure) {
            //std::string s = jpure.dump(1); 
            if (!jpure.is_array()) {
                throw teqp::InvalidArgument("JSON data passed to PureIdealHelmholtz must be an array");
            }
            for (auto& term : jpure) {
                if (!term.is_object()) {
                    throw teqp::InvalidArgument("JSON data for pure fluid must be an array");
                }
                //std::string s = term.dump(1);
                if (term.at("type") == "Constant") { // a
                    contributions.emplace_back(IdealHelmholtzConstant(term.at("a")));
                }
                else if (term.at("type") == "Lead") { // ln(rho) + a_1 + a_2/T
                    contributions.emplace_back(IdealHelmholtzLead(term.at("a_1"), term.at("a_2")));
                }
                else if (term.at("type") == "LogT") { // a*ln(T)
                    contributions.emplace_back(IdealHelmholtzLogT(term.at("a")));
                }
                else if (term.at("type") == "PlanckEinstein") {
                    contributions.emplace_back(IdealHelmholtzPlanckEinstein(term.at("n"), term.at("theta")));
                }
                else if (term.at("type") == "GERG2004Cosh") {
                    contributions.emplace_back(IdealHelmholtzGERG2004Cosh(term.at("n"), term.at("theta")));
                }
                else if (term.at("type") == "GERG2004Sinh") {
                    contributions.emplace_back(IdealHelmholtzGERG2004Sinh(term.at("n"), term.at("theta")));
                }
                else {
                    throw InvalidArgument("Don't understand this type: " + term.at("type").get<std::string>());
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
            if (!jpures.is_array()) {
                throw teqp::InvalidArgument("JSON data passed to IdealHelmholtz must be an array");
            }
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