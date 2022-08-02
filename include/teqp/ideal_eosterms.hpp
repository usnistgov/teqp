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

    class IdealHelmholtz {
    public:
        std::vector<IdealHelmholtzTerms> contributions;
        IdealHelmholtz(const nlohmann::json& j) {
            for (auto& term : j) {
                if (term.at("type") == "Lead") {
                    contributions.emplace_back(IdealHelmholtzLead(term.at("a_1"), term.at("a_2")));
                }
                else {
                    throw InvalidArgument("Don't understand this type: " + term.at("type"));
                }
            }
        }
        template<typename TType, typename RhoType, typename MoleFrac>
        auto alphaig(const TType& T, const RhoType &rho, const MoleFrac& /**/) const{
            std::common_type_t <TType, RhoType> ig = 0.0;
            for (const auto& term : contributions) {
                auto contrib = std::visit([&](auto& t) { return t.alphaig(T, rho); }, term);
                ig = ig + contrib;
            }
            return ig;
        }
    };

}