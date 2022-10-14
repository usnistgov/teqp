#pragma once
#include <variant>
#include <filesystem>

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/json_tools.hpp"

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
    \f$ \alpha^{\rm ig}= \sum_k n_kT^{t_k} \f$
    */
    class IdealHelmholtzPowerT {
    public:
        const std::valarray<double> n, t;
        IdealHelmholtzPowerT(const std::valarray<double>& n, const std::valarray<double>& t) : n(n), t(t) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0; i < n.size(); ++i) {
                summer = summer + n[i] * pow(T, t[i]);
            }
            return forceeval(summer);
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
    \f$ \alpha^{\rm ig}= \sum_k n_k\ln(c_k+d_k\exp(-\theta_k/T)) \f$
    */
    class IdealHelmholtzPlanckEinsteinGeneralized {
    public:
        const std::valarray<double> n, c, d, theta;
        IdealHelmholtzPlanckEinsteinGeneralized(
            const std::valarray<double>& n,
            const std::valarray<double>& c,
            const std::valarray<double>& d,
            const std::valarray<double>& theta
        ) : n(n), c(c), d(d), theta(theta) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& rho) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0; i < n.size(); ++i) {
                summer = summer + n[i] * log(c[i] + d[i]*exp(-theta[i] / T));
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
        IdealHelmholtzPowerT,
        IdealHelmholtzPlanckEinstein,
        IdealHelmholtzPlanckEinsteinGeneralized,
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
                    throw teqp::InvalidArgument("JSON data for pure fluid must be an object");
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
                else if (term.at("type") == "PowerT") { // sum_i n_i * T^i
                    contributions.emplace_back(IdealHelmholtzPowerT(term.at("n"), term.at("t")));
                }
                else if (term.at("type") == "PlanckEinstein") {
                    contributions.emplace_back(IdealHelmholtzPlanckEinstein(term.at("n"), term.at("theta")));
                }
                else if (term.at("type") == "PlanckEinsteinGeneralized") {
                    contributions.emplace_back(IdealHelmholtzPlanckEinsteinGeneralized(term.at("n"), term.at("c"), term.at("d"), term.at("theta")));
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
     * where \f$x_i\f$ are mole fractions
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

    inline nlohmann::json CoolProp2teqp_alphaig_term_reformatter(const nlohmann::json &term, double Tri, double rhori){
        //std::string s = term.dump(1);
        
        if (term.at("type") == "IdealGasHelmholtzLead") {
            // Was ln(delta) + a_1 + a_2*tau ==> ln(rho) + a_1 + a_2/T
            // new a_1 is old a_1 - ln(rho_ri)
            // new a_2 is old a_2 * Tri
            return {{{"type", "Lead"}, {"a_1", term.at("a1").get<double>() - log(rhori)}, {"a_2", term.at("a2").get<double>() * Tri}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzEnthalpyEntropyOffset") {
            // Was a_1 + a_2*tau ==> a_1 + a_2/T
            std::valarray<double> n = {term.at("a1").get<double>(), term.at("a2").get<double>()*Tri};
            std::valarray<double> t = {0, -1};
            return {{{"type", "PowerT"}, {"n", n}, {"t", t}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzLogTau") { // a*ln(tau)
            // Was a*ln(tau) = a*ln(Tri) - a*ln(T) ==> a*ln(T)
            // Breaks into two pieces, first is a constant term a*ln(Tri), next is a*ln(T) piece
            double a = term.at("a").get<double>();
            nlohmann::json term1 = {{"type", "Constant"}, {"a", a*log(Tri)}};
            nlohmann::json term2 = {{"type", "LogT"}, {"a", -a}};
            return {term1, term2};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinstein") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> theta = term.at("t").get<std::valarray<double>>()*Tri;
            return {{{"type", "PlanckEinstein"}, {"n", n}, {"theta", theta}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinsteinFunctionT") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> theta = term.at("v");
            return {{{"type", "PlanckEinstein"}, {"n", n}, {"theta", theta}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinsteinGeneralized") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> c = term.at("c");
            std::valarray<double> d = term.at("d");
            std::valarray<double> theta = term.at("t").get<std::valarray<double>>();
            return {{{"type", "PlanckEinsteinGeneralized"}, {"n", n}, {"c", c}, {"d", d}, {"theta", theta}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPower") {
            // Was
            std::valarray<double> n = term.at("n").get<std::valarray<double>>();
            std::valarray<double> t = term.at("t").get<std::valarray<double>>();
            for (auto i = 0; i < n.size(); ++i){
                n[i] *= pow(Tri, t[i]);
                t[i] *= -1; // T is in the denominator in CoolProp terms, in the numerator in teqp
            }
            return {{{"type", "PowerT"}, {"n", n}, {"t", t}}};
        }
//        else if (term.at("type") == "GERG2004Cosh") {
//            //contributions.emplace_back(IdealHelmholtzGERG2004Cosh(term.at("n"), term.at("theta")));
//        }
//        else if (term.at("type") == "GERG2004Sinh") {
//            //contributions.emplace_back(IdealHelmholtzGERG2004Sinh(term.at("n"), term.at("theta")));
//        }
        else {
            throw InvalidArgument("Don't understand this type of CoolProp ideal-gas Helmholtz energy term: " + term.at("type").get<std::string>());
        }
    }

    /**
    \brief Load the ideal-gas term for a term from CoolProp-formatted JSON structure
     
    \param path A string, pointing to a filesystem file, or the JSON contents to be parsed
     
     The key difference in the approach in CoolProp and teqp is that the contributions in teqp
     are based on temperature and density as the independent variables, whereas the
     implementation in CoolProp uses the pure fluid reciprocal reduced temperature and reduced
     density as independent variables
     */
    inline auto convert_CoolProp_idealgas(const std::string &s, int index){
        
        nlohmann::json j;
        
        // Get the JSON structure to be parsed
        try{
            // First assume that the input argument is a path
            std::filesystem::path p = s;
            j = load_a_JSON_file(s);
        }
        catch(std::exception &){
            j = nlohmann::json::parse(s);
        }
        
        // The CoolProp-formatted data structure
        auto jEOS = j.at("EOS")[index];
        auto jCP = jEOS.at("alpha0");
        double Tri = jEOS.at("STATES").at("reducing").at("T");
        double rhori = jEOS.at("STATES").at("reducing").at("rhomolar");
        
        // Now we transform the inputs to teqp-formatted terms
        nlohmann::json newterms = nlohmann::json::array();
        for (const auto& term : jCP){
            // Converted can be a two-element array, so all terms are returned as array
            // and then put into newterms
            auto converted = CoolProp2teqp_alphaig_term_reformatter(term, Tri, rhori);
            for (auto newterm : converted){
                newterms.push_back(newterm);
            }
        }
        
        // And return our new data structure for this fluid
        return newterms;
    }
}
