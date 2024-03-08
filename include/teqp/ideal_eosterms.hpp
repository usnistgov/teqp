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
        const double a, R;
        IdealHelmholtzConstant(double a, double R) : a(a), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& /*T*/, const RhoType& /*rho*/) const {
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(a));
        }
    };

    /**
    \f$ \alpha^{\rm ig}= a\cdot \ln(T) \f$

    which should be compared with the original form in GERG (and REFPROP and CoolProp)

    \f$ \alpha^{\rm ig}= a^*\ln(\tau) \f$

    with \f$\tau=T_r/T \f$
    */
    class IdealHelmholtzLogT {
    public:
        const double a, R;
        IdealHelmholtzLogT(double a, double R) : a(a), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(a * log(T)));
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \ln(\rho) + a_1 + a_2/T \f$

    which should be compared with the original form in GERG (and REFPROP and CoolProp)

    \f$ \alpha^{\rm ig}= \ln(\delta) + a_1^* + a_2^*\tau \f$

    Note that \f$a_1\f$ contains an additive factor of \f$-ln(\rho_r)\f$ and \f$a_2\f$ contains a multiplicative factor of \f$T_r\f$
    relative to the former

    */
    class IdealHelmholtzLead {
    public:
        const double a_1, a_2, R;
        IdealHelmholtzLead(double a_1, double a_2, double R) : a_1(a_1), a_2(a_2), R(R) {};

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
        const double R;
        IdealHelmholtzPowerT(const std::valarray<double>& n, const std::valarray<double>& t, double R) : n(n), t(t), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0U; i < n.size(); ++i) {
                summer += n[i] * pow(T, t[i]);
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
        const double R;
        IdealHelmholtzPlanckEinstein(const std::valarray<double>& n, const std::valarray<double>& theta, double R) : n(n), theta(theta), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0U; i < n.size(); ++i) {
                summer += n[i] * log(1.0 - exp(-theta[i] / T));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k\ln(c_k+d_k\exp(\theta_k/T)) \f$
    */
    class IdealHelmholtzPlanckEinsteinGeneralized {
    public:
        const std::valarray<double> n, c, d, theta;
        const double R;
        IdealHelmholtzPlanckEinsteinGeneralized(
            const std::valarray<double>& n,
            const std::valarray<double>& c,
            const std::valarray<double>& d,
            const std::valarray<double>& theta,
                                                double R
        ) : n(n), c(c), d(d), theta(theta), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0U; i < n.size(); ++i) {
                summer += n[i] * log(c[i] + d[i]*exp(theta[i] / T));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k \ln(|\cosh(\theta_k/T)|) \f$

    See Table 7.6 in GERG-2004 monograph
    */
    class IdealHelmholtzGERG2004Cosh {
    public:
        const std::valarray<double> n, theta;
        const double R;
        IdealHelmholtzGERG2004Cosh(const std::valarray<double>& n, const std::valarray<double>& theta, double R) : n(n), theta(theta), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0U; i < n.size(); ++i) {
                using std::abs;
                TType cosh_theta_over_T = cosh(theta[i] / T);
                summer += n[i] * log(abs(cosh_theta_over_T));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= \sum_k n_k \ln(|\sinh(\theta_k/T)|) \f$

    See Table 7.6 in GERG-2004 monograph
    */
    class IdealHelmholtzGERG2004Sinh {
    public:
        const std::valarray<double> n, theta;
        const double R;
        IdealHelmholtzGERG2004Sinh(const std::valarray<double>& n, const std::valarray<double>& theta, double R) : n(n), theta(theta), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            std::common_type_t <TType, RhoType> summer = 0.0;
            for (auto i = 0U; i < n.size(); ++i) {
                using std::abs;
                TType sinh_theta_over_T = sinh(theta[i] / T);
                summer += n[i] * log(abs(sinh_theta_over_T));
            }
            return forceeval(summer);
        }
    };

    /**
    \f$ \alpha^{\rm ig}= c\left( \frac{T-T_0}{T}-\ln\left(\frac{T}{T_0}\right)\right) \f$
     
    from a term that is like
    
    \f$ \frac{c_{p0}}{R}= c \f$
    */
    class IdealHelmholtzCp0Constant {
    public:
        const double c, T_0;
        const double R;
        IdealHelmholtzCp0Constant(
          const double c, const double T_0, const double R
        ) : c(c), T_0(T_0), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(
                c*((T-T_0)/T-log(T/T_0))
            ));
        }
    };

    /**
    \f$ \alpha^{\rm ig}= c\left[T^{t}\left(\frac{1}{t+1}-\frac{1}{t}\right)-\frac{T_0^{t+1}}{T(t+1)}+\frac{T_0^t}{t}\right] \f$
     
    from a term that is like

    \f$ \frac{c_{p0}}{R}= cT^t, t \neq 0 \f$
    */
    class IdealHelmholtzCp0PowerT {
    public:
        const double c, t, T_0;
        const double R;
        IdealHelmholtzCp0PowerT(
            const double c, const double t, const double T_0, const double R
        ) : c(c), t(t), T_0(T_0), R(R) {};

        template<typename TType, typename RhoType>
        auto alphaig(const TType& T, const RhoType& /*rho*/) const {
            using otype = std::common_type_t <TType, RhoType>;
            return forceeval(static_cast<otype>(
                c*(pow(T,t)*(1/(t+1)-1/t) - pow(T_0,t+1)/(T*(t+1)) + pow(T_0,t)/t)
            ));
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
        IdealHelmholtzGERG2004Sinh,
        IdealHelmholtzCp0Constant,
        IdealHelmholtzCp0PowerT
    > ;

    class PureIdealHelmholtz {
    public:
        std::vector<IdealHelmholtzTerms> contributions;
        PureIdealHelmholtz(const nlohmann::json& jpure) {
            //std::string s = jpure.dump(1); 
            if (jpure.is_array()) {
                throw teqp::InvalidArgument("JSON data passed to PureIdealHelmholtz must be an object and contain the fields \"R\" and \"terms\"");
            }
            double R = jpure.at("R");
            for (auto& term : jpure.at("terms")) {
                if (!term.is_object()) {
                    throw teqp::InvalidArgument("JSON data for pure fluid must be an object");
                }
                //std::string s = term.dump(1);
                if (term.at("type") == "Constant") { // a
                    contributions.emplace_back(IdealHelmholtzConstant(term.at("a"), R));
                }
                else if (term.at("type") == "Lead") { // ln(rho) + a_1 + a_2/T
                    contributions.emplace_back(IdealHelmholtzLead(term.at("a_1"), term.at("a_2"), R));
                }
                else if (term.at("type") == "LogT") { // a*ln(T)
                    contributions.emplace_back(IdealHelmholtzLogT(term.at("a"), R));
                }
                else if (term.at("type") == "PowerT") { // sum_i n_i * T^i
                    contributions.emplace_back(IdealHelmholtzPowerT(term.at("n"), term.at("t"), R));
                }
                else if (term.at("type") == "PlanckEinstein") {
                    contributions.emplace_back(IdealHelmholtzPlanckEinstein(term.at("n"), term.at("theta"), R));
                }
                else if (term.at("type") == "PlanckEinsteinGeneralized") {
                    contributions.emplace_back(IdealHelmholtzPlanckEinsteinGeneralized(term.at("n"), term.at("c"), term.at("d"), term.at("theta"), R));
                }
                else if (term.at("type") == "GERG2004Cosh") {
                    contributions.emplace_back(IdealHelmholtzGERG2004Cosh(term.at("n"), term.at("theta"), R));
                }
                else if (term.at("type") == "GERG2004Sinh") {
                    contributions.emplace_back(IdealHelmholtzGERG2004Sinh(term.at("n"), term.at("theta"), R));
                }
                else if (term.at("type") == "Cp0Constant") {
                    contributions.emplace_back(IdealHelmholtzCp0Constant(term.at("c"), term.at("T_0"), R));
                }
                else if (term.at("type") == "Cp0PowerT") {
                    contributions.emplace_back(IdealHelmholtzCp0PowerT(term.at("c"), term.at("t"), term.at("T_0"), R));
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
            if (static_cast<std::size_t>(molefrac.size()) != pures.size()){
                throw teqp::InvalidArgument("molefrac and pures are not the same length");
            }
            std::size_t i = 0;
            for (auto &pure : pures){
                if (getbaseval(molefrac[i]) != 0){
                    ig += molefrac[i]*(pure.alphaig(T, rho) + log(molefrac[i]));
                }
                else{
                    // lim_{x\to 0} x*ln(x) => 0
                }
                i++;
            }
            return ig;
        }
        
        /// This pass-through function is required to allow this model to sit in the AllowedModels variant
        /// which allows the ideal-gas Helmholtz terms to be treated just the same as the residual terms
        template<typename TType, typename RhoType, typename MoleFrac>
        auto alphar(const TType& T, const RhoType &rho, const MoleFrac &molefrac) const {
            return alphaig(T, rho, molefrac);
        }
        
        /** For now this is a placeholder, but it should be the "correct" R, however that is ultimately decided upon */
        template<typename MoleFrac>
        auto R(const MoleFrac &/*molefrac*/) const{
            return 8.31446261815324; // J/mol/K
        }
        
    };

    inline nlohmann::json CoolProp2teqp_alphaig_term_reformatter(const nlohmann::json &term, double Tri, double rhori, double R){
        //std::string s = term.dump(1);
        
        if (term.at("type") == "IdealGasHelmholtzLead") {
            // Was ln(delta) + a_1 + a_2*tau ==> ln(rho) + a_1 + a_2/T
            // new a_1 is old a_1 - ln(rho_ri)
            // new a_2 is old a_2 * Tri
            return {{{"type", "Lead"}, {"a_1", term.at("a1").get<double>() - log(rhori)}, {"a_2", term.at("a2").get<double>() * Tri}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzEnthalpyEntropyOffset") {
            // Was a_1 + a_2*tau ==> a_1 + a_2/T
            std::valarray<double> n = {term.at("a1").get<double>(), term.at("a2").get<double>()*Tri};
            std::valarray<double> t = {0, -1};
            return {{{"type", "PowerT"}, {"n", n}, {"t", t}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzLogTau") { // a*ln(tau)
            // Was a*ln(tau) = a*ln(Tri) - a*ln(T) ==> a*ln(T)
            // Breaks into two pieces, first is a constant term a*ln(Tri), next is a*ln(T) piece
            double a = term.at("a").get<double>();
            nlohmann::json term1 = {{"type", "Constant"}, {"a", a*log(Tri)}, {"R", R}};
            nlohmann::json term2 = {{"type", "LogT"}, {"a", -a}, {"R", R}};
            return {term1, term2};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinstein") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> theta = term.at("t").get<std::valarray<double>>()*Tri;
            return {{{"type", "PlanckEinstein"}, {"n", n}, {"theta", theta}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinsteinFunctionT") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> theta = term.at("v");
            return {{{"type", "PlanckEinstein"}, {"n", n}, {"theta", theta}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPlanckEinsteinGeneralized") {
            // Was
            std::valarray<double> n = term.at("n");
            std::valarray<double> c = term.at("c");
            std::valarray<double> d = term.at("d");
            std::valarray<double> theta = term.at("t").get<std::valarray<double>>()*Tri;
//            std::cout << term.dump() << std::endl;
            return {{{"type", "PlanckEinsteinGeneralized"}, {"n", n}, {"c", c}, {"d", d}, {"theta", theta}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzPower") {
            // Was
            std::valarray<double> n = term.at("n").get<std::valarray<double>>();
            std::valarray<double> t = term.at("t").get<std::valarray<double>>();
            for (auto i = 0U; i < n.size(); ++i){
                n[i] *= pow(Tri, t[i]);
                t[i] *= -1; // T is in the denominator in CoolProp terms, in the numerator in teqp
            }
            return {{{"type", "PowerT"}, {"n", n}, {"t", t}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzCP0PolyT") {
            // Was
            nlohmann::json newterms = nlohmann::json::array();
//            std::cout << term.dump() << std::endl;
            std::valarray<double> t = term.at("t"), c = term.at("c");
            double T_0 = term.at("T0");
            for (auto i = 0U; i < t.size(); ++i){
                if (t[i] == 0){
                    newterms.push_back({{"type", "Cp0Constant"}, {"c", c[i]}, {"T_0", T_0}, {"R", R}});
                }
                else{
                    newterms.push_back({{"type", "Cp0PowerT"}, {"c", c[i]}, {"t", t[i]}, {"T_0", T_0}, {"R", R}});
                }
            }
            return newterms;
        }
        else if (term.at("type") == "IdealGasHelmholtzCP0Constant") {
//            std::cout << term.dump() << std::endl;
            double T_0 = term.at("T0");
            return {{{"type", "Cp0Constant"}, {"c", term.at("cp_over_R")}, {"T_0", T_0}, {"R", R}}};
        }
        else if (term.at("type") == "IdealGasHelmholtzCP0AlyLee") {
            // Was
            nlohmann::json newterms = nlohmann::json::array();
//            std::cout << term.dump() << std::endl;
            std::valarray<double> constants = term.at("c");
            double T_0 = term.at("T0");
            
            // Take the constant term if nonzero
            if (std::abs(constants[0]) > 1e-14) {
                newterms.push_back({{"type", "Cp0Constant"}, {"c", constants[0]}, {"T_0", T_0}, {"R", R}});
            }
            
            std::vector<double> n, c, d, t;
            if (std::abs(constants[1]) > 1e-14) {
                // sinh term can be converted by setting  a_k = C, b_k = 2*D, c_k = -1, d_k = 1
                n.push_back(constants[1]);
                t.push_back(-2 * constants[2]);
                c.push_back(1);
                d.push_back(-1);
            }
            if (std::abs(constants[3]) > 1e-14) {
                // cosh term can be converted by setting  a_k = C, b_k = 2*D, c_k = 1, d_k = 1
                n.push_back(-constants[3]);
                t.push_back(-2 * constants[4]);
                c.push_back(1);
                d.push_back(1);
            }
            newterms.push_back(
                   {{"type", "PlanckEinsteinGeneralized"}, {"n", n}, {"c", c}, {"d", d}, {"theta", t}, {"R", R}}
            );
            return newterms;
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
    \brief Convert the ideal-gas term for a term from CoolProp-formatted JSON structure
     
    \param s A string, pointing to a filesystem file, or the JSON contents to be parsed
    \param index The index of the model to load, should be zero in general
    \returns j The JSON
     
     The key difference in the approach in CoolProp and teqp is that the contributions in teqp
     are based on temperature and density as the independent variables, whereas the
     implementation in CoolProp uses the pure fluid reciprocal reduced temperature and reduced
     density as independent variables
     */
    inline nlohmann::json convert_CoolProp_idealgas(const std::string &s, int index){
        
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
        
        // Extract the things from the CoolProp-formatted data structure
        auto jEOS = j.at("EOS")[index];
        auto jCP = jEOS.at("alpha0");
        double Tri = jEOS.at("STATES").at("reducing").at("T");
        double rhori = jEOS.at("STATES").at("reducing").at("rhomolar");
        double R = jEOS.at("gas_constant");
        
        // Now we transform the inputs to teqp-formatted terms
        nlohmann::json newterms = nlohmann::json::array();
        for (const auto& term : jCP){
            // Converted can be a two-element array, so all terms are returned as array
            // and then put into newterms
            auto converted = CoolProp2teqp_alphaig_term_reformatter(term, Tri, rhori, R);
            for (auto newterm : converted){
                newterms.push_back(newterm);
                if (!newterms.back().is_object()){
                    std::cout << newterm.dump() << std::endl;
                }
            }
        }
        
        // And return our new data structure for this fluid
        return {{"terms", newterms}, {"R", R}};
    }
}
