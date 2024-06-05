#pragma once

#include "nlohmann/json.hpp"

#include "teqp/models/multifluid.hpp"

namespace teqp{
namespace ECSHuberEly{

/**
 Implements the ECS model of Huber and Ely from 1994
 */
class ECSHuberEly1994{
private:
    using multifluid_t = decltype(multifluidfactory(nlohmann::json{}));
    
    double acentric_reference, Z_crit_reference, T_crit_reference, rhomolar_crit_reference;
    multifluid_t reference_model;
    double acentric_fluid, Z_crit_fluid, T_crit_fluid, rhomolar_crit_fluid;
    std::vector<double> f_T_coeffs, h_T_coeffs;
    
public:
    ECSHuberEly1994(const nlohmann::json& j): reference_model(build_multifluid_model({j.at("reference_fluid").at("name")}, "")) {
        const auto& ref = j.at("reference_fluid");
        acentric_reference = ref.at("acentric");
        Z_crit_reference = ref.at("Z_crit");
        T_crit_reference = ref.at("T_crit / K");
        rhomolar_crit_reference = ref.at("rhomolar_crit / mol/m^3");
        
        const auto& fl = j.at("fluid");
        acentric_fluid = fl.at("acentric");
        Z_crit_fluid = fl.at("Z_crit");
        T_crit_fluid = fl.at("T_crit / K");
        rhomolar_crit_fluid = fl.at("rhomolar_crit / mol/m^3");
        f_T_coeffs = fl.at("f_T_coeffs").get<std::vector<double>>();
        h_T_coeffs = fl.at("h_T_coeffs").get<std::vector<double>>();
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }
    
    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        
        auto Tri = T/T_crit_fluid;
        
        // These are the basic definitions from Huber and Ely
        auto theta = 1.0 + (acentric_fluid-acentric_reference)*(f_T_coeffs[0] + f_T_coeffs[1]*log(Tri)); // Eq. 30
        auto phi = Z_crit_reference/Z_crit_fluid*(1.0 + (acentric_fluid - acentric_reference)*(h_T_coeffs[0] + h_T_coeffs[1]*log(Tri))); // Eq. 31
        
        auto f = T_crit_fluid/T_crit_reference*theta;
        auto h = rhomolar_crit_reference/rhomolar_crit_fluid*phi;
        
        // Calculate the effective temperature and density (sometimes called conformal temperature and conformal density)
        auto T_effective = forceeval(T/f);
        auto rho_effective = forceeval(rhomolar*h);
        
        return reference_model.alphar(T_effective, rho_effective, mole_fractions);
    }
};


}
}
