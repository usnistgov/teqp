#pragma once // Only include header once in a compilation unit if it is included multiple times

#include <set>

#include "teqp/constants.hpp" // used for R
#include "teqp/types.hpp" // needed for forceeval
#include "teqp/exceptions.hpp" // to return teqp error messages

#include "nlohmann/json.hpp" 

namespace teqp{
namespace LKP{

struct LKPFluidParameters {
    std::vector<double> b, c, d;
    double beta, gamma_, omega;
};

class LKPMix{
public:
    const LKPFluidParameters simple{{0.0, 0.1181193, 0.265728, 0.154790, 0.303230e-1}, // b, with pad to match indices
        {0.0, 0.236744e-1, 0.186984e-1, 0, 0.427240e-1},  // c, with pad to match indices
        {0, 0.155428e-4, 0.623689e-4}, // d, with pad to match indices
        0.653920, 0.601670e-1, 0.0}, // beta, gamma, omega
    ref{{0, 0.2026579, 0.331511, 0.276550e-1, 0.203488},  // b, with pad to match indices
        {0, 0.313385e-1, 0.503618e-1, 0.169010e-1,0.41577e-1}, // c, with pad to match indices
        {0, 0.487360e-4, 0.740336e-5},  // d, with pad to match indices
        1.226, 0.03754, 0.3978}; // beta, gamma, omega
    const std::vector<double> Tcrit, pcrit, acentric;
    const double m_R; ///< molar gas constant to be used in this model, in J/mol/K
    const std::vector<std::vector<double>> kmat;
    
    LKPMix(const std::vector<double>& Tcrit, const std::vector<double>& pcrit, const std::vector<double>& acentric, double R, const std::vector<std::vector<double>>& kmat) : Tcrit(Tcrit), pcrit(pcrit), acentric(acentric), m_R(R), kmat(kmat){
        std::size_t N = Tcrit.size();
        if (std::set<std::size_t>{Tcrit.size(), pcrit.size(), acentric.size()}.size() > 1){
            throw teqp::InvalidArgument("The arrays should all be the same size.");
        }
        std::string kmaterr = "The kmat is the wrong size. It should be square with dimension " + std::to_string(N);
        if (kmat.size() != N){
            throw teqp::InvalidArgument(kmaterr);
        }
        else{
            for (auto& krow: kmat){
                if(krow.size() != N){
                    throw teqp::InvalidArgument(kmaterr);
                }
            }
        }
    }
        
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const { return m_R; }
    
    /// Calculate the contribution for one of the fluids, depending on the parameter set passed in
    template<typename TTYPE, typename RhoType, typename ZcType>
    auto alphar_func(const TTYPE& tau, const RhoType& delta, const ZcType& Zc, const LKPFluidParameters& params) const {
        auto B = params.b[1] - params.b[2]*tau - params.b[3]*powi(tau, 2) - params.b[4]*powi(tau, 3);
        auto C = params.c[1] - params.c[2]*tau + params.c[3]*powi(tau, 3);
        auto D = params.d[1] + params.d[2]*tau;
        auto deltaZc = forceeval(delta/Zc);
        return forceeval(B/Zc*delta + 1.0/2.0*C*powi(deltaZc, 2) + 1.0/5.0*D*powi(deltaZc, 5) - params.c[4]*powi(tau, 3)/(2*params.gamma_)*(params.gamma_*powi(deltaZc, 2)+params.beta+1.0)*exp(-params.gamma_*powi(deltaZc, 2)) + params.c[4]*powi(tau,3)/(2*params.gamma_)*(params.beta+1.0));
    }
    
    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        
        if (static_cast<std::size_t>(mole_fractions.size()) != acentric.size()){
            throw teqp::InvalidArgument("The mole fractions should be of of size "+ std::to_string(acentric.size()));
        }
        
        const VecType& x = mole_fractions; // just an alias to save typing, no copy is invoked
        std::decay_t<decltype(mole_fractions[0])> summer_omega = 0.0, summer_vcmix = 0.0, summer_Tcmix = 0.0;
        double Ru = m_R;
        
        for (auto i = 0; i < mole_fractions.size(); ++i){
            summer_omega += mole_fractions[i]*acentric[i];
            auto v_ci = (0.2905-0.085*acentric[i])*Ru*Tcrit[i]/pcrit[i];
            for (auto j = 0; j < mole_fractions.size(); ++j){
                auto v_cj = (0.2905-0.085*acentric[j])*Ru*Tcrit[j]/pcrit[j];
                auto v_c_ij = 1.0/8.0*powi(cbrt(v_ci) + cbrt(v_cj), 3);
                auto T_c_ij = kmat[i][j]*sqrt(Tcrit[i]*Tcrit[j]);
                summer_vcmix += x[i]*x[j]*v_c_ij;
                summer_Tcmix += x[i]*x[j]*pow(v_c_ij, 0.25)*T_c_ij;
            }
        }
        auto omega_mix = summer_omega;
        auto vc_mix = summer_vcmix;
        auto Tc_mix = 1.0/pow(summer_vcmix, 0.25)*summer_Tcmix;
//        auto pc_mix = (0.2905-0.085*omega_mix)*Ru*Tc_mix/vc_mix;
        auto Zc = forceeval(0.2905-0.085*omega_mix);
        auto tau = forceeval(Tc_mix/T);
        auto delta = forceeval(vc_mix*rhomolar);
        
        auto retval = (1.0-omega_mix/ref.omega)*alphar_func(tau, delta, Zc, simple) + (omega_mix/ref.omega)*alphar_func(tau, delta, Zc, ref);
        return forceeval(retval);
    }
};

// Factory function takes in JSON data and returns an instance of the LKPMix class
inline auto make_LKPMix(const nlohmann::json& j){
    return LKPMix(j.at("Tcrit / K"), j.at("pcrit / Pa"), j.at("acentric"), j.at("R / J/mol/K"), j.at("kmat"));
}

}
}
