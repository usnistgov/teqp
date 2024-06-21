#pragma once

/**
 This header contains methods that pertain to polar contributions to SAFT models
 
 Initially the contribution of Gross and Vrabec were implemented for PC-SAFT, but they can be used with other
 non-polar base models as well, so this header collects all the things in one place
 */

#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/exceptions.hpp"
#include "correlation_integrals.hpp"
#include <optional>
#include <Eigen/Dense>  
#include "teqp/math/pow_templates.hpp"
#include "teqp/models/saft/polar_terms/GrossVrabec.hpp"
#include "teqp/models/saft/polar_terms/types.hpp"
#include <variant>

namespace teqp{

namespace SAFTpolar{

using namespace teqp::saft::polar_terms;

template<typename KType, typename RhoType, typename Txy>
auto get_Kijk(const KType& Kint, const RhoType& rhostar, const Txy &Tstarij, const Txy &Tstarik, const Txy &Tstarjk){
    return forceeval(pow(forceeval(Kint.get_K(Tstarij, rhostar)*Kint.get_K(Tstarik, rhostar)*Kint.get_K(Tstarjk, rhostar)), 1.0/3.0));
};

// Special treatment needed here because the 334,445 term is negative, so negative*negative*negative is negative, and negative^{1/3} is undefined
// First flip the sign on the triple product, do the evaluation, the flip it back. Not documented in Gubbins&Twu, but this seem reasonable,
// in the spirit of the others.
template<typename KType, typename RhoType, typename Txy>
auto get_Kijk_334445(const KType& Kint, const RhoType& rhostar, const Txy &Tstarij, const Txy &Tstarik, const Txy &Tstarjk){
    return forceeval(-pow(-forceeval(Kint.get_K(Tstarij, rhostar)*Kint.get_K(Tstarik, rhostar)*Kint.get_K(Tstarjk, rhostar)), 1.0/3.0));
};

/**
 \tparam JIntegral A type that can be indexed with a single integer n to give the J^{(n)} integral
 \tparam KIntegral A type that can be indexed with a two integers a and b to give the K(a,b) integral
 
 The flexibility was added to include J and K integrals from either Luckas et al. or Gubbins and Twu (or any others following the interface)
 */
template<class JIntegral, class KIntegral>
class MultipolarContributionGubbinsTwu {
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNm3_rhostar_molefractions;
private:
    const Eigen::ArrayXd sigma_m, epsilon_over_k, mubar2, Qbar2;
    const bool has_a_polar;
    const Eigen::ArrayXd sigma_m3, sigma_m5;
    
    const JIntegral J6{6};
    const JIntegral J8{8};
    const JIntegral J10{10};
    const JIntegral J11{11};
    const JIntegral J13{13};
    const JIntegral J15{15};
    const KIntegral K222_333{222, 333};
    const KIntegral K233_344{233, 344};
    const KIntegral K334_445{334, 445};
    const KIntegral K444_555{444, 555};
    const double epsilon_0 = 8.8541878128e-12; // https://en.wikipedia.org/wiki/Vacuum_permittivity, in F/m, or C^2⋅N^−1⋅m^−2
    const double PI_ = static_cast<double>(EIGEN_PI);
    Eigen::MatrixXd SIGMAIJ, EPSKIJ;
    multipolar_rhostar_approach approach = multipolar_rhostar_approach::use_packing_fraction;
    
public:
    MultipolarContributionGubbinsTwu(const Eigen::ArrayX<double> &sigma_m, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayX<double> &mubar2, const Eigen::ArrayX<double> &Qbar2, multipolar_rhostar_approach approach) : sigma_m(sigma_m), epsilon_over_k(epsilon_over_k), mubar2(mubar2), Qbar2(Qbar2), has_a_polar(mubar2.cwiseAbs().sum() > 0 || Qbar2.cwiseAbs().sum() > 0), sigma_m3(sigma_m.pow(3)), sigma_m5(sigma_m.pow(5)), approach(approach) {
        // Check lengths match
        if (sigma_m.size() != mubar2.size()){
            throw teqp::InvalidArgument("bad size of mubar2");
        }
        if (sigma_m.size() != Qbar2.size()){
            throw teqp::InvalidArgument("bad size of Qbar2");
        }
        // Pre-calculate the mixing terms;
        const std::size_t N = sigma_m.size();
        SIGMAIJ.resize(N, N); EPSKIJ.resize(N, N);
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                // Lorentz-Berthelot mixing rules
                double epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                double sigmaij = (sigma_m[i]+sigma_m[j])/2;
                SIGMAIJ(i, j) = sigmaij;
                EPSKIJ(i, j) = epskij;
            }
        }
    }
    MultipolarContributionGubbinsTwu& operator=( const MultipolarContributionGubbinsTwu& ) = delete; // non copyable
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha2(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        
        const std::size_t N = mole_fractions.size();
        using XTtype = std::common_type_t<TTYPE, decltype(mole_fractions[0])>;
        std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])> alpha2_112 = 0.0, alpha2_123 = 0.0, alpha2_224 = 0.0;
        
        const RhoType factor_112 = -2.0*PI_*rhoN/3.0;
        const RhoType factor_123 = -PI_*rhoN;
        const RhoType factor_224 = -14.0*PI_*rhoN/5.0;
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){

                TTYPE Tstari = forceeval(T/EPSKIJ(i, i)), Tstarj = forceeval(T/EPSKIJ(j, j));
                XTtype leading = forceeval(x[i]*x[j]/(Tstari*Tstarj)); // common for all alpha_2 terms
                TTYPE Tstarij = forceeval(T/EPSKIJ(i, j));
                double sigmaij = SIGMAIJ(i,j);
                {
                    double dbl = sigma_m3[i]*sigma_m3[j]/powi(sigmaij,3)*mubar2[i]*mubar2[j];
                    alpha2_112 += leading*dbl*J6.get_J(Tstarij, rhostar);
                }
                {
                    double dbl = sigma_m3[i]*sigma_m5[j]/powi(sigmaij,5)*mubar2[i]*Qbar2[j];
                    alpha2_123 += leading*dbl*J8.get_J(Tstarij, rhostar);
                }
                {
                    double dbl = sigma_m5[i]*sigma_m5[j]/powi(sigmaij,7)*Qbar2[i]*Qbar2[j];
                    alpha2_224 += leading*dbl*J10.get_J(Tstarij, rhostar);
                }
            }
        }
        return forceeval(factor_112*alpha2_112 + 2.0*factor_123*alpha2_123 + factor_224*alpha2_224);
    }
    
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha3(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const VecType& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])>;
        using XTtype = std::common_type_t<TTYPE, decltype(mole_fractions[0])>;
        type summerA_112_112_224 = 0.0, summerA_112_123_213 = 0.0, summerA_123_123_224 = 0.0, summerA_224_224_224 = 0.0;
        type summerB_112_112_112 = 0.0, summerB_112_123_123 = 0.0, summerB_123_123_224 = 0.0, summerB_224_224_224 = 0.0;
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){

                TTYPE Tstari = forceeval(T/EPSKIJ(i,i)), Tstarj = forceeval(T/EPSKIJ(j,j));
                TTYPE Tstarij = forceeval(T/EPSKIJ(i,j));

                XTtype leading = forceeval(x[i]*x[j]/pow(forceeval(Tstari*Tstarj), 3.0/2.0)); // common for all alpha_3A terms
                double sigmaij = SIGMAIJ(i,j);
                double POW4sigmaij = powi(sigmaij, 4);
                double POW8sigmaij = POW4sigmaij*POW4sigmaij;
                double POW10sigmaij = powi(sigmaij, 10);
                double POW12sigmaij = POW4sigmaij*POW8sigmaij;
                
                {
                    double dbl = pow(sigma_m[i]*sigma_m[j], 11.0/2.0)/POW8sigmaij*mubar2[i]*mubar2[j]*sqrt(Qbar2[i]*Qbar2[j]);
                    summerA_112_112_224 += leading*dbl*J11.get_J(Tstarij, rhostar);
                }
                {
                    double dbl = pow(sigma_m[i]*sigma_m[j], 11.0/2.0)/POW8sigmaij*mubar2[i]*mubar2[j]*sqrt(Qbar2[i]*Qbar2[j]);
                    summerA_112_123_213 += leading*dbl*J11.get_J(Tstarij, rhostar);
                }
                {
                    double dbl = pow(sigma_m[i], 11.0/2.0)*pow(sigma_m[j], 15.0/2.0)/POW10sigmaij*mubar2[i]*sqrt(Qbar2[i])*pow(Qbar2[j], 3.0/2.0);
                    summerA_123_123_224 += leading*dbl*J13.get_J(Tstarij, rhostar);
                }
                {
                    double dbl = pow(sigma_m[i]*sigma_m[j], 15.0/2.0)/POW12sigmaij*pow(Qbar2[i], 3.0/2.0)*pow(Qbar2[j], 3.0/2.0);
                    summerA_224_224_224 += leading*dbl*J15.get_J(Tstarij, rhostar);
                }

                for (std::size_t k = 0; k < N; ++k){
                    TTYPE Tstark = forceeval(T/EPSKIJ(k,k));
                    TTYPE Tstarik = forceeval(T/EPSKIJ(i,k));
                    TTYPE Tstarjk = forceeval(T/EPSKIJ(j,k));
                    double sigmaik = SIGMAIJ(i,k), sigmajk = SIGMAIJ(j,k);

                    // Lorentz-Berthelot mixing rules for sigma
                    XTtype leadingijk = forceeval(x[i]*x[j]*x[k]/(Tstari*Tstarj*Tstark));

                    if (std::abs(mubar2[i]*mubar2[j]*mubar2[k]) > 0){
                        auto K222333 = get_Kijk(K222_333, rhostar, Tstarij, Tstarik, Tstarjk);
                        double dbl = sigma_m3[i]*sigma_m3[j]*sigma_m3[k]/(sigmaij*sigmaik*sigmajk)*mubar2[i]*mubar2[j]*mubar2[k];
                        summerB_112_112_112 += forceeval(leadingijk*dbl*K222333);
                    }
                    if (std::abs(mubar2[i]*mubar2[j]*Qbar2[k]) > 0){
                        auto K233344 = get_Kijk(K233_344, rhostar, Tstarij, Tstarik, Tstarjk);
                        double dbl = sigma_m3[i]*sigma_m3[j]*sigma_m5[k]/(sigmaij*POW2(sigmaik*sigmajk))*mubar2[i]*mubar2[j]*Qbar2[k];
                        summerB_112_123_123 += leadingijk*dbl*K233344;
                    }
                    if (std::abs(mubar2[i]*Qbar2[j]*Qbar2[k]) > 0){
                        auto K334445 = get_Kijk_334445(K334_445, rhostar, Tstarij, Tstarik, Tstarjk);
                        double dbl = sigma_m3[i]*sigma_m5[j]*sigma_m5[k]/(POW2(sigmaij*sigmaik)*POW3(sigmajk))*mubar2[i]*Qbar2[j]*Qbar2[k];
                        summerB_123_123_224 += leadingijk*dbl*K334445;
                    }
                    if (std::abs(Qbar2[i]*Qbar2[j]*Qbar2[k]) > 0){
                        auto K444555 = get_Kijk(K444_555, rhostar, Tstarij, Tstarik, Tstarjk);
                        double dbl = POW5(sigma_m[i]*sigma_m[j]*sigma_m[k])/(POW3(sigmaij*sigmaik*sigmajk))*Qbar2[i]*Qbar2[j]*Qbar2[k];
                        summerB_224_224_224 += leadingijk*dbl*K444555;
                    }
                }
            }
        }
        
        type alpha3A_112_112_224 = 8.0*PI_*rhoN/25.0*summerA_112_112_224;
        type alpha3A_112_123_213 = 8.0*PI_*rhoN/75.0*summerA_112_123_213;
        type alpha3A_123_123_224 = 8.0*PI_*rhoN/35.0*summerA_123_123_224;
        type alpha3A_224_224_224 = 144.0*PI_*rhoN/245.0*summerA_224_224_224;

        type alpha3A = 3.0*alpha3A_112_112_224 + 6.0*alpha3A_112_123_213 + 6.0*alpha3A_123_123_224 + alpha3A_224_224_224;

        RhoType rhoN2 = rhoN*rhoN;

        type alpha3B_112_112_112 = 32.0*POW3(PI_)*rhoN2/135.0*sqrt(14*PI_/5.0)*summerB_112_112_112;
        type alpha3B_112_123_123 = 64.0*POW3(PI_)*rhoN2/315.0*sqrt(3.0*PI_)*summerB_112_123_123;
        type alpha3B_123_123_224 = -32.0*POW3(PI_)*rhoN2/45.0*sqrt(22.0*PI_/63.0)*summerB_123_123_224;
        type alpha3B_224_224_224 = 32.0*POW3(PI_)*rhoN2/2025.0*sqrt(2002.0*PI_)*summerB_224_224_224;

        type alpha3B = alpha3B_112_112_112 + 3.0*alpha3B_112_123_123 + 3.0*alpha3B_123_123_224 + alpha3B_224_224_224;

        return forceeval(alpha3A + alpha3B);
    }
    
    template<typename RhoType, typename PFType, typename MoleFractions>
    auto get_rhostar(const RhoType rhoN, const PFType& packing_fraction, const MoleFractions& mole_fractions) const{
        using type = std::common_type_t<RhoType, PFType, decltype(mole_fractions[0])>;
        type rhostar;
        if (approach == multipolar_rhostar_approach::calculate_Gubbins_rhostar){
            // Calculate the effective reduced diameter (cubed) to be used for evaluation
            // Eq. 24 from Gubbins
            type sigma_x3 = 0.0;
            error_if_expr(sigma_m3);
            auto N = mole_fractions.size();
            for (auto i = 0; i < N; ++i){
                for (auto j = 0; j < N; ++j){
                    auto sigmaij = (sigma_m[i] + sigma_m[j])/2;
                    sigma_x3 += mole_fractions[i]*mole_fractions[j]*POW3(sigmaij);
                }
            }
            rhostar = forceeval(rhoN*sigma_x3);
        }
        else if (approach == multipolar_rhostar_approach::use_packing_fraction){
            // The packing fraction is defined by eta = pi/6*rho^*, so use the (temperature-dependent) eta to obtain rho^*
            rhostar = forceeval(packing_fraction/(static_cast<double>(EIGEN_PI)/6.0));
        }
        else{
            throw teqp::InvalidArgument("The method used to determine rho^* is invalid");
        }
        return rhostar;
    }
    
    /***
     * \brief Get the contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const {
        error_if_expr(T); error_if_expr(rhoN); error_if_expr(mole_fractions[0]);
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])>;
        
        type alpha2 = 0.0, alpha3 = 0.0, alpha = 0.0;
        if (has_a_polar){
            alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions);
            alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions);
            alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
        }
        return MultipolarContributionGubbinsTwuTermsGT<type>{alpha2, alpha3, alpha};
    }
};

struct PolarizableArrays{
    Eigen::ArrayXd alpha_symm_C2m2J, alpha_asymm_C2m2J, alpha_isotropic_C2m2J, alpha_anisotropic_C2m2J;
};

/**
 \tparam JIntegral A type that can be indexed with a single integer n to give the J^{(n)} integral
 \tparam KIntegral A type that can be indexed with a two integers a and b to give the K(a,b) integral
 
 The flexibility was added to include J and K integrals from either Luckas et al. or Gubbins and Twu (or any others following the interface)
 */
template<class JIntegral, class KIntegral>
class MultipolarContributionGrayGubbins {
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNm3_rhostar_molefractions;
private:
    const Eigen::ArrayXd sigma_m, epsilon_over_k;
    Eigen::MatrixXd SIGMAIJ, EPSKIJ;
    const Eigen::ArrayXd mu, Q, mu2, Q2, Q3;
    const bool has_a_polar;
    const Eigen::ArrayXd sigma_m3, sigma_m5;
    
    const JIntegral J6{6};
    const JIntegral J8{8};
    const JIntegral J10{10};
    const JIntegral J11{11};
    const JIntegral J13{13};
    const JIntegral J15{15};
    const KIntegral K222_333{222, 333};
    const KIntegral K233_344{233, 344};
    const KIntegral K334_445{334, 445};
    const KIntegral K444_555{444, 555};
    
    const double PI_ = static_cast<double>(EIGEN_PI);
    const double PI3 = PI_*PI_*PI_;
    const double epsilon_0 = 8.8541878128e-12; // https://en.wikipedia.org/wiki/Vacuum_permittivity, in F/m, or C^2⋅N^−1⋅m^−2
    const double k_e = teqp::constants::k_e; // coulomb constant, with units of N m^2 / C^2
    const double k_B = teqp::constants::k_B; // Boltzmann constant, with units of J/K
    
    multipolar_rhostar_approach approach = multipolar_rhostar_approach::use_packing_fraction;
    
    // These values were adjusted in the model of Paricaud, JPCB, 2023
    /// The C3b is the C of Paricaud, and C3 is the tau of Paricaud. They were renamed to be not cause confusion with the multifluid modeling approach
    const double C3, C3b;
    std::optional<PolarizableArrays> polarizable;
    
    double get_C3(const std::optional<nlohmann::json>& flags, double default_value=1.0){
        if (flags){ return flags.value().value("C3", default_value); }
        return default_value;
    }
    double get_C3b(const std::optional<nlohmann::json>& flags, double default_value=1.0){
        if (flags){ return flags.value().value("C3b", default_value); }
        return default_value;
    }
    multipolar_rhostar_approach get_approach(const std::optional<nlohmann::json>& flags){
        if (flags){ return flags.value().value("approach", multipolar_rhostar_approach::use_packing_fraction); }
        return multipolar_rhostar_approach::use_packing_fraction;
    }
    std::optional<PolarizableArrays> get_polarizable(const std::optional<nlohmann::json>& flags){
        if (flags && flags.value().contains("polarizable")){
            PolarizableArrays arrays;
            auto toeig = [](const std::valarray<double>& x) -> Eigen::ArrayXd{ return Eigen::Map<const Eigen::ArrayXd>(&(x[0]), x.size());};
            auto alpha_symm_m3 = toeig(flags.value()["polarizable"].at("alpha_symm / m^3"));
            auto alpha_asymm_m3 = toeig(flags.value()["polarizable"].at("alpha_asymm / m^3"));
            arrays.alpha_symm_C2m2J = alpha_symm_m3/k_e;
            arrays.alpha_asymm_C2m2J = alpha_asymm_m3/k_e;
            arrays.alpha_isotropic_C2m2J = 1.0/3.0*(arrays.alpha_symm_C2m2J + 2.0*arrays.alpha_asymm_C2m2J);
            arrays.alpha_anisotropic_C2m2J = arrays.alpha_symm_C2m2J - arrays.alpha_asymm_C2m2J;
            return arrays;
        }
        return std::nullopt;
    }
    
public:
    MultipolarContributionGrayGubbins(const Eigen::ArrayX<double> &sigma_m, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::MatrixXd& SIGMAIJ, const Eigen::MatrixXd& EPSKIJ, const Eigen::ArrayX<double> &mu, const Eigen::ArrayX<double> &Q, const std::optional<nlohmann::json>& flags)
    
    : sigma_m(sigma_m), epsilon_over_k(epsilon_over_k), SIGMAIJ(SIGMAIJ), EPSKIJ(EPSKIJ), mu(mu), Q(Q), mu2(mu.pow(2)), Q2(Q.pow(2)), Q3(Q.pow(3)), has_a_polar(Q.cwiseAbs().sum() > 0 || mu.cwiseAbs().sum() > 0), sigma_m3(sigma_m.pow(3)), sigma_m5(sigma_m.pow(5)), approach(get_approach(flags)), C3(get_C3(flags)), C3b(get_C3b(flags)), polarizable(get_polarizable(flags)) {
        // Check lengths match
        if (sigma_m.size() != mu.size()){
            throw teqp::InvalidArgument("bad size of mu");
        }
        if (sigma_m.size() != Q.size()){
            throw teqp::InvalidArgument("bad size of Q");
        }
        if (polarizable){
            if (polarizable.value().alpha_symm_C2m2J.size() != sigma_m.size() || polarizable.value().alpha_asymm_C2m2J.size() != sigma_m.size()){
                throw teqp::InvalidArgument("bad size of alpha arrays");
            }
        }
    }
    MultipolarContributionGrayGubbins& operator=( const MultipolarContributionGrayGubbins& ) = delete; // non copyable
    
    /// Appendix B of Gray et al.
    template<typename Jintegral, typename TTYPE, typename RhoStarType>
    auto get_In(const Jintegral& J, int n, double sigmaij, const TTYPE& Tstar, const RhoStarType& rhostar) const{
        return 4.0*PI_/pow(sigmaij, n-3)*J.get_J(Tstar, rhostar);
    }
    
    /// Appendix B of Gray et al.
    template<typename TTYPE, typename RhoStarType>
    auto Immm(std::size_t i, std::size_t j, std::size_t k, const TTYPE& T, const RhoStarType& rhostar) const {
        auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
        const double coeff = 64.0*PI3/5.0*sqrt(14*PI_/5.0)/SIGMAIJ(i,j)/SIGMAIJ(i,k)/SIGMAIJ(j,k);
        return coeff*get_Kijk(K222_333, rhostar, Tstarij, Tstarik, Tstarjk);
    }
    template<typename TTYPE, typename RhoStarType>
    auto ImmQ(std::size_t i, std::size_t j, std::size_t k, const TTYPE& T, const RhoStarType& rhostar) const {
        auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
        const double coeff = 2048.0*PI3/7.0*sqrt(3.0*PI_)/SIGMAIJ(i,j)/POW2(SIGMAIJ(i,k)*SIGMAIJ(j,k));
        return coeff*get_Kijk(K233_344, rhostar, Tstarij, Tstarik, Tstarjk);
    }
    template<typename TTYPE, typename RhoStarType>
    auto ImQQ(std::size_t i, std::size_t j, std::size_t k, const TTYPE& T, const RhoStarType& rhostar) const {
        auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
        const double coeff = -4096.0*PI3/9.0*sqrt(22.0*PI_/7.0)/POW2(SIGMAIJ(i,j)*SIGMAIJ(i,k))/POW3(SIGMAIJ(j,k));
        return coeff*get_Kijk_334445(K334_445, rhostar, Tstarij, Tstarik, Tstarjk);
    }
    template<typename TTYPE, typename RhoStarType>
    auto IQQQ(std::size_t i, std::size_t j, std::size_t k, const TTYPE& T, const RhoStarType& rhostar) const {
        auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
        const double coeff = 8192.0*PI3/81.0*sqrt(2002.0*PI_)/POW3(SIGMAIJ(i,j)*SIGMAIJ(i,k)*SIGMAIJ(j,k));
        return coeff*get_Kijk(K444_555, rhostar, Tstarij, Tstarik, Tstarjk);
    }
    
    /// Return \f$\alpha_2=A_2/(Nk_BT)\f$, thus this is a nondimensional term. This is equivalent to \f$-w_o^{(2)}/\rhoN\f$ from Gray et al.
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto get_alpha2(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& muprime) const{
        const auto& x = mole_fractions; // concision
        
        const std::size_t N = mole_fractions.size();
        
        std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])> summer = 0.0;
        
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const auto muprime2 = POW2(muprime).eval();
        
        using ztype = std::common_type_t<TTYPE, decltype(muprime[0])>;
        // We have to do this type promotion to the ztype to allow for multiplication with
        // Eigen array types, as some type promotion does not happen automatically
        //
        // alpha has units of m^3, divide by k_e (has units of J m / C^2) to get C^2m^2/J, beta has units of 1/J, muprime^2 has units of C m
        Eigen::ArrayX<ztype> z1 = 1.0/3.0*(muprime2.template cast<ztype>()*static_cast<ztype>(beta));
        Eigen::ArrayX<ztype> z2 = 0.0*z1;
        if (polarizable){
            z1 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>();
            z2 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>();
        }
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                TTYPE Tstarij = forceeval(T/EPSKIJ(i, j));
                double sigmaij = SIGMAIJ(i,j);
                summer += x[i]*x[j]*(
                     3.0/2.0*(z1[i]*z1[j] - z2[i]*z2[j])*get_In(J6, 6, sigmaij, Tstarij, rhostar)
                    + 3.0/2.0*z1[i]*beta*Q2[j]*get_In(J8, 8, sigmaij, Tstarij, rhostar)
                    +7.0/10.0*beta*beta*Q2[i]*Q2[j]*get_In(J10, 10, sigmaij, Tstarij, rhostar)
                );
            }
        }
        return forceeval(-rhoN*k_e*k_e*summer); // The factor of k_e^2 takes us from CGS to SI units
    }
    
    /// Return \f$\alpha_2=A_2/(Nk_BT)\f$, thus this is a nondimensional term. This is equivalent to \f$-w_o^{(2)}/\rhoN\f$ from Gray et al.
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto get_alpha2_muprime_gradient(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& muprime) const{
        const auto& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        
        using type_ = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])>;
        
        const TTYPE beta = 1.0/(k_B*T);
        using ztype = std::common_type_t<TTYPE, decltype(muprime[0])>;
        // We have to do this type promotion to the ztype to allow for multiplication with
        // Eigen array types, as some type promotion does not happen automatically
        Eigen::ArrayX<ztype> z1 = 1.0/3.0*(POW2(muprime).template cast<ztype>()*static_cast<ztype>(beta));
        if (polarizable){
            z1 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>();
        }
        
        // Exactly the term as defined in Gray et al., Eq. 2.11
        Eigen::ArrayX<type_> Eprime2(N);
        for (std::size_t i = 0; i < N; ++i){
            type_ summer = 0;
            for (std::size_t j = 0; j < N; ++j){
                TTYPE Tstarij = T/EPSKIJ(i, j);
                double sigmaij = SIGMAIJ(i, j);
                auto rhoj = rhoN*x[j];
                summer += rhoj*(2.0*z1[i]*get_In(J6, 6, sigmaij, Tstarij, rhostar) + beta*Q2[j]*get_In(J8, 8, sigmaij, Tstarij, rhostar) );
            }
            Eprime2[i] = muprime[i]*summer;
        }
        // And now to get dalpha2/dmu', multiply by -beta*x
        return (-k_e*k_e*Eprime2*mole_fractions.template cast<type_>()*beta).eval(); // The factor of k_e^2 takes us from CGS to SI units
    }
    
    /// Return \f$\alpha_3=A_3/(Nk_BT)\f$, thus this is a nondimensional term. This is equivalent to \f$-w_o^{(3)}/\rhoN\f$ from Gray et al.
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto get_alpha3(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& muprime) const{
        const VecType& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])>;
        type summer_a = 0.0, summer_b = 0.0;
        
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const auto muprime2 = POW2(muprime).eval();
        // We have to do this type promotion to the ztype to allow for multiplication with
        // Eigen array types, as some type promotion does not happen automatically
        using ztype = std::common_type_t<TTYPE, decltype(muprime[0])>;
        Eigen::ArrayX<ztype> z1 = 1.0/3.0*(muprime2.template cast<ztype>()*static_cast<ztype>(beta));
        Eigen::ArrayX<ztype> z2 = 0.0*z1;
        Eigen::ArrayX<ztype> gamma = 0.0*z1;
        if (polarizable){
            z1 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>(); // alpha has units of m^3, divide by k_e (has units of J m / C^2) to get C^2m^2/J, beta has units of 1/J, muprime^2 has units of C m
            z2 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>();
            gamma += polarizable.value().alpha_symm_C2m2J.template cast<ztype>() - polarizable.value().alpha_asymm_C2m2J.template cast<ztype>();
        }
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                
                TTYPE Tstarij = forceeval(T/EPSKIJ(i,j));
                double sigmaij = SIGMAIJ(i,j);
                
                auto a_ij = ((2.0/5.0*beta*beta*muprime2[i]*muprime2[j] + 4.0/5.0*gamma[i]*beta*muprime2[j] + 4.0/25.0*gamma[i]*gamma[j])*beta*Q[i]*Q[j]*get_In(J11, 11, sigmaij, Tstarij, rhostar)
                             +12.0/35.0*(beta*muprime2[i] + gamma[i])*beta*beta*Q[i]*POW3(Q[j])*get_In(J13, 13, sigmaij, Tstarij, rhostar)
                             + 36.0/245.0*POW3(beta)*Q3[i]*Q3[j]*get_In(J15, 15, sigmaij, Tstarij, rhostar)
                             );
                summer_a += x[i]*x[j]*a_ij;
                
                for (std::size_t k = 0; k < N; ++k){
                    auto b_ijk = (
                      1.0/2.0*(z1[i]*z1[j]*z1[k] - z2[i]*z2[j]*z2[k])*Immm(i, j, k, T, rhostar)
                      +C3b*(3.0/160.0*z1[i]*z1[j]*beta*Q2[k]*ImmQ(i, j, k, T, rhostar) + 3.0/640.0*z1[i]*POW2(beta)*Q2[j]*Q2[k]*ImQQ(i,j,k,T, rhostar))
                      +1.0/6400.0*POW3(beta)*Q2[i]*Q2[j]*Q2[k]*IQQQ(i, j, k, T, rhostar)
                    );
                    summer_b += x[i]*x[j]*x[k]*b_ijk;
                }
            }
        }

        return forceeval(C3*(rhoN*summer_a + rhoN*rhoN*summer_b)*k_e*k_e*k_e); // The factor of k_e^3 takes us from CGS to SI units
    }
    
    /// Return \f$\alpha_3=A_3/(Nk_BT)\f$, thus this is a nondimensional term. This is equivalent to \f$-w_o^{(3)}/\rhoN\f$ from Gray et al.
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto get_alpha3_muprime_gradient(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& muprime) const{
        const VecType& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        using type_ = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])>;
        
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const auto muprime2 = POW2(muprime).eval();
        // We have to do this type promotion to the ztype to allow for multiplication with
        // Eigen array types, as some type promotion does not happen automatically
        using ztype = std::common_type_t<TTYPE, decltype(muprime[0])>;
        Eigen::ArrayX<ztype> z1 = 1.0/3.0*(muprime2.template cast<ztype>()*static_cast<ztype>(beta));
        Eigen::ArrayX<ztype> gamma = 0.0*z1;
        if (polarizable){
            z1 += polarizable.value().alpha_isotropic_C2m2J.template cast<ztype>(); // alpha has units of m^3, divide by k_e (has units of J m / C^2) to get C^2m^2/J, beta has units of 1/J, muprime^2 has units of C m
            gamma += polarizable.value().alpha_symm_C2m2J.template cast<ztype>() - polarizable.value().alpha_asymm_C2m2J.template cast<ztype>();
        }
        
        // Exactly as in Eq. 2.12 of Gray et al. (aside from the scaling parameters of Paricaud)
        Eigen::ArrayX<type_> Eprime3(N);
        type_ summer_ij = 0, summer_ijk = 0;
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                
                TTYPE Tstarij = forceeval(T/EPSKIJ(i,j));
                double sigmaij = SIGMAIJ(i,j);
                auto p_ij = 8.0/5.0*(beta*muprime2[j] + gamma[j])*beta*Q[i]*Q[j]*get_In(J11, 11, sigmaij, Tstarij, rhostar) + 24.0/35.0*beta*beta*Q[i]*Q3[j]*get_In(J13, 13, sigmaij, Tstarij, rhostar);
                summer_ij += (rhoN*x[j]*p_ij);
                
                for (std::size_t k = 0; k < N; ++k){
                    auto q_ijk = (
                      z1[j]*z1[k]*Immm(i, j, k, T, rhostar)
                      +C3b*1.0/40.0*z1[j]*beta*Q2[k]*ImmQ(i, j, k, T, rhostar)
                      + 1.0/320.0*POW2(beta)*Q2[j]*Q2[k]*ImQQ(i,j,k,T, rhostar)
                    );
                    summer_ijk += POW2(rhoN)*x[j]*x[k]*q_ijk;
                }
            }
            Eprime3[i] = -muprime[i]*(summer_ij + summer_ijk);
        }

        return (-C3*Eprime3*k_e*k_e*k_e*mole_fractions.template cast<type_>()*beta).eval(); // The factor of k_e^3 takes us from CGS to SI units
    }
    
    template<typename RhoType, typename PFType, typename MoleFractions>
    auto get_rhostar(const RhoType rhoN, const PFType& packing_fraction, const MoleFractions& mole_fractions) const{
        using type = std::common_type_t<RhoType, PFType, decltype(mole_fractions[0])>;
        type rhostar;
        if (approach == multipolar_rhostar_approach::calculate_Gubbins_rhostar){
            // Calculate the effective reduced diameter (cubed) to be used for evaluation
            // Eq. 24 from Gubbins
            type sigma_x3 = 0.0;
            error_if_expr(sigma_m3);
            auto N = mole_fractions.size();
            for (auto i = 0; i < N; ++i){
                for (auto j = 0; j < N; ++j){
                    auto sigmaij = (sigma_m[i] + sigma_m[j])/2;
                    sigma_x3 += mole_fractions[i]*mole_fractions[j]*POW3(sigmaij);
                }
            }
            rhostar = forceeval(rhoN*sigma_x3);
        }
        else if (approach == multipolar_rhostar_approach::use_packing_fraction){
            // The packing fraction is defined by eta = pi/6*rho^*, so use the (temperature-dependent) eta to obtain rho^*
            rhostar = forceeval(packing_fraction/(static_cast<double>(EIGEN_PI)/6.0));
        }
        else{
            throw teqp::InvalidArgument("The method used to determine rho^* is invalid");
        }
        return rhostar;
    }
    
    /// Get the polarization term \f$ E' \equiv -\frac{1}{\vec{N}}\left(\frac{\partial A_{\rm perturb}}{\partial \mu'}\right)_{V,T,\vec{N}} \f$
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto get_Eprime(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& muprime) const{
        if (!polarizable){
            throw teqp::InvalidArgument("Can only use polarizable code if polarizability is enabled");
        }
//        {
//            // The lambda function to evaluate the gradient of the perturbational term
//            // w.r.t. the effective dipole moment
//            auto f = [&](const auto& muprime){
//                auto alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions, muprime);
//                auto alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions, muprime);
//                auto alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
//                return forceeval(alpha);
//            };
//            Eigen::ArrayX<autodiff::real> muprimead = muprime.template cast<autodiff::real>();
//            Eigen::ArrayXd dalphaperturb_dmuprimead = autodiff::gradient(f, wrt(muprimead), at(muprimead));  // units of 1/(C m)
//            
//            auto f2 = [&](const auto& muprime){
//                return get_alpha2(T, rhoN, rhostar, mole_fractions, muprime);
//            };
//            Eigen::ArrayXd dalphaperturb2_dmuprimead = autodiff::gradient(f2, wrt(muprimead), at(muprimead));  // units of 1/(C m)
//            Eigen::ArrayXd dalphaperturb2_dmuprime = get_alpha2_muprime_gradient(T, rhoN, rhostar, mole_fractions, muprime);
//            
//            auto f3 = [&](const auto& muprime){
//                return get_alpha3(T, rhoN, rhostar, mole_fractions, muprime);
//            };
//            Eigen::ArrayXd dalphaperturb3_dmuprimead = autodiff::gradient(f3, wrt(muprimead), at(muprimead));  // units of 1/(C m)
//            Eigen::ArrayXd dalphaperturb3_dmuprime = get_alpha3_muprime_gradient(T, rhoN, rhostar, mole_fractions, muprime);
//            
//            std::cout << dalphaperturb3_dmuprimead << " || " << dalphaperturb3_dmuprime << std::endl;
//            
//            auto dmu = 1e-6*muprime[0];
//            Eigen::ArrayXd muprimep = muprime; muprimep[0] += dmu;
//            Eigen::ArrayXd muprimem = muprime; muprimem[0] -= dmu;
//            auto dalphaperturb2_dmuprime_centered = (f2(muprimep)-f2(muprimem))/(2*dmu);
//        }
        
        auto alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions, muprime);
        auto alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions, muprime);
        auto dalphaperturb2_dmuprime = get_alpha2_muprime_gradient(T, rhoN, rhostar, mole_fractions, muprime);
        auto dalphaperturb3_dmuprime = get_alpha3_muprime_gradient(T, rhoN, rhostar, mole_fractions, muprime);
        auto dalphaperturb_dmuprime = (((1.0-2.0*alpha3/alpha2)*dalphaperturb2_dmuprime + dalphaperturb3_dmuprime)/POW2(1.0-alpha3/alpha2)).eval();
        
        return (-k_B*T*dalphaperturb_dmuprime).eval(); // Eprime has units of J /(C m) because alpha is dimensionless
    }
    
    /// Use successive substitution to obtain the effective dipole moment based solely on the perturbation term (and not the \f$U_p\f$ term)
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MuPrimeType>
    auto iterate_muprime_SS(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MuPrimeType& mu, const int max_steps) const{
        if (!polarizable){
            throw teqp::InvalidArgument("Can only use polarizable code if polarizability is enabled");
        }
        using otype = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(mu[0])>;
        Eigen::ArrayX<otype> muprime = mu.template cast<otype>();
        for (auto counter = 0; counter < max_steps; ++counter){
            auto Eprime = get_Eprime(T, rhoN, rhostar, mole_fractions, muprime); // units of J /(C m)
            // alpha*Eprime has units of J m^3/(C m), divide by k_e (has units of J m / C^2) to get C m
            muprime = mu.template cast<otype>() + polarizable.value().alpha_symm_C2m2J.template cast<otype>()*Eprime.template cast<otype>(); // Units of C m
        }
        return muprime;
    }
    
    /***
     * \brief Get the contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const {
        error_if_expr(T); error_if_expr(rhoN); error_if_expr(mole_fractions[0]);
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])>;
        
        if (!polarizable){
            type alpha2 = 0.0, alpha3 = 0.0, alpha = 0.0;
            if (has_a_polar){
                alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions, mu);
                alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions, mu);
                alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
                // Handle the case where the polar term is present but the contribution is zero
                if (getbaseval(alpha2) == 0){
                    alpha2 = 0;
                    alpha = 0;
                }
            }
            return MultipolarContributionGubbinsTwuTermsGT<type>{alpha2, alpha3, alpha};
        }
        else{
            // First solve for the effective dipole moments
            auto muprime = iterate_muprime_SS(T, rhoN, rhostar, mole_fractions, mu, 10); // C m, array
            // And the polarization energy derivative, units of J /(C m)
            auto Eprime = get_Eprime(T, rhoN, rhostar, mole_fractions, muprime); // array
            using Eprime_t = std::decay_t<decltype(Eprime[0])>;
            // And finally the polarization contribution to total polar term
            auto U_p_over_rhoN = 0.5*(mole_fractions.template cast<Eprime_t>()*polarizable.value().alpha_symm_C2m2J.template cast<Eprime_t>()*Eprime*Eprime).eval().sum(); // U_p divided by rhoN, has units of J
            auto alpha_polarization = U_p_over_rhoN/(k_B*T); // nondimensional, scalar
            
            auto alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions, muprime);
            auto alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions, muprime);
            auto alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
            return MultipolarContributionGubbinsTwuTermsGT<type>{alpha2, alpha3, alpha + alpha_polarization};
        }
    }
};

/// The variant containing the multipolar types that can be provided
using multipolar_contributions_variant = std::variant<
    teqp::saft::polar_terms::GrossVrabec::MultipolarContributionGrossVrabec,
    MultipolarContributionGrayGubbins<GubbinsTwuJIntegral, GubbinsTwuKIntegral>,
    MultipolarContributionGrayGubbins<GottschalkJIntegral, GottschalkKIntegral>,
    MultipolarContributionGrayGubbins<LuckasJIntegral, LuckasKIntegral>,
    MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>,
    MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>,
    MultipolarContributionGubbinsTwu<GottschalkJIntegral, GottschalkKIntegral>
>;

}
}
