#pragma once

/**
 This header contains methods that pertain to polar contributions to SAFT models
 
 Initially the contribution of Gross and Vrabec were implemented for PC-SAFT, but they can be used with other
 non-polar base models as well, so this header collects all the things in one place
 */

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "correlation_integrals.hpp"
#include <optional>
#include <Eigen/Dense>

namespace teqp{

namespace SAFTpolar{

template<typename A> auto POW2(const A& x) { return forceeval(x*x); }
template<typename A> auto POW3(const A& x) { return forceeval(POW2(x)*x); }
template<typename A> auto POW4(const A& x) { return forceeval(POW2(x)*POW2(x)); }
template<typename A> auto POW5(const A& x) { return forceeval(POW2(x)*POW3(x)); }
template<typename A> auto POW7(const A& x) { return forceeval(POW2(x)*POW5(x)); }
template<typename A> auto POW8(const A& x) { return forceeval(POW4(x)*POW4(x)); }
template<typename A> auto POW10(const A& x) { return forceeval(POW2(x)*POW8(x)); }
template<typename A> auto POW12(const A& x) { return forceeval(POW4(x)*POW8(x)); }

/// Eq. 10 from Gross and Vrabec
template <typename Eta, typename MType, typename TType>
auto get_JDD_2ij(const Eta& eta, const MType& mij, const TType& Tstarij) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(5) << 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(5) << 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(5) << -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575).finished();

    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(5) << 0.2187939, -1.1896431, 1.1626889, 0, 0).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(5) << -0.5873164, 1.2489132, -0.5085280, 0, 0).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(5) << 3.4869576, -14.915974, 15.372022, 0, 0).finished();
    
    std::common_type_t<Eta, MType> summer = 0.0;
    for (auto n = 0; n < 5; ++n){
        auto anij = a_0[n] + (mij-1)/mij*a_1[n] + (mij-1)/mij*(mij-2)/mij*a_2[n]; // Eq. 12
        auto bnij = b_0[n] + (mij-1)/mij*b_1[n] + (mij-1)/mij*(mij-2)/mij*b_2[n]; // Eq. 13
        summer += (anij + bnij/Tstarij)*pow(eta, n);
    }
    return forceeval(summer);
}

/// Eq. 11 from Gross and Vrabec
template <typename Eta, typename MType>
auto get_JDD_3ijk(const Eta& eta, const MType& mijk) {
    static Eigen::ArrayXd c_0 = (Eigen::ArrayXd(5) << -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0.0).finished();
    static Eigen::ArrayXd c_1 = (Eigen::ArrayXd(5) << -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0.0).finished();
    static Eigen::ArrayXd c_2 = (Eigen::ArrayXd(5) << -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0.0).finished();
    std::common_type_t<Eta, MType> summer = 0.0;
    for (auto n = 0; n < 5; ++n){
        auto cnijk = c_0[n] + (mijk-1)/mijk*c_1[n] + (mijk-1)/mijk*(mijk-2)/mijk*c_2[n]; // Eq. 14
        summer += cnijk*pow(eta, n);
    }
    return forceeval(summer);
}

/// Eq. 12 from Gross and Vrabec, AICHEJ
template <typename Eta, typename MType, typename TType>
auto get_JQQ_2ij(const Eta& eta, const MType& mij, const TType& Tstarij) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(5) << 1.2378308, 2.4355031, 1.6330905, -1.6118152, 6.9771185).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(5) << 1.2854109, -11.465615, 22.086893, 7.4691383, -17.197772).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(5) << 1.7942954, 0.7695103, 7.2647923, 94.486699, -77.148458).finished();

    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(5) << 0.4542718, -4.5016264, 3.5858868, 0.0, 0.0).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(5) << -0.8137340, 10.064030, -10.876631, 0.0, 0.0).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(5) << 6.8682675, -5.1732238, -17.240207, 0.0, 0.0).finished();
    
    std::common_type_t<Eta, MType> summer = 0.0;
    for (auto n = 0; n < 5; ++n){
        auto anij = a_0[n] + (mij-1)/mij*a_1[n] + (mij-1)/mij*(mij-2)/mij*a_2[n]; // Eq. 12
        auto bnij = b_0[n] + (mij-1)/mij*b_1[n] + (mij-1)/mij*(mij-2)/mij*b_2[n]; // Eq. 13
        summer += (anij + bnij/Tstarij)*pow(eta, n);
    }
    return forceeval(summer);
}

/// Eq. 13 from Gross and Vrabec, AICHEJ
template <typename Eta, typename MType>
auto get_JQQ_3ijk(const Eta& eta, const MType& mijk) {
    static Eigen::ArrayXd c_0 = (Eigen::ArrayXd(5) << 0.5000437, 6.5318692, -16.014780, 14.425970, 0.0).finished();
    static Eigen::ArrayXd c_1 = (Eigen::ArrayXd(5) << 2.0002094, -6.7838658, 20.383246, -10.895984, 0.0).finished();
    static Eigen::ArrayXd c_2 = (Eigen::ArrayXd(5) << 3.1358271, 7.2475888, 3.0759478, 0.0, 0.0).finished();
    std::common_type_t<Eta, MType> summer = 0.0;
    for (auto n = 0; n < 5; ++n){
        auto cnijk = c_0[n] + (mijk-1)/mijk*c_1[n] + (mijk-1)/mijk*(mijk-2)/mijk*c_2[n]; // Eq. 14
        summer += cnijk*pow(eta, n);
    }
    return forceeval(summer);
}

/***
 * \brief The dipolar contribution given in Gross and Vrabec
 */
class DipolarContributionGrossVrabec {
private:
    const Eigen::ArrayXd m, sigma_Angstrom, epsilon_over_k, mustar2, nmu;
public:
    const bool has_a_polar;
    DipolarContributionGrossVrabec(const Eigen::ArrayX<double> &m, const Eigen::ArrayX<double> &sigma_Angstrom, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayX<double> &mustar2, const Eigen::ArrayX<double> &nmu) : m(m), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), mustar2(mustar2), nmu(nmu), has_a_polar(mustar2.cwiseAbs().sum() > 0) {
        // Check lengths match
        if (m.size() != mustar2.size()){
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (m.size() != nmu.size()){
            throw teqp::InvalidArgument("bad size of n");
        }
    }
    
    /// Eq. 8 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha2DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                auto ninj = nmu[i]*nmu[j];
                if (ninj > 0){
                    // Lorentz-Berthelot mixing rules
                    auto epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                    auto sigmaij = (sigma[i]+sigma[j])/2;
                    
                    auto Tstarij = T/epskij;
                    auto mij = std::min(sqrt(m[i]*m[j]), 2.0);
                    summer += x[i]*x[j]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*POW3(sigma[i]*sigma[j]/sigmaij)*ninj*mustar2[i]*mustar2[j]*get_JDD_2ij(eta, mij, Tstarij);
                }
            }
        }
        return forceeval(-static_cast<double>(EIGEN_PI)*rhoN_A3*summer);
    }
    
    /// Eq. 9 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha3DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                for (auto k = 0; k < N; ++k){
                    auto ninjnk = nmu[i]*nmu[j]*nmu[k];
                    if (ninjnk > 0){
                        // Lorentz-Berthelot mixing rules for sigma
                        auto sigmaij = (sigma[i]+sigma[j])/2;
                        auto sigmaik = (sigma[i]+sigma[k])/2;
                        auto sigmajk = (sigma[j]+sigma[k])/2;
                        
                        auto mijk = std::min(pow(m[i]*m[j]*m[k], 1.0/3.0), 2.0);
                        summer += x[i]*x[j]*x[k]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*epsilon_over_k[k]/T*POW3(sigma[i]*sigma[j]*sigma[k])/(sigmaij*sigmaik*sigmajk)*ninjnk*mustar2[i]*mustar2[j]*mustar2[k]*get_JDD_3ijk(eta, mijk);
                    }
                }
            }
        }
        return forceeval(-4.0*POW2(static_cast<double>(EIGEN_PI))/3.0*POW2(rhoN_A3)*summer);
    }
    
    /***
     * \brief Get the dipolar contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        auto alpha2 = get_alpha2DD(T, rho_A3, eta, mole_fractions);
        auto alpha3 = get_alpha3DD(T, rho_A3, eta, mole_fractions);
        auto alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
        
        using alpha2_t = decltype(alpha2);
        using alpha3_t = decltype(alpha3);
        using alpha_t = decltype(alpha);
        struct DipolarContributionTerms{
            alpha2_t alpha2;
            alpha3_t alpha3;
            alpha_t alpha;
        };
        return DipolarContributionTerms{alpha2, alpha3, alpha};
    }
};

/***
 * \brief The quadrupolar contribution from Gross and Vrabec
 *
 */
class QuadrupolarContributionGrossVrabec {
private:
    const Eigen::ArrayXd m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ;
    
public:
    const bool has_a_polar;
    QuadrupolarContributionGrossVrabec(const Eigen::ArrayX<double> &m, const Eigen::ArrayX<double> &sigma_Angstrom, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayX<double> &Qstar2, const Eigen::ArrayX<double> &nQ) : m(m), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), Qstar2(Qstar2), nQ(nQ), has_a_polar(Qstar2.cwiseAbs().sum() > 0) {
        // Check lengths match
        if (m.size() != Qstar2.size()){
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (m.size() != nQ.size()){
            throw teqp::InvalidArgument("bad size of n");
        }
    }
    QuadrupolarContributionGrossVrabec& operator=( const QuadrupolarContributionGrossVrabec& ) = delete; // non copyable
    
    /// Eq. 9 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha2QQ(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                auto ninj = nQ[i]*nQ[j];
                if (ninj > 0){
                    // Lorentz-Berthelot mixing rules
                    auto epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                    auto sigmaij = (sigma[i]+sigma[j])/2;
                    
                    auto Tstarij = T/epskij;
                    auto mij = std::min(sqrt(m[i]*m[j]), 2.0);
                    summer += x[i]*x[j]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*POW5(sigma[i]*sigma[j])/POW7(sigmaij)*ninj*Qstar2[i]*Qstar2[j]*get_JQQ_2ij(eta, mij, Tstarij);
                }
            }
        }
        return forceeval(-static_cast<double>(EIGEN_PI)*POW2(3.0/4.0)*rhoN_A3*summer);
    }
    
    /// Eq. 1 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha3QQ(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                for (auto k = 0; k < N; ++k){
                    auto ninjnk = nQ[i]*nQ[j]*nQ[k];
                    if (ninjnk > 0){
                        // Lorentz-Berthelot mixing rules for sigma
                        auto sigmaij = (sigma[i]+sigma[j])/2;
                        auto sigmaik = (sigma[i]+sigma[k])/2;
                        auto sigmajk = (sigma[j]+sigma[k])/2;
                        
                        auto mijk = std::min(pow(m[i]*m[j]*m[k], 1.0/3.0), 2.0);
                        summer += x[i]*x[j]*x[k]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*epsilon_over_k[k]/T*POW5(sigma[i]*sigma[j]*sigma[k])/POW3(sigmaij*sigmaik*sigmajk)*ninjnk*Qstar2[i]*Qstar2[j]*Qstar2[k]*get_JDD_3ijk(eta, mijk);
                    }
                }
            }
        }
        return forceeval(-4.0*POW2(static_cast<double>(EIGEN_PI))/3.0*POW3(3.0/4.0)*POW2(rhoN_A3)*summer);
    }
    
    /***
     * \brief Get the quadrupolar contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        auto alpha2 = get_alpha2QQ(T, rho_A3, eta, mole_fractions);
        auto alpha3 = get_alpha3QQ(T, rho_A3, eta, mole_fractions);
        auto alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
        
        using alpha2_t = decltype(alpha2);
        using alpha3_t = decltype(alpha3);
        using alpha_t = decltype(alpha);
        struct QuadrupolarContributionTerms{
            alpha2_t alpha2;
            alpha3_t alpha3;
            alpha_t alpha;
        };
        return QuadrupolarContributionTerms{alpha2, alpha3, alpha};
    }
};

enum class multipolar_argument_spec {
    TK_rhoNA3_packingfraction_molefractions,
    TK_rhoNm3_molefractions
};

class MultipolarContributionGrossVrabec{
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNA3_packingfraction_molefractions;
    
    const std::optional<DipolarContributionGrossVrabec> di;
    const std::optional<QuadrupolarContributionGrossVrabec> quad;
    // TODO: add cross term
    
    MultipolarContributionGrossVrabec(
      const Eigen::ArrayX<double> &m,
      const Eigen::ArrayX<double> &sigma_Angstrom,
      const Eigen::ArrayX<double> &epsilon_over_k,
      const Eigen::ArrayX<double> &mustar2,
      const Eigen::ArrayX<double> &nmu,
      const Eigen::ArrayX<double> &Qstar2,
      const Eigen::ArrayX<double> &nQ)
    : di(((nmu.sum() > 0) ? decltype(di)(DipolarContributionGrossVrabec(m, sigma_Angstrom, epsilon_over_k, mustar2, nmu)) : std::nullopt)),
      quad(((nQ.sum() > 0) ? decltype(quad)(QuadrupolarContributionGrossVrabec(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ)) : std::nullopt)) {};
    
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        
        using type = std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])>;
        type alpha2DD = 0.0, alpha3DD = 0.0, alphaDD = 0.0;
        if (di && di.value().has_a_polar){
            alpha2DD = di.value().get_alpha2DD(T, rho_A3, eta, mole_fractions);
            alpha3DD = di.value().get_alpha3DD(T, rho_A3, eta, mole_fractions);
            alphaDD = forceeval(alpha2DD/(1.0-alpha3DD/alpha2DD));
        }
        
        type alpha2QQ = 0.0, alpha3QQ = 0.0, alphaQQ = 0.0;
        if (quad && quad.value().has_a_polar){
            alpha2QQ = quad.value().get_alpha2QQ(T, rho_A3, eta, mole_fractions);
            alpha3QQ = quad.value().get_alpha3QQ(T, rho_A3, eta, mole_fractions);
            alphaQQ = forceeval(alpha2QQ/(1.0-alpha3QQ/alpha2QQ));
        }
        
        auto alpha = forceeval(alphaDD + alphaQQ);
        
        struct Terms{
            type alpha2DD;
            type alpha3DD;
            type alphaDD;
            type alpha2QQ;
            type alpha3QQ;
            type alphaQQ;
            type alpha;
        };
        return Terms{alpha2DD, alpha3DD, alphaDD, alpha2QQ, alpha3QQ, alphaQQ, alpha};
    }
    
};

/**
 \tparam JIntegral A type that can be indexed with a single integer n to give the J^{(n)} integral
 \tparam KIntegral A type that can be indexed with a two integers a and b to give the K(a,b) integral
 
 The flexibility was added to include J and K integrals from either Luckas et al. or Gubbins and Twu (or any others following the interface)
 */
template<class JIntegral, class KIntegral>
class MultipolarContributionGubbinsTwu {
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNm3_molefractions;
private:
    const Eigen::ArrayXd sigma_m, epsilon_over_k, mubar2, Qbar2;
    const bool has_a_polar;
    
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
    
public:
    MultipolarContributionGubbinsTwu(const Eigen::ArrayX<double> &sigma_m, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayX<double> &mubar2, const Eigen::ArrayX<double> &Qbar2) : sigma_m(sigma_m), epsilon_over_k(epsilon_over_k), mubar2(mubar2), Qbar2(Qbar2), has_a_polar(mubar2.cwiseAbs().sum() > 0 || Qbar2.cwiseAbs().sum() > 0) {
        // Check lengths match
        if (sigma_m.size() != mubar2.size()){
            throw teqp::InvalidArgument("bad size of mubar2");
        }
        if (sigma_m.size() != Qbar2.size()){
            throw teqp::InvalidArgument("bad size of Qbar2");
        }
    }
    MultipolarContributionGubbinsTwu& operator=( const MultipolarContributionGubbinsTwu& ) = delete; // non copyable
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha2(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_m; // concision
        
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> alpha2_112 = 0.0, alpha2_123 = 0.0, alpha2_224 = 0.0;
        
        const auto factor_112 = forceeval(-2.0*PI_*rhoN/3.0); //*POW2(4*PI_*epsilon_0)
        const auto factor_123 = forceeval(-PI_*rhoN/3.0);
        const auto factor_224 = forceeval(-14.0*PI_*rhoN/5.0);
                
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                // Lorentz-Berthelot mixing rules
                auto epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                auto sigmaij = (sigma[i]+sigma[j])/2;
                
                auto Tstari = T/epsilon_over_k[i], Tstarj = T/epsilon_over_k[j];
                auto leading = x[i]*x[j]/(Tstari*Tstarj); // common for all alpha_2 terms
                auto Tstarij = forceeval(T/epskij);
                
                alpha2_112 += factor_112*leading*POW3(sigma[i]*sigma[j])/POW3(sigmaij)*mubar2[i]*mubar2[j]*J6.get_J(Tstarij, rhostar);
                alpha2_123 += factor_123*leading*POW3(sigma[i])*POW5(sigma[j])/POW5(sigmaij)*mubar2[i]*Qbar2[j]*J8.get_J(Tstarij, rhostar);
                alpha2_224 += factor_224*leading*POW5(sigma[i]*sigma[j])/POW7(sigmaij)*Qbar2[i]*Qbar2[j]*J10.get_J(Tstarij, rhostar);
            }
        }
        
        return forceeval(alpha2_112 + 2.0*alpha2_123 + alpha2_224);
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha3(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_m; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summerA_112_112_224 = 0.0, summerA_112_123_213 = 0.0, summerA_123_123_224 = 0.0, summerA_224_224_224 = 0.0;
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summerB_112_112_112 = 0.0, summerB_112_123_123 = 0.0, summerB_123_123_224 = 0.0, summerB_224_224_224 = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                
                // Lorentz-Berthelot mixing rules
                auto epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                auto sigmaij = (sigma[i]+sigma[j])/2;
                
                auto Tstari = T/epsilon_over_k[i], Tstarj = T/epsilon_over_k[j];
                auto Tstarij = forceeval(T/epskij);
                
                auto leading = x[i]*x[j]/pow(Tstari*Tstarj, 3.0/2.0); // common for all alpha_3A terms
                
                summerA_112_112_224 += leading*pow(sigma[i]*sigma[j], 11.0/2.0)/POW8(sigmaij)*mubar2[i]*mubar2[j]*sqrt(Qbar2[i]*Qbar2[j])*J11.get_J(Tstarij, rhostar);
                summerA_112_123_213 += leading*pow(sigma[i]*sigma[j], 11.0/2.0)/POW8(sigmaij)*mubar2[i]*mubar2[j]*sqrt(Qbar2[i]*Qbar2[j])*J11.get_J(Tstarij, rhostar);
                summerA_123_123_224 += leading*pow(sigma[i], 11.0/2.0)*pow(sigma[j], 15.0/2.0)/POW10(sigmaij)*mubar2[i]*sqrt(Qbar2[i])*pow(Qbar2[j], 3.0/2.0)*J13.get_J(Tstarij, rhostar);
                summerA_224_224_224 += leading*pow(sigma[i]*sigma[j], 15.0/2.0)/POW12(sigmaij)*Qbar2[i]*Qbar2[j]*J15.get_J(Tstarij, rhostar);
                
                for (auto k = 0; k < N; ++k){
                    auto Tstark = T/epsilon_over_k[k];
                    auto epskik = sqrt(epsilon_over_k[i]*epsilon_over_k[k]);
                    auto epskjk = sqrt(epsilon_over_k[j]*epsilon_over_k[k]);
                    auto Tstarik = T/epskik;
                    auto Tstarjk = T/epskjk;
                    
                    // Lorentz-Berthelot mixing rules for sigma
                    auto sigmaij = (sigma[i]+sigma[j])/2;
                    auto sigmaik = (sigma[i]+sigma[k])/2;
                    auto sigmajk = (sigma[j]+sigma[k])/2;
                    auto leadingijk = x[i]*x[j]*x[k]/(Tstari*Tstarj*Tstark);
                    
                    auto get_Kijk = [&](const auto& Kint){
                        return forceeval(pow(Kint.get_K(Tstarij, rhostar)*Kint.get_K(Tstarik, rhostar)*Kint.get_K(Tstarjk, rhostar), 1.0/3.0));
                    };
                    
                    // Special treatment needed here because the 334,445 term is negative, so negative*negative*negative is negative, and negative^{1/3} is undefined
                    // First flip the sign on the triple product, do the evaluation, the flip it back. Not documented in Gubbins&Twu, but this seem reasonable,
                    // in the spirit of the others.
                    auto get_Kijk_334445 = [&](const auto& Kint){
                        return forceeval(-pow(-Kint.get_K(Tstarij, rhostar)*Kint.get_K(Tstarik, rhostar)*Kint.get_K(Tstarjk, rhostar), 1.0/3.0));
                    };
                        
                    if (std::abs(mubar2[i]*mubar2[j]*mubar2[k]) > 0){
                        auto K222333 = get_Kijk(K222_333);
                        summerB_112_112_112 += leadingijk*POW3(sigma[i]*sigma[j]*sigma[k])/(sigmaij*sigmaik*sigmajk)*mubar2[i]*mubar2[j]*mubar2[k]*K222333;
                    }
                    if (std::abs(mubar2[i]*mubar2[j]*Qbar2[k]) > 0){
                        auto K233344 = get_Kijk(K233_344);
                        summerB_112_123_123 += leadingijk*POW3(sigma[i]*sigma[j])*POW5(sigma[k])/(sigmaij*POW2(sigmaik*sigmajk))*mubar2[i]*mubar2[j]*Qbar2[k]*K233344;
                    }
                    if (std::abs(mubar2[i]*Qbar2[j]*Qbar2[k]) > 0){
                        auto K334445 = get_Kijk_334445(K334_445);
                        summerB_123_123_224 += leadingijk*POW3(sigma[i])*POW5(sigma[j]*sigma[k])/(POW2(sigmaij*sigmaik)*POW3(sigmajk))*mubar2[i]*Qbar2[j]*Qbar2[k]*K334445;
                    }
                    if (std::abs(Qbar2[i]*Qbar2[j]*Qbar2[k]) > 0){
                        auto K444555 = get_Kijk(K444_555);
                        summerB_224_224_224 += leadingijk*POW5(sigma[i]*sigma[j]*sigma[k])/(POW3(sigmaij*sigmaik*sigmajk))*Qbar2[i]*Qbar2[j]*Qbar2[k]*K444555;
                    }
                }
            }
        }
        auto alpha3A_112_112_224 = 8.0*PI_*rhoN/25.0*summerA_112_112_224;
        auto alpha3A_112_123_213 = 8.0*PI_*rhoN/75.0*summerA_112_123_213;
        auto alpha3A_123_123_224 = 8.0*PI_*rhoN/35.0*summerA_123_123_224;
        auto alpha3A_224_224_224 = 144.0*PI_*rhoN/245.0*summerA_224_224_224;
        
        auto alpha3A = forceeval(3.0*alpha3A_112_112_224 + 6.0*alpha3A_112_123_213 + 6.0*alpha3A_123_123_224 + alpha3A_224_224_224);
        
        auto alpha3B_112_112_112 = 32.0*POW3(PI_)*POW2(rhoN)/135.0*sqrt(14*PI_/5.0)*summerB_112_112_112;
        auto alpha3B_112_123_123 = 64.0*POW3(PI_)*POW2(rhoN)/315.0*sqrt(3.0*PI_)*summerB_112_123_123;
        auto alpha3B_123_123_224 = -32.0*POW3(PI_)*POW2(rhoN)/45.0*sqrt(22.0*PI_/63.0)*summerB_123_123_224;
        auto alpha3B_224_224_224 = 32.0*POW3(PI_)*POW2(rhoN)/2025.0*sqrt(2002.0*PI_)*summerB_224_224_224;
        
        auto alpha3B = forceeval(alpha3B_112_112_112 + 3.0*alpha3B_112_123_123 + 3.0*alpha3B_123_123_224 + alpha3B_224_224_224);
        
        return forceeval(alpha3A + alpha3B);
    }
    
    /***
     * \brief Get the contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rhoN, const VecType& mole_fractions) const {
        using type = std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])>;
        
        // Calculate the effective reduced diameter (cubed) to be used for evaluation
        // Eq. 24 from Gubbins
        type sigma_x3 = 0.0;
        auto N = mole_fractions.size();
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                auto sigmaij = (sigma_m[i] + sigma_m[j])/2;
                sigma_x3 += mole_fractions[i]*mole_fractions[j]*POW3(sigmaij);
            }
        }
        type rhostar = forceeval(rhoN*sigma_x3);
        
        type alpha2 = 0.0, alpha3 = 0.0, alpha = 0.0;
        if (has_a_polar){
            alpha2 = get_alpha2(T, rhoN, rhostar, mole_fractions);
            alpha3 = get_alpha3(T, rhoN, rhostar, mole_fractions);
            alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
        }
        
        using alpha2_t = decltype(alpha2);
        using alpha3_t = decltype(alpha3);
        using alpha_t = decltype(alpha);
        struct Terms{
            alpha2_t alpha2;
            alpha3_t alpha3;
            alpha_t alpha;
        };
        return Terms{alpha2, alpha3, alpha};
    }
};

/// The variant containing the multipolar types that can be provided
using multipolar_contributions_variant = std::variant<
    MultipolarContributionGrossVrabec,
    MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>,
    MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>
>;

}
}
