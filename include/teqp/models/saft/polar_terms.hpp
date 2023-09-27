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
#include "teqp/math/pow_templates.hpp"
#include <variant>

namespace teqp{

namespace SAFTpolar{

/// Eq. 10 from Gross and Vrabec
template <typename Eta, typename MType, typename TType>
auto get_JDD_2ij(const Eta& eta, const MType& mij, const TType& Tstarij) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(5) << 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(5) << 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(5) << -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575).finished();

    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(5) << 0.2187939, -1.1896431, 1.1626889, 0, 0).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(5) << -0.5873164, 1.2489132, -0.5085280, 0, 0).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(5) << 3.4869576, -14.915974, 15.372022, 0, 0).finished();
    
    std::common_type_t<Eta, MType, TType> summer = 0.0;
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
    
    std::common_type_t<Eta, MType, TType> summer = 0.0;
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


/// Eq. 16 from Vrabec and Gross, JPCB, 2008. doi: 10.1021/jp072619u
template <typename Eta, typename MType, typename TType>
auto get_JDQ_2ij(const Eta& eta, const MType& mij, const TType& Tstarij) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(4) << 0.6970950, -0.6335541, 2.9455090, -1.4670273).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(4) << -0.6734593, -1.4258991, 4.1944139, 1.0266216).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(4) << 0.6703408, -4.3384718, 7.2341684, 0).finished();

    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(4) << -0.4840383, 1.9704055, -2.1185727, 0).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(4) << 0.6765101, -3.0138675, 0.4674266, 0).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(4) << -1.1675601, 2.1348843, 0, 0).finished();
    
    std::common_type_t<Eta, MType, TType> summer = 0.0;
    for (auto n = 0; n < 4; ++n){
        auto anij = a_0[n] + (mij-1)/mij*a_1[n] + (mij-1)/mij*(mij-2)/mij*a_2[n]; // Eq. 18
        auto bnij = b_0[n] + (mij-1)/mij*b_1[n] + (mij-1)/mij*(mij-2)/mij*b_2[n]; // Eq. 19
        summer += (anij + bnij/Tstarij)*pow(eta, n);
    }
    return forceeval(summer);
}


/// Eq. 17 from Vrabec and Gross, JPCB, 2008. doi: 10.1021/jp072619u
template <typename Eta, typename MType>
auto get_JDQ_3ijk(const Eta& eta, const MType& mijk) {
    static Eigen::ArrayXd c_0 = (Eigen::ArrayXd(4) << 7.846431, 33.42700, 4.689111, 0).finished();
    static Eigen::ArrayXd c_1 = (Eigen::ArrayXd(4) << -20.72202, -58.63904, -1.764887, 0).finished();
    std::common_type_t<Eta, MType> summer = 0.0;
    for (auto n = 0; n < 4; ++n){
        auto cnijk = c_0[n] + (mijk-1)/mijk*c_1[n]; // Eq. 20
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
                    
                    auto Tstarij = forceeval(T/epskij);
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
 * \brief The quadrupolar contribution from Gross, AICHEJ, doi: 10.1002/aic.10502
 *
 */
class QuadrupolarContributionGross {
private:
    const Eigen::ArrayXd m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ;
    
public:
    const bool has_a_polar;
    QuadrupolarContributionGross(const Eigen::ArrayX<double> &m, const Eigen::ArrayX<double> &sigma_Angstrom, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayX<double> &Qstar2, const Eigen::ArrayX<double> &nQ) : m(m), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), Qstar2(Qstar2), nQ(nQ), has_a_polar(Qstar2.cwiseAbs().sum() > 0) {
        // Check lengths match
        if (m.size() != Qstar2.size()){
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (m.size() != nQ.size()){
            throw teqp::InvalidArgument("bad size of n");
        }
    }
    QuadrupolarContributionGross& operator=( const QuadrupolarContributionGross& ) = delete; // non copyable
    
    /// Eq. 9 from Gross, AICHEJ, doi: 10.1002/aic.10502
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
                    
                    auto Tstarij = forceeval(T/epskij);
                    auto mij = std::min(sqrt(m[i]*m[j]), 2.0);
                    summer += x[i]*x[j]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*POW5(sigma[i]*sigma[j])/POW7(sigmaij)*ninj*Qstar2[i]*Qstar2[j]*get_JQQ_2ij(eta, mij, Tstarij);
                }
            }
        }
        return forceeval(-static_cast<double>(EIGEN_PI)*POW2(3.0/4.0)*rhoN_A3*summer);
    }
    
    /// Eq. 10 from Gross, AICHEJ, doi: 10.1002/aic.10502
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha3QQ(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const std::size_t N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                for (std::size_t k = 0; k < N; ++k){
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


/***
 * \brief The cross dipolar-quadrupolar contribution from Vrabec and Gross
 *  doi: https://doi.org/10.1021/jp072619u
 *
 */
class DipolarQuadrupolarContributionVrabecGross {
private:
    const Eigen::ArrayXd m, sigma_Angstrom, epsilon_over_k, mustar2, nmu, Qstar2, nQ;
    
public:
    DipolarQuadrupolarContributionVrabecGross(
      const Eigen::ArrayX<double> &m,
      const Eigen::ArrayX<double> &sigma_Angstrom,
      const Eigen::ArrayX<double> &epsilon_over_k,
      const Eigen::ArrayX<double> &mustar2,
      const Eigen::ArrayX<double> &nmu,
      const Eigen::ArrayX<double> &Qstar2,
      const Eigen::ArrayX<double> &nQ
    ) : m(m), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), mustar2(mustar2), nmu(nmu), Qstar2(Qstar2), nQ(nQ) {
        // Check lengths match
        if (m.size() != Qstar2.size()){
            throw teqp::InvalidArgument("bad size of Qstar2");
        }
        if (m.size() != nQ.size()){
            throw teqp::InvalidArgument("bad size of nQ");
        }
        if (m.size() != mustar2.size()){
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (m.size() != nmu.size()){
            throw teqp::InvalidArgument("bad size of n");
        }
        if (Qstar2.cwiseAbs().sum() == 0 || mustar2.cwiseAbs().sum() == 0){
            throw teqp::InvalidArgument("Invalid to have either missing polar or quadrupolar term in cross-polar term");
        }
    }
    DipolarQuadrupolarContributionVrabecGross& operator=( const DipolarQuadrupolarContributionVrabecGross& ) = delete; // non copyable
    
    /// Eq. 14 from Vrabec and Gross
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha2DQ(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const auto N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                auto ninj = nmu[i]*nQ[j];
                if (ninj > 0){
                    // Lorentz-Berthelot mixing rules
                    auto epskij = sqrt(epsilon_over_k[i]*epsilon_over_k[j]);
                    auto sigmaij = (sigma[i]+sigma[j])/2;
                    
                    auto Tstarij = forceeval(T/epskij);
                    auto mij = std::min(sqrt(m[i]*m[j]), 2.0);
                    summer += x[i]*x[j]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*POW3(sigma[i])*POW5(sigma[j])/POW5(sigmaij)*ninj*mustar2[i]*Qstar2[j]*get_JDQ_2ij(eta, mij, Tstarij);
                }
            }
        }
        return forceeval(-static_cast<double>(EIGEN_PI)*9.0/4.0*rhoN_A3*summer);
    }
    
    /// Eq. 15 from Vrabec and Gross
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha3DQ(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        const auto& sigma = sigma_Angstrom; // concision
        const std::size_t N = mole_fractions.size();
        std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])> summer = 0.0;
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                for (std::size_t k = 0; k < N; ++k){
                    auto ninjnk1 = nmu[i]*nmu[j]*nQ[k];
                    auto ninjnk2 = nmu[i]*nQ[j]*nQ[k];
                    if (ninjnk1 > 0 || ninjnk2 > 0){
                        // Lorentz-Berthelot mixing rules for sigma
                        auto sigmaij = (sigma[i]+sigma[j])/2;
                        auto sigmaik = (sigma[i]+sigma[k])/2;
                        auto sigmajk = (sigma[j]+sigma[k])/2;
                        
                        auto mijk = std::min(pow(m[i]*m[j]*m[k], 1.0/3.0), 2.0);
                        double alpha_GV = 1.19374; // Table 3
                        auto polars = ninjnk1*mustar2[i]*mustar2[j]*Qstar2[k] + ninjnk2*alpha_GV*mustar2[i]*Qstar2[j]*Qstar2[k];
                        summer += x[i]*x[j]*x[k]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*epsilon_over_k[k]/T*POW4(sigma[i]*sigma[j]*sigma[k])/POW2(sigmaij*sigmaik*sigmajk)*polars*get_JDQ_3ijk(eta, mijk);
                    }
                }
            }
        }
        return forceeval(-POW2(rhoN_A3)*summer);
    }
    
    /***
     * \brief Get the cross-polar contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        auto alpha2 = get_alpha2DQ(T, rho_A3, eta, mole_fractions);
        auto alpha3 = get_alpha3DQ(T, rho_A3, eta, mole_fractions);
        auto alpha = forceeval(alpha2/(1.0-alpha3/alpha2));
        
        using alpha2_t = decltype(alpha2);
        using alpha3_t = decltype(alpha3);
        using alpha_t = decltype(alpha);
        struct DipolarQuadrupolarContributionTerms{
            alpha2_t alpha2;
            alpha3_t alpha3;
            alpha_t alpha;
        };
        return DipolarQuadrupolarContributionTerms{alpha2, alpha3, alpha};
    }
};

enum class multipolar_argument_spec {
    TK_rhoNA3_packingfraction_molefractions,
    TK_rhoNm3_rhostar_molefractions
};

enum class multipolar_rhostar_approach {
    kInvalid,
    use_packing_fraction,
    calculate_Gubbins_rhostar
};

// map multipolar_rhostar_approach values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM( multipolar_rhostar_approach, {
    {multipolar_rhostar_approach::kInvalid, nullptr},
    {multipolar_rhostar_approach::use_packing_fraction, "use_packing_fraction"},
    {multipolar_rhostar_approach::calculate_Gubbins_rhostar, "calculate_Gubbins_rhostar"},
})

template<typename type>
struct MultipolarContributionGrossVrabecTerms{
    type alpha2DD;
    type alpha3DD;
    type alphaDD;
    type alpha2QQ;
    type alpha3QQ;
    type alphaQQ;
    type alpha2DQ;
    type alpha3DQ;
    type alphaDQ;
    type alpha;
};

class MultipolarContributionGrossVrabec{
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNA3_packingfraction_molefractions;
    
    const std::optional<DipolarContributionGrossVrabec> di;
    const std::optional<QuadrupolarContributionGross> quad;
    const std::optional<DipolarQuadrupolarContributionVrabecGross> diquad;
    
    MultipolarContributionGrossVrabec(
      const Eigen::ArrayX<double> &m,
      const Eigen::ArrayX<double> &sigma_Angstrom,
      const Eigen::ArrayX<double> &epsilon_over_k,
      const Eigen::ArrayX<double> &mustar2,
      const Eigen::ArrayX<double> &nmu,
      const Eigen::ArrayX<double> &Qstar2,
      const Eigen::ArrayX<double> &nQ)
    : di((((nmu*mustar2 > 0).cast<int>().sum() > 0) ? decltype(di)(DipolarContributionGrossVrabec(m, sigma_Angstrom, epsilon_over_k, mustar2, nmu)) : std::nullopt)),
      quad((((nQ*Qstar2 > 0).cast<int>().sum() > 0) ? decltype(quad)(QuadrupolarContributionGross(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ)) : std::nullopt)),
      diquad((di && quad) ? decltype(diquad)(DipolarQuadrupolarContributionVrabecGross(m, sigma_Angstrom, epsilon_over_k, mustar2, nmu, Qstar2, nQ)) : std::nullopt)
    {};
    
    template<typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        
        using type = std::common_type_t<TTYPE, RhoType, EtaType, decltype(mole_fractions[0])>;
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
        
        type alpha2DQ = 0.0, alpha3DQ = 0.0, alphaDQ = 0.0;
        if (diquad){
            alpha2DQ = diquad.value().get_alpha2DQ(T, rho_A3, eta, mole_fractions);
            alpha3DQ = diquad.value().get_alpha3DQ(T, rho_A3, eta, mole_fractions);
            alphaDQ = forceeval(alpha2DQ/(1.0-alpha3DQ/alpha2DQ));
        }
        
        auto alpha = forceeval(alphaDD + alphaQQ + alphaDQ);
        
        return MultipolarContributionGrossVrabecTerms<type>{alpha2DD, alpha3DD, alphaDD, alpha2QQ, alpha3QQ, alphaQQ, alpha2DQ, alpha3DQ, alphaDQ, alpha};
    }
    
};

template<typename type>
struct MultipolarContributionGubbinsTwuTermsGT{
    type alpha2;
    type alpha3;
    type alpha;
};

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
    const double epsilon_0 = 8.8541878128e-12; // https://en.wikipedia.org/wiki/Vacuum_permittivity, in F/m, or C^2⋅N^−1⋅m^−2
    const double k_e = 1.0/(4.0*PI_*epsilon_0); // coulomb constant, with units of N m^2 / C^2
    
    multipolar_rhostar_approach approach = multipolar_rhostar_approach::use_packing_fraction;
    
    // These values were adjusted in the model of Paricaud, JPCB, 2023
    /// The C3b is the C of Paricaud, and C3 is the tau of Paricaud. They were renamed to be not cause confusion with the multifluid modeling approach
    const double C3;
    const double C3b;
    
    double get_C3(const std::optional<nlohmann::json>& flags){
        if (flags){ return flags.value().value("C3", 1.0); }
        return 1.0;
    }
    double get_C3b(const std::optional<nlohmann::json>& flags){
        if (flags){ return flags.value().value("C3b", 1.0); }
        return 1.0;
    }
    multipolar_rhostar_approach get_approach(const std::optional<nlohmann::json>& flags){
        if (flags){ return flags.value().value("approach", multipolar_rhostar_approach::use_packing_fraction); }
        return multipolar_rhostar_approach::use_packing_fraction;
    }
    
public:
    MultipolarContributionGrayGubbins(const Eigen::ArrayX<double> &sigma_m, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::MatrixXd& SIGMAIJ, const Eigen::MatrixXd& EPSKIJ, const Eigen::ArrayX<double> &mu, const Eigen::ArrayX<double> &Q, const std::optional<nlohmann::json>& flags) : sigma_m(sigma_m), epsilon_over_k(epsilon_over_k), SIGMAIJ(SIGMAIJ), EPSKIJ(EPSKIJ), mu(mu), Q(Q), mu2(mu.pow(2)), Q2(Q.pow(2)), Q3(Q.pow(3)), has_a_polar(Q.cwiseAbs().sum() > 0 || mu.cwiseAbs().sum() > 0), sigma_m3(sigma_m.pow(3)), sigma_m5(sigma_m.pow(5)), approach(get_approach(flags)), C3(get_C3(flags)), C3b(get_C3b(flags)) {
        // Check lengths match
        if (sigma_m.size() != mu.size()){
            throw teqp::InvalidArgument("bad size of mu");
        }
        if (sigma_m.size() != Q.size()){
            throw teqp::InvalidArgument("bad size of Q");
        }
    }
    MultipolarContributionGrayGubbins& operator=( const MultipolarContributionGrayGubbins& ) = delete; // non copyable
    
    /// Appendix B of Gray and Gubbins
    template<typename Jintegral, typename TTYPE, typename RhoStarType>
    auto get_In(const Jintegral& J, int n, double sigmaij, const TTYPE& Tstar, const RhoStarType& rhostar) const{
        return 4.0*PI_/pow(sigmaij, n-3)*J.get_J(Tstar, rhostar);
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha2(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const auto& x = mole_fractions; // concision
        
        const std::size_t N = mole_fractions.size();
        
        std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])> summer = 0.0;
        
        const auto k_B = 1.380649e-23;
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const Eigen::ArrayXd muprime2 = mu2; // Polarizability ignored for now
        const auto z1 = (1.0/3.0*muprime2).cast<TTYPE>() * beta;
        const auto z2 = 0.0*z1;
//        const auto gamma = 0.0*z1; // Anisotropy of polarizability also ignored
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                TTYPE Tstarij = forceeval(T/EPSKIJ(i, j));
                double sigmaij = SIGMAIJ(i,j);
                summer += x[i]*x[j]*(
                     3.0/2.0*(z1[i]*z1[j] - z2[i]*z2[j])*get_In(J6, 6, sigmaij, Tstarij, rhostar)
                    + 3.0/2.0*z1[i]*beta*Q2[j]*get_In(J8, 8, sigmaij, Tstarij, rhostar)
                    +7.0/10*beta*beta*Q2[i]*Q2[j]*get_In(J10, 10, sigmaij, Tstarij, rhostar)
                );
            }
        }
        return forceeval(-rhoN*k_e*k_e*summer); // The factor of k_e^2 takes us from CGS to SI units
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType>
    auto get_alpha3(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions) const{
        const VecType& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0])>;
        type summer_a = 0.0, summer_b = 0.0;
        
        const auto k_B = 1.380649e-23;
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const Eigen::ArrayXd muprime2 = mu2; // Polarizability ignored for now
        const auto z1 = (1.0/3.0*muprime2).cast<TTYPE>() * beta;
        const auto z2 = 0.0*z1;
        const auto gamma = 0.0*z1; // Anisotropy of polarizability also ignored
        
        /// Following Appendix B of Gray and Gubbins
        const double PI3 = POW3(PI_);
        auto Immm = [&](std::size_t i, std::size_t j, std::size_t k, const auto& T, const auto& rhostar){
            auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
            const double coeff = 64.0*PI3/5.0*sqrt(14*PI_/5.0)/SIGMAIJ(i,j)/SIGMAIJ(i,k)/SIGMAIJ(j,k);
            return coeff*get_Kijk(K222_333, rhostar, Tstarij, Tstarik, Tstarjk);
        };
        auto ImmQ = [&](std::size_t i, std::size_t j, std::size_t k, const auto& T, const auto& rhostar){
            auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
            const double coeff = 2048.0*PI3/7.0*sqrt(3.0*PI_)/SIGMAIJ(i,j)/POW2(SIGMAIJ(i,k)*SIGMAIJ(j,k));
            return coeff*get_Kijk(K233_344, rhostar, Tstarij, Tstarik, Tstarjk);
        };
        auto ImQQ = [&](std::size_t i, std::size_t j, std::size_t k, const auto& T, const auto& rhostar){
            auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
            const double coeff = -4096.0*PI3/9.0*sqrt(22.0*PI_/7.0)/POW2(SIGMAIJ(i,j)*SIGMAIJ(i,k))/POW3(SIGMAIJ(j,k));
            return coeff*get_Kijk_334445(K334_445, rhostar, Tstarij, Tstarik, Tstarjk);
        };
        auto IQQQ = [&](std::size_t i, std::size_t j, std::size_t k, const auto& T, const auto& rhostar){
            auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
            const double coeff = 8192.0*PI3/81.0*sqrt(2002.0*PI_)/POW3(SIGMAIJ(i,j)*SIGMAIJ(i,k)*SIGMAIJ(j,k));
            return coeff*get_Kijk(K444_555, rhostar, Tstarij, Tstarik, Tstarjk);
        };
        
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

/// The variant containing the multipolar types that can be provided
using multipolar_contributions_variant = std::variant<
    MultipolarContributionGrossVrabec,
    MultipolarContributionGrayGubbins<GubbinsTwuJIntegral, GubbinsTwuKIntegral>,
    MultipolarContributionGrayGubbins<GottschalkJIntegral, GottschalkKIntegral>,
    MultipolarContributionGrayGubbins<LuckasJIntegral, LuckasKIntegral>,
    MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>,
    MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>,
    MultipolarContributionGubbinsTwu<GottschalkJIntegral, GottschalkKIntegral>
>;

}
}
