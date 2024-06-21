#pragma once
#include "types.hpp"
#include "teqp/math/pow_templates.hpp"

namespace teqp::saft::polar_terms::GrossVrabec{

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

}
