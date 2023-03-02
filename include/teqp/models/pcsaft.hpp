/***
 
 \brief This file contains the contributions that can be composed together to form SAFT models

*/

#pragma once

#include "nlohmann/json.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/constants.hpp"

namespace teqp {
namespace PCSAFT {

/// Coefficients for one fluid
struct SAFTCoeffs {
    std::string name; ///< Name of fluid
    double m = -1, ///< number of segments
        sigma_Angstrom = -1, ///< [A] segment diameter
        epsilon_over_k = -1; ///< [K] depth of pair potential divided by Boltzman constant
    std::string BibTeXKey; ///< The BibTeXKey for the reference for these coefficients
};

/// Manager class for PCSAFT coefficients
class PCSAFTLibrary {
    std::map<std::string, SAFTCoeffs> coeffs;
public:
    PCSAFTLibrary() {
        insert_normal_fluid("Methane", 1.0000, 3.7039, 150.03, "Gross-IECR-2001");
        insert_normal_fluid("Ethane", 1.6069, 3.5206, 191.42, "Gross-IECR-2001");
        insert_normal_fluid("Propane", 2.0020, 3.6184, 208.11, "Gross-IECR-2001");
    }
    void insert_normal_fluid(const std::string& name, double m, const double sigma_Angstrom, const double epsilon_over_k, const std::string& BibTeXKey) {
        SAFTCoeffs coeff;
        coeff.name = name;
        coeff.m = m;
        coeff.sigma_Angstrom = sigma_Angstrom;
        coeff.epsilon_over_k = epsilon_over_k;
        coeff.BibTeXKey = BibTeXKey;
        coeffs.insert(std::pair<std::string, SAFTCoeffs>(name, coeff));
    }
    const auto& get_normal_fluid(const std::string& name) {
        auto it = coeffs.find(name);
        if (it != coeffs.end()) {
            return it->second;
        }
        else {
            throw std::invalid_argument("Bad name:" + name);
        }
    }
    auto get_coeffs(const std::vector<std::string>& names){
        std::vector<SAFTCoeffs> c;
        for (auto n : names){
            c.push_back(get_normal_fluid(n));
        }
        return c;
    }
};

/// Eqn. A.11
/// Erratum: should actually be 1/RHS of equation A.11 according to sample
/// FORTRAN code
template <typename Eta, typename Mbar>
auto C1(const Eta& eta, Mbar mbar) {
    return forceeval(1.0 / (1.0
        + mbar * (8.0 * eta - 2.0 * eta * eta) / pow(1.0 - eta, 4)
        + (1.0 - mbar) * (20.0 * eta - 27.0 * eta * eta + 12.0 * pow(eta, 3) - 2.0 * pow(eta, 4)) / pow((1.0 - eta) * (2.0 - eta), 2)));
}
/// Eqn. A.31
template <typename Eta, typename Mbar>
auto C2(const Eta& eta, Mbar mbar) {
    return forceeval(-pow(C1(eta, mbar), 2) * (
        mbar * (-4.0 * eta * eta + 20.0 * eta + 8.0) / pow(1.0 - eta, 5)
        + (1.0 - mbar) * (2.0 * eta * eta * eta + 12.0 * eta * eta - 48.0 * eta + 40.0) / pow((1.0 - eta) * (2.0 - eta), 3)
        ));
}
/// Eqn. A.18
template<typename TYPE>
auto get_a(TYPE mbar) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(7) << 0.9105631445, 0.6361281449, 2.6861347891, -26.547362491, 97.759208784, -159.59154087, 91.297774084).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(7) << -0.3084016918, 0.1860531159, -2.5030047259, 21.419793629, -65.255885330, 83.318680481, -33.746922930).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(7) << -0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131, -4.1302112531, 13.776631870, -8.6728470368).finished();
    return forceeval(a_0.cast<TYPE>().array() + ((mbar - 1.0) / mbar) * a_1.cast<TYPE>().array() + ((mbar - 1.0) / mbar * (mbar - 2.0) / mbar) * a_2.cast<TYPE>().array()).eval();
}
/// Eqn. A.19
template<typename TYPE>
auto get_b(TYPE mbar) {
    // See https://stackoverflow.com/a/35170514/1360263
    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(7) << 0.7240946941, 2.2382791861, -4.0025849485, -21.003576815, 26.855641363, 206.55133841, -355.60235612).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(7) << -0.5755498075, 0.6995095521, 3.8925673390, -17.215471648, 192.67226447, -161.82646165, -165.20769346).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(7) << 0.0976883116, -0.2557574982, -9.1558561530, 20.642075974, -38.804430052, 93.626774077, -29.666905585).finished();
    return forceeval(b_0.cast<TYPE>().array() + (mbar - 1.0) / mbar * b_1.cast<TYPE>().array() + (mbar - 1.0) / mbar * (mbar - 2.0) / mbar * b_2.cast<TYPE>().array()).eval();
}
/// Residual contribution to alphar from hard-sphere (Eqn. A.6)
template<typename VecType>
auto get_alphar_hs(const VecType& zeta) {
    auto Upsilon = 1.0 - zeta[3];
    return forceeval(1.0 / zeta[0] * (3.0 * zeta[1] * zeta[2] / Upsilon
        + zeta[2] * zeta[2] * zeta[2] / zeta[3] / Upsilon / Upsilon
        + (zeta[2] * zeta[2] * zeta[2] / (zeta[3] * zeta[3]) - zeta[0]) * log(1.0 - zeta[3])
        ));
}

/// Residual contribution from hard-sphere (Eqn. A.26)
template<typename VecType>
auto Z_hs(const VecType& zeta) {
    auto Upsilon = 1.0 - zeta[3];
    return forceeval(zeta[3] / Upsilon
        + 3.0 * zeta[1] * zeta[2] / (zeta[0] * pow(Upsilon, 2))
        + (3.0 * pow(zeta[2], 3) - zeta[3] * pow(zeta[2], 3)) / (zeta[0] * pow(Upsilon, 3)));
}
/// Derivative term from Eqn. A.27
template<typename zVecType, typename dVecType>
auto rho_A3_dgij_HS_drhoA3(const zVecType& zeta, const dVecType& d,
    std::size_t i, std::size_t j) {
    auto Upsilon = 1.0 - zeta[3];
    return forceeval(zeta[3] / pow(Upsilon, 2)
        + d[i] * d[j] / (d[i] + d[j]) * (3.0 * zeta[2] / pow(Upsilon, 2) + 6.0 * zeta[2] * zeta[3] / pow(Upsilon, 3))
        + pow(d[i] * d[j] / (d[i] + d[j]), 2) * (4.0 * pow(zeta[2], 2) / pow(Upsilon, 3) + 6.0 * pow(zeta[2], 2) * zeta[3] / pow(Upsilon, 4)));
}
/// Term from Eqn. A.7
template<typename zVecType, typename dVecType>
auto gij_HS(const zVecType& zeta, const dVecType& d,
    std::size_t i, std::size_t j) {
    auto Upsilon = 1.0 - zeta[3];
    return forceeval(1.0 / (Upsilon)+d[i] * d[j] / (d[i] + d[j]) * 3.0 * zeta[2] / pow(Upsilon, 2)
        + pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2.0 * pow(zeta[2], 2) / pow(Upsilon, 3));
}
/// Eqn. A.16, Eqn. A.29
template <typename Eta, typename MbarType>
auto get_I1(const Eta& eta, MbarType mbar) {
    auto avec = get_a(mbar);
    Eta summer_I1 = 0.0, summer_etadI1deta = 0.0;
    for (std::size_t i = 0; i < 7; ++i) {
        auto increment = avec(i) * pow(eta, static_cast<int>(i));
        summer_I1 = summer_I1 + increment;
        summer_etadI1deta = summer_etadI1deta + increment * (i + 1.0);
    }
    return std::make_tuple(forceeval(summer_I1), forceeval(summer_etadI1deta));
}
/// Eqn. A.17, Eqn. A.30
template <typename Eta, typename MbarType>
auto get_I2(const Eta& eta, MbarType mbar) {
    auto bvec = get_b(mbar);
    Eta summer_I2 = 0.0 * eta, summer_etadI2deta = 0.0 * eta;
    for (std::size_t i = 0; i < 7; ++i) {
        auto increment = bvec(i) * pow(eta, static_cast<int>(i));
        summer_I2 = summer_I2 + increment;
        summer_etadI2deta = summer_etadI2deta + increment * (i + 1.0);
    }
    return std::make_tuple(forceeval(summer_I2), forceeval(summer_etadI2deta));
}

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

/**
Sum up three array-like objects that can each have different container types and value types
*/
template<typename VecType1, typename NType>
auto powvec(const VecType1& v1, NType n) {
    auto o = v1;
    for (auto i = 0; i < v1.size(); ++i) {
        o[i] = pow(v1[i], n);
    }
    return o;
}

/**
Sum up the coefficient-wise product of three array-like objects that can each have different container types and value types
*/
template<typename VecType1, typename VecType2, typename VecType3>
auto sumproduct(const VecType1& v1, const VecType2& v2, const VecType3& v3) {
    using ResultType = typename std::common_type_t<decltype(v1[0]), decltype(v2[0]), decltype(v3[0])>;
    return forceeval((v1.template cast<ResultType>().array() * v2.template cast<ResultType>().array() * v3.template cast<ResultType>().array()).sum());
}

/// Parameters for model evaluation
template<typename NumType, typename ProductType>
class SAFTCalc {
public:
    // Just temperature dependent things
    Eigen::ArrayX<NumType> d;

    // These things also have composition dependence
    ProductType m2_epsilon_sigma3_bar, ///< Eq. A. 12
                m2_epsilon2_sigma3_bar; ///< Eq. A. 13
};

/** A class used to evaluate mixtures using PC-SAFT model

This is the classical Gross and Sadowski model from 2001: https://doi.org/10.1021/ie0003887
 
with the error fixed as noted in a comment: https://doi.org/10.1021/acs.iecr.9b01515
*/
class PCSAFTMixture {
protected:
    Eigen::ArrayX<double> m, ///< number of segments
        mminus1, ///< m-1
        sigma_Angstrom, ///< 
        epsilon_over_k; ///< depth of pair potential divided by Boltzman constant
    std::vector<std::string> names;
    Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix

    void check_kmat(std::size_t N) {
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows and columns are not identical");
        }
        if (kmat.cols() == 0) {
            kmat.resize(N, N); kmat.setZero();
        }
        else if (kmat.cols() != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components");
        }
    };
    auto get_coeffs_from_names(const std::vector<std::string> &names){
        PCSAFTLibrary library;
        return library.get_coeffs(names);
    }
public:
    PCSAFTMixture(const std::vector<std::string> &names, const Eigen::ArrayXXd& kmat = {}) : PCSAFTMixture(get_coeffs_from_names(names), kmat){};
    PCSAFTMixture(const std::vector<SAFTCoeffs> &coeffs, const Eigen::ArrayXXd &kmat = {}) : kmat(kmat)
    {
        check_kmat(coeffs.size());

        m.resize(coeffs.size());
        mminus1.resize(coeffs.size());
        sigma_Angstrom.resize(coeffs.size());
        epsilon_over_k.resize(coeffs.size());
        names.resize(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            m[i] = coeff.m;
            mminus1[i] = m[i] - 1;
            sigma_Angstrom[i] = coeff.sigma_Angstrom;
            epsilon_over_k[i] = coeff.epsilon_over_k;
            names[i] = coeff.name;
            i++;
        }
    };
    auto get_m() const { return m; }
    auto get_sigma_Angstrom() const { return sigma_Angstrom; }
    auto get_epsilon_over_k_K() const { return epsilon_over_k; }
    auto get_kmat() const { return kmat; }

    auto print_info() {
        std::string s = std::string("i m sigma / A e/kB / K \n  ++++++++++++++") + "\n";
        for (auto i = 0; i < m.size(); ++i) {
            s += std::to_string(i) + " " + std::to_string(m[i]) + " " + std::to_string(sigma_Angstrom[i]) + " " + std::to_string(epsilon_over_k[i]) + "\n";
        }
        return s;
    }
    template<typename VecType>
    double max_rhoN(const double T, const VecType& mole_fractions) const {
        auto N = mole_fractions.size();
        Eigen::ArrayX<double> d(N);
        for (auto i = 0; i < N; ++i) {
            d[i] = sigma_Angstrom[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
        }
        return 6 * 0.74 / EIGEN_PI / (mole_fractions*m*powvec(d, 3)).sum()*1e30; // particles/m^3
    }
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {

        std::size_t N = m.size();

        if (mole_fractions.size() != N) {
            throw std::invalid_argument("Length of mole_fractions (" + std::to_string(mole_fractions.size()) + ") is not the length of components (" + std::to_string(N) + ")");
        }

        using TRHOType = std::common_type_t<std::decay_t<TTYPE>, std::decay_t<RhoType>, std::decay_t<decltype(mole_fractions[0])>, std::decay_t<decltype(m[0])>>;

        SAFTCalc<TTYPE, TRHOType> c;
        c.m2_epsilon_sigma3_bar = static_cast<TRHOType>(0.0);
        c.m2_epsilon2_sigma3_bar = static_cast<TRHOType>(0.0);
        c.d.resize(N); 
        for (std::size_t i = 0; i < N; ++i) {
            c.d[i] = sigma_Angstrom[i]*(1.0 - 0.12 * exp(-3.0*epsilon_over_k[i]/T)); // [A]
            for (std::size_t j = 0; j < N; ++j) {
                // Eq. A.5
                auto sigma_ij = 0.5 * sigma_Angstrom[i] + 0.5 * sigma_Angstrom[j];
                auto eij_over_k = sqrt(epsilon_over_k[i] * epsilon_over_k[j]) * (1.0 - kmat(i,j));
                c.m2_epsilon_sigma3_bar = c.m2_epsilon_sigma3_bar + mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * eij_over_k / T * pow(sigma_ij, 3);
                c.m2_epsilon2_sigma3_bar = c.m2_epsilon2_sigma3_bar + mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * pow(eij_over_k / T, 2) * pow(sigma_ij, 3);
            }
        }
        auto mbar = (mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()).sum();

        /// Convert from molar density to number density in molecules/Angstrom^3
        RhoType rho_A3 = rhomolar * N_A * 1e-30; //[molecules (not moles)/A^3]

        constexpr double MY_PI = EIGEN_PI;
        double pi6 = (MY_PI / 6.0);

        /// Evaluate the components of zeta
        using ta = std::common_type_t<decltype(pi6), decltype(m[0]), decltype(c.d[0]), decltype(rho_A3)>;
        std::vector<ta> zeta(4);
        for (std::size_t n = 0; n < 4; ++n) {
            // Eqn A.8
            auto dn = c.d.pow(n).eval();
            TRHOType xmdn = forceeval((mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()*dn.template cast<TRHOType>().array()).sum());
            zeta[n] = forceeval(pi6*rho_A3*xmdn);
        }

        /// Packing fraction is the 4-th value in zeta, at index 3
        const auto &eta = zeta[3];
        
        auto [I1, etadI1deta] = get_I1(eta, mbar);
        auto [I2, etadI2deta] = get_I2(eta, mbar);

        // Hard chain contribution from G&S
        using tt = std::common_type_t<decltype(zeta[0]), decltype(c.d[0])>;
        Eigen::ArrayX<tt> lngii_hs(mole_fractions.size());
        for (auto i = 0; i < lngii_hs.size(); ++i) {
            lngii_hs[i] = log(gij_HS(zeta, c.d, i, i));
        }
        auto alphar_hc = mbar * get_alphar_hs(zeta) - sumproduct(mole_fractions, mminus1, lngii_hs); // Eq. A.4
        
        // Dispersive contribution
        auto alphar_disp = -2 * MY_PI * rho_A3 * I1 * c.m2_epsilon_sigma3_bar - MY_PI * rho_A3 * mbar * C1(eta, mbar) * I2 * c.m2_epsilon2_sigma3_bar;
        
        return forceeval(alphar_hc + alphar_disp);
    }
};

/***
 * \brief A dipolar model formed of the hard-chain contribution from vanilla PC-SAFT plus the dipolar contribution from Gross and Vrabec
 */
class PCSAFTDMixture : public PCSAFTMixture{
private:
    Eigen::ArrayXd mustar2, nmu;
    template<typename A> auto POW2(const A& x) const { return forceeval(x*x); }
    template<typename A> auto POW3(const A& x) const { return forceeval(POW2(x)*x); }
public:
    PCSAFTDMixture(const std::vector<SAFTCoeffs> &coeffs, const Eigen::ArrayXXd &kmat, const Eigen::ArrayXd& mustar2, const Eigen::ArrayXd& n) : PCSAFTMixture(coeffs, kmat), mustar2(mustar2), nmu(n) {
        // Check lengths match
        if (coeffs.size() != mustar2.size()){
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (coeffs.size() != n.size()){
            throw teqp::InvalidArgument("bad size of n");
        }
    }
    
    /// Eq. 8 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename VecType>
    auto get_alpha2DD(const TTYPE& T, const RhoType& rhoN_A3, const RhoType& eta, const VecType& mole_fractions) const{
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
                    auto mij = sqrt(m[i]*m[j]);
                    summer += x[i]*x[j]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*POW3(sigma[i]*sigma[j]/sigmaij)*ninj*mustar2[i]*mustar2[j]*get_JDD_2ij(eta, mij, Tstarij);
                }
            }
        }
        return forceeval(-EIGEN_PI*rhoN_A3*summer);
    }
    
    /// Eq. 9 from Gross and Vrabec
    template<typename TTYPE, typename RhoType, typename VecType>
    auto get_alpha3DD(const TTYPE& T, const RhoType& rhoN_A3, const RhoType& eta, const VecType& mole_fractions) const{
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
                        
                        auto mijk = pow(m[i]*m[j]*m[k], 1.0/3.0);
                        summer += x[i]*x[j]*x[k]*epsilon_over_k[i]/T*epsilon_over_k[j]/T*epsilon_over_k[k]/T*POW3(sigma[i]*sigma[j]*sigma[k])/(sigmaij*sigmaik*sigmajk)*ninjnk*mustar2[i]*mustar2[j]*mustar2[k]*get_JDD_3ijk(eta, mijk);
                    }
                }
            }
        }
        return forceeval(-4.0*POW2(EIGEN_PI)/3.0*POW2(rhoN_A3)*summer);
    }
    
    /***
     * \brief Get the dipolar contribution to \f$ \alpha = A/(NkT) \f$
     */
    template<typename TTYPE, typename RhoType, typename VecType>
    auto get_alphar_dipolar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        
        /// Convert from molar density to number density in molecules/Angstrom^3
        RhoType rho_A3 = rhomolar * N_A * 1e-30; //[molecules (not moles)/A^3]

        constexpr double MY_PI = EIGEN_PI;
        double pi6 = (MY_PI / 6.0);
        using TRHOType = RhoType;
        
        auto d3 = (sigma_Angstrom*(1.0 - 0.12 * exp(-3.0*epsilon_over_k/T))).pow(3);
        TRHOType xmdn = forceeval((mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()*d3.template cast<TRHOType>().array()).sum());
        RhoType eta = pi6*rho_A3*xmdn;
        auto alpha2 = get_alpha2DD(T, rho_A3, eta, mole_fractions);
        auto alpha3 = get_alpha3DD(T, rho_A3, eta, mole_fractions);
        return alpha2/(1-alpha3/alpha2);
    }
    
    /// The function to get $\alpha$ of the combined model
    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        // call base class for the primary contributions
        auto base = PCSAFTMixture::alphar(T, rhomolar, mole_fractions);
        auto dipolar = get_alphar_dipolar(T, rhomolar, mole_fractions);
        return forceeval(base + dipolar);
    }
};

inline auto PCSAFTfactory(const nlohmann::json& json) {
    std::vector<SAFTCoeffs> coeffs;
    for (auto j : json) {
        SAFTCoeffs c;
        c.name = j.at("name");
        c.m = j.at("m");
        c.sigma_Angstrom = j.at("sigma_Angstrom");
        c.epsilon_over_k = j.at("epsilon_over_k");
        c.BibTeXKey = j.at("BibTeXKey");
        coeffs.push_back(c);
    }
    return PCSAFTMixture(coeffs);
};

} /* namespace PCSAFT */
}; // namespace teqp
