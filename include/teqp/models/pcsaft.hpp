/***
 
 \brief This file contains the contributions that can be composed together to form SAFT models

*/

#pragma once

#include "nlohmann/json.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/constants.hpp"
#include "teqp/json_tools.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include <optional>

namespace teqp {
namespace PCSAFT {

//#define PCSAFTDEBUG

/// Coefficients for one fluid
struct SAFTCoeffs {
    std::string name; ///< Name of fluid
    double m = -1, ///< number of segments
        sigma_Angstrom = -1, ///< [A] segment diameter
        epsilon_over_k = -1; ///< [K] depth of pair potential divided by Boltzman constant
    std::string BibTeXKey; ///< The BibTeXKey for the reference for these coefficients
    double mustar2 = 0, ///< nondimensional, the reduced dipole moment squared
           nmu = 0, ///< number of dipolar segments
           Qstar2 = 0, ///< nondimensional, the reduced quadrupole squared
           nQ = 0; ///< number of quadrupolar segments
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
auto C1(const Eta& eta, const Mbar& mbar) {
    auto oneeta = 1.0 - eta;
    return forceeval(1.0 / (1.0
        + mbar * (8.0 * eta - 2.0 * eta * eta) / (oneeta*oneeta*oneeta*oneeta)
        + (1.0 - mbar) * (20.0 * eta - 27.0 * eta * eta + 12.0 * eta*eta*eta - 2.0 * eta*eta*eta*eta) / ((1.0 - eta) * (2.0 - eta)*(1.0 - eta) * (2.0 - eta))));
}
/// Eqn. A.31
template <typename Eta, typename Mbar>
auto C2(const Eta& eta, const Mbar& mbar) {
    return forceeval(-pow(C1(eta, mbar), 2) * (
        mbar * (-4.0 * eta * eta + 20.0 * eta + 8.0) / pow(1.0 - eta, 5)
        + (1.0 - mbar) * (2.0 * eta * eta * eta + 12.0 * eta * eta - 48.0 * eta + 40.0) / pow((1.0 - eta) * (2.0 - eta), 3)
        ));
}
/// Eqn. A.18
template<typename TYPE>
auto get_a(const TYPE& mbar) {
    static Eigen::ArrayXd a_0 = (Eigen::ArrayXd(7) << 0.9105631445, 0.6361281449, 2.6861347891, -26.547362491, 97.759208784, -159.59154087, 91.297774084).finished();
    static Eigen::ArrayXd a_1 = (Eigen::ArrayXd(7) << -0.3084016918, 0.1860531159, -2.5030047259, 21.419793629, -65.255885330, 83.318680481, -33.746922930).finished();
    static Eigen::ArrayXd a_2 = (Eigen::ArrayXd(7) << -0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131, -4.1302112531, 13.776631870, -8.6728470368).finished();
    return forceeval(a_0.cast<TYPE>().array() + ((mbar - 1.0) / mbar) * a_1.cast<TYPE>().array() + ((mbar - 1.0) / mbar * (mbar - 2.0) / mbar) * a_2.cast<TYPE>().array()).eval();
}
/// Eqn. A.19
template<typename TYPE>
auto get_b(const TYPE& mbar) {
    // See https://stackoverflow.com/a/35170514/1360263
    static Eigen::ArrayXd b_0 = (Eigen::ArrayXd(7) << 0.7240946941, 2.2382791861, -4.0025849485, -21.003576815, 26.855641363, 206.55133841, -355.60235612).finished();
    static Eigen::ArrayXd b_1 = (Eigen::ArrayXd(7) << -0.5755498075, 0.6995095521, 3.8925673390, -17.215471648, 192.67226447, -161.82646165, -165.20769346).finished();
    static Eigen::ArrayXd b_2 = (Eigen::ArrayXd(7) << 0.0976883116, -0.2557574982, -9.1558561530, 20.642075974, -38.804430052, 93.626774077, -29.666905585).finished();
    return forceeval(b_0.cast<TYPE>().array() + (mbar - 1.0) / mbar * b_1.cast<TYPE>().array() + (mbar - 1.0) / mbar * (mbar - 2.0) / mbar * b_2.cast<TYPE>().array()).eval();
}
/// Residual contribution to alphar from hard-sphere (Eqn. A.6)
template<typename VecType, typename VecType2>
auto get_alphar_hs(const VecType& zeta, const VecType2& D) {
    /*
    The limit of alphar_hs in the case of density going to zero is zero,
    but its derivatives must still match so that the automatic differentiation tooling
    will work properly, so a Taylor series around rho=0 is constructed. The first term is
    needed for calculations of virial coefficient temperature derivatives.
    The term zeta_0 in the denominator is zero, but *ratios* of zeta values are ok because
    they cancel the rho (in the limit at least) so we can write that zeta_x/zeta_y = D_x/D_x where
    D_i = sum_i x_im_id_{ii}. This allows for the substitution into the series expansion terms.
    
    <sympy>
     from sympy import *
     zeta_0, zeta_1, zeta_2, zeta_3, rho = symbols('zeta_0, zeta_1, zeta_2, zeta_3, rho')
     D_0, D_1, D_2, D_3 = symbols('D_0, D_1, D_2, D_3')
     POW2 = lambda x: x**2
     POW3 = lambda x: x**3
     alpha = 1/zeta_0*(3*zeta_1*zeta_2/(1-zeta_3) + zeta_2**3/(zeta_3*POW2(1-zeta_3)) + (POW3(zeta_2)/POW2(zeta_3)-zeta_0)*log(1-zeta_3))
     alpha = alpha.subs(zeta_0, rho*D_0).subs(zeta_1, rho*D_1).subs(zeta_2, rho*D_2).subs(zeta_3, rho*D_3)
     for Nderiv in [1, 2, 3, 4, 5]:
         display(simplify((simplify(diff(alpha, rho, Nderiv)).subs(rho,0)*rho**Nderiv/factorial(Nderiv)).subs(D_0, zeta_0/rho).subs(D_1, zeta_1/rho).subs(D_2, zeta_2/rho).subs(D_3, zeta_3/rho)))
    </sympy>
    */
    if (getbaseval(zeta[3]) == 0){
        return forceeval(
             0.0 // 0-th order term, the limit of the function at zero density is zero
             + zeta[3] + 3.0*D[1]*zeta[2]/D[0] // 1st order term f'(x=0)*x/1!
             + (zeta[3]*zeta[3] + 6.0*D[1]/D[0]*zeta[2]*zeta[3] + 3.0*zeta[2]*zeta[2]*D[2]/D[0])/2.0 // 2nd order term f''(x=0)*x^2/2!
             + D[3]/D[0]*(zeta[0]*zeta[3]*zeta[3] + 9.0*zeta[1]*zeta[2]*zeta[3] + 8.0*zeta[2]*zeta[2]*zeta[2])/3.0 // 3rd order term f'''(x=0)*x^3/3!
             + zeta[3]*D[3]/D[0]*(zeta[0]*zeta[3]*zeta[3] + 12.0*zeta[1]*zeta[2]*zeta[3] + 15.0*zeta[2]*zeta[2]*zeta[2])/4.0 // 4th order term f''''(x=0)*x^4/4!
             // ... and so on
         );
    }
    auto Upsilon = 1.0 - zeta[3];
    return forceeval(1.0 / zeta[0] * (3.0 * zeta[1] * zeta[2] / Upsilon
        + zeta[2] * zeta[2] * zeta[2] / zeta[3] / Upsilon / Upsilon
        + (zeta[2] * zeta[2] * zeta[2] / (zeta[3] * zeta[3]) - zeta[0]) * log(1.0 - zeta[3])
        ));
}

/// Term from Eqn. A.7
template<typename zVecType, typename dVecType>
auto gij_HS(const zVecType& zeta, const dVecType& d,
    std::size_t i, std::size_t j) {
    auto Upsilon = 1.0 - zeta[3];
#if defined(PCSAFTDEBUG)
    auto term1 = forceeval(1.0 / (Upsilon));
    auto term2 = forceeval(d[i] * d[j] / (d[i] + d[j]) * 3.0 * zeta[2] / pow(Upsilon, 2));
    auto term3 = forceeval(pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2.0 * zeta[2]*zeta[2] / pow(Upsilon, 3));
#endif
    return forceeval(1.0 / (Upsilon)+d[i] * d[j] / (d[i] + d[j]) * 3.0 * zeta[2] / pow(Upsilon, 2)
        + pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2.0 * zeta[2]*zeta[2] / pow(Upsilon, 3));
}
/// Eqn. A.16, Eqn. A.29
template <typename Eta, typename MbarType>
auto get_I1(const Eta& eta, const MbarType& mbar) {
    auto avec = get_a(mbar);
    Eta summer_I1 = 0.0, summer_etadI1deta = 0.0;
    for (std::size_t i = 0; i < 7; ++i) {
        auto increment = avec(i) * powi(eta, static_cast<int>(i));
        summer_I1 = summer_I1 + increment;
        summer_etadI1deta = summer_etadI1deta + increment * (i + 1.0);
    }
    return std::make_tuple(forceeval(summer_I1), forceeval(summer_etadI1deta));
}
/// Eqn. A.17, Eqn. A.30
template <typename Eta, typename MbarType>
auto get_I2(const Eta& eta, const MbarType& mbar) {
    auto bvec = get_b(mbar);
    Eta summer_I2 = 0.0 * eta, summer_etadI2deta = 0.0 * eta;
    for (std::size_t i = 0; i < 7; ++i) {
        auto increment = bvec(i) * powi(eta, static_cast<int>(i));
        summer_I2 = summer_I2 + increment;
        summer_etadI2deta = summer_etadI2deta + increment * (i + 1.0);
    }
    return std::make_tuple(forceeval(summer_I2), forceeval(summer_etadI2deta));
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

/***
 * \brief This class provides the evaluation of the hard chain contribution from classic PC-SAFT
 */
class PCSAFTHardChainContribution{
    
protected:
    const Eigen::ArrayX<double> m, ///< number of segments
        mminus1, ///< m-1
        sigma_Angstrom, ///<
        epsilon_over_k; ///< depth of pair potential divided by Boltzman constant
    const Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix

public:
    PCSAFTHardChainContribution(const Eigen::ArrayX<double> &m, const Eigen::ArrayX<double> &mminus1, const Eigen::ArrayX<double> &sigma_Angstrom, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayXXd &kmat)
    : m(m), mminus1(mminus1), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), kmat(kmat) {}
    
    PCSAFTHardChainContribution& operator=( const PCSAFTHardChainContribution& ) = delete; // non copyable
    
    template<typename TTYPE, typename RhoType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        
        Eigen::Index N = m.size();
        
        if (mole_fractions.size() != N) {
            throw std::invalid_argument("Length of mole_fractions (" + std::to_string(mole_fractions.size()) + ") is not the length of components (" + std::to_string(N) + ")");
        }
        
        using TRHOType = std::common_type_t<std::decay_t<TTYPE>, std::decay_t<RhoType>, std::decay_t<decltype(mole_fractions[0])>, std::decay_t<decltype(m[0])>>;
        
        SAFTCalc<TTYPE, TRHOType> c;
        c.m2_epsilon_sigma3_bar = static_cast<TRHOType>(0.0);
        c.m2_epsilon2_sigma3_bar = static_cast<TRHOType>(0.0);
        c.d.resize(N);
        for (auto i = 0L; i < N; ++i) {
            c.d[i] = sigma_Angstrom[i]*(1.0 - 0.12 * exp(-3.0*epsilon_over_k[i]/T)); // [A]
            for (auto j = 0; j < N; ++j) {
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
        std::vector<ta> zeta(4), D(4);
        for (std::size_t n = 0; n < 4; ++n) {
            // Eqn A.8
            auto dn = pow(c.d, static_cast<int>(n));
            TRHOType xmdn = forceeval((mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()*dn.template cast<TRHOType>().array()).sum());
            D[n] = forceeval(pi6*xmdn);
            zeta[n] = forceeval(D[n]*rho_A3);
        }
        
        /// Packing fraction is the 4-th value in zeta, at index 3
        auto eta = zeta[3];
        
        auto [I1, etadI1deta] = get_I1(eta, mbar);
        auto [I2, etadI2deta] = get_I2(eta, mbar);
        
        // Hard chain contribution from G&S
        using tt = std::common_type_t<decltype(zeta[0]), decltype(c.d[0])>;
        Eigen::ArrayX<tt> lngii_hs(mole_fractions.size());
        for (auto i = 0; i < lngii_hs.size(); ++i) {
            lngii_hs[i] = log(gij_HS(zeta, c.d, i, i));
        }
        auto alphar_hc = forceeval(mbar * get_alphar_hs(zeta, D) - sumproduct(mole_fractions, mminus1, lngii_hs)); // Eq. A.4
        
        // Dispersive contribution
        auto C1_ = C1(eta, mbar);
        auto alphar_disp = forceeval(-2 * MY_PI * rho_A3 * I1 * c.m2_epsilon_sigma3_bar - MY_PI * rho_A3 * mbar * C1_ * I2 * c.m2_epsilon2_sigma3_bar);
                                    
        if (!std::isfinite(getbaseval(alphar_hc))){
            throw teqp::InvalidValue("An invalid value was obtained for alphar_hc; please investigate");
        }
        if (!std::isfinite(getbaseval(I1))){
            throw teqp::InvalidValue("An invalid value was obtained for I1; please investigate");
        }
        if (!std::isfinite(getbaseval(I2))){
            throw teqp::InvalidValue("An invalid value was obtained for I2; please investigate");
        }
        if (!std::isfinite(getbaseval(C1_))){
            throw teqp::InvalidValue("An invalid value was obtained for C1; please investigate");
        }
        if (!std::isfinite(getbaseval(alphar_disp))){
            throw teqp::InvalidValue("An invalid value was obtained for alphar_disp; please investigate");
        }
        using eta_t = decltype(eta);
        using hc_t = decltype(alphar_hc);
        using disp_t = decltype(alphar_disp);
        struct PCSAFTHardChainContributionTerms{
            eta_t eta;
            hc_t alphar_hc;
            disp_t alphar_disp;
        };
        return PCSAFTHardChainContributionTerms{forceeval(eta), alphar_hc, alphar_disp};
    }
};

/** A class used to evaluate mixtures using PC-SAFT model

This is the classical Gross and Sadowski model from 2001: https://doi.org/10.1021/ie0003887
 
with the errors fixed as noted in a comment: https://doi.org/10.1021/acs.iecr.9b01515
*/
class PCSAFTMixture {
public:
    using PCSAFTDipolarContribution = SAFTpolar::DipolarContributionGrossVrabec;
    using PCSAFTQuadrupolarContribution = SAFTpolar::QuadrupolarContributionGross;
protected:
    Eigen::ArrayX<double> m, ///< number of segments
        mminus1, ///< m-1
        sigma_Angstrom, ///< 
        epsilon_over_k; ///< depth of pair potential divided by Boltzman constant
    std::vector<std::string> names, bibtex;
    Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix
    
    PCSAFTHardChainContribution hardchain;
    std::optional<PCSAFTDipolarContribution> dipolar; // Can be present or not
    std::optional<PCSAFTQuadrupolarContribution> quadrupolar; // Can be present or not

    void check_kmat(Eigen::Index N) {
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
    auto get_coeffs_from_names(const std::vector<std::string> &the_names){
        PCSAFTLibrary library;
        return library.get_coeffs(the_names);
    }
    auto build_hardchain(const std::vector<SAFTCoeffs> &coeffs){
        check_kmat(coeffs.size());

        m.resize(coeffs.size());
        mminus1.resize(coeffs.size());
        sigma_Angstrom.resize(coeffs.size());
        epsilon_over_k.resize(coeffs.size());
        names.resize(coeffs.size());
        bibtex.resize(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            m[i] = coeff.m;
            mminus1[i] = m[i] - 1;
            sigma_Angstrom[i] = coeff.sigma_Angstrom;
            epsilon_over_k[i] = coeff.epsilon_over_k;
            names[i] = coeff.name;
            bibtex[i] = coeff.BibTeXKey;
            i++;
        }
        return PCSAFTHardChainContribution(m, mminus1, sigma_Angstrom, epsilon_over_k, kmat);
    }
    auto extract_names(const std::vector<SAFTCoeffs> &coeffs){
        std::vector<std::string> names_;
        for (const auto& c: coeffs){
            names_.push_back(c.name);
        }
        return names_;
    }
    auto build_dipolar(const std::vector<SAFTCoeffs> &coeffs) -> std::optional<PCSAFTDipolarContribution>{
        Eigen::ArrayXd mustar2(coeffs.size()), nmu(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            mustar2[i] = coeff.mustar2;
            nmu[i] = coeff.nmu;
            i++;
        }
        if ((mustar2*nmu).cwiseAbs().sum() == 0){
            return std::nullopt; // No dipolar contribution is present
        }
        // The dispersive and hard chain initialization has already happened at this point
        return PCSAFTDipolarContribution(m, sigma_Angstrom, epsilon_over_k, mustar2, nmu);
    }
    auto build_quadrupolar(const std::vector<SAFTCoeffs> &coeffs) -> std::optional<PCSAFTQuadrupolarContribution>{
        // The dispersive and hard chain initialization has already happened at this point
        Eigen::ArrayXd Qstar2(coeffs.size()), nQ(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            Qstar2[i] = coeff.Qstar2;
            nQ[i] = coeff.nQ;
            i++;
        }
        if ((Qstar2*nQ).cwiseAbs().sum() == 0){
            return std::nullopt; // No quadrupolar contribution is present
        }
        return PCSAFTQuadrupolarContribution(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ);
    }
public:
    PCSAFTMixture(const std::vector<std::string> &names, const Eigen::ArrayXXd& kmat = {}) : PCSAFTMixture(get_coeffs_from_names(names), kmat){};
    PCSAFTMixture(const std::vector<SAFTCoeffs> &coeffs, const Eigen::ArrayXXd &kmat = {}) : names(extract_names(coeffs)), kmat(kmat), hardchain(build_hardchain(coeffs)), dipolar(build_dipolar(coeffs)), quadrupolar(build_quadrupolar(coeffs)) {};
    
//    PCSAFTMixture( const PCSAFTMixture& ) = delete; // non construction-copyable
    PCSAFTMixture& operator=( const PCSAFTMixture& ) = delete; // non copyable
    
    auto get_m() const { return m; }
    auto get_sigma_Angstrom() const { return sigma_Angstrom; }
    auto get_epsilon_over_k_K() const { return epsilon_over_k; }
    auto get_kmat() const { return kmat; }
    auto get_names() const { return names;}
    auto get_BibTeXKeys() const { return bibtex;}

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
        // First values for the chain with dispersion (always included)
        auto vals = hardchain.eval(T, rhomolar, mole_fractions);
        auto alphar = forceeval(vals.alphar_hc + vals.alphar_disp);
        
        auto rho_A3 = forceeval(rhomolar*N_A*1e-30);
        // If dipole is present, add its contribution
        if (dipolar){
            auto valsdip = dipolar.value().eval(T, rho_A3, vals.eta, mole_fractions);
            alphar += valsdip.alpha;
        }
        // If quadrupole is present, add its contribution
        if (quadrupolar){
            auto valsquad = quadrupolar.value().eval(T, rho_A3, vals.eta, mole_fractions);
            alphar += valsquad.alpha;
        }
        return forceeval(alphar);
    }
};

/// A JSON-based factory function for the PC-SAFT model
inline auto PCSAFTfactory(const nlohmann::json& spec) {
    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat") && spec.at("kmat").is_array() && spec.at("kmat").size() > 0){
        kmat = build_square_matrix(spec["kmat"]);
    }
    
    if (spec.contains("names")){
        std::vector<std::string> names = spec["names"];
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != names.size()){
            throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return PCSAFTMixture(names, kmat.value_or(Eigen::ArrayXXd{}));
    }
    else if (spec.contains("coeffs")){
        std::vector<SAFTCoeffs> coeffs;
        for (auto j : spec["coeffs"]) {
            SAFTCoeffs c;
            c.name = j.at("name");
            c.m = j.at("m");
            c.sigma_Angstrom = j.at("sigma_Angstrom");
            c.epsilon_over_k = j.at("epsilon_over_k");
            c.BibTeXKey = j.at("BibTeXKey");
            if (j.contains("(mu^*)^2") && j.contains("nmu")){
                c.mustar2 = j.at("(mu^*)^2");
                c.nmu = j.at("nmu");
            }
            if (j.contains("(Q^*)^2") && j.contains("nQ")){
                c.Qstar2 = j.at("(Q^*)^2");
                c.nQ = j.at("nQ");
            }
            coeffs.push_back(c);
        }
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != coeffs.size()){
            throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return PCSAFTMixture(coeffs, kmat.value_or(Eigen::ArrayXXd{}));
    }
    else{
        throw std::invalid_argument("you must provide names or coeffs, but not both");
    }
}

/**
 The model of Gross & Sadowski, simplified down to the case of pure fluids
 */
class PCSAFTPureGrossSadowski2001{
private:
    Eigen::Array<double, 7, 1> aim, bim;
public:
    const double pi = 3.141592653589793238462643383279502884197;
    const Eigen::Array<double, 7, 6> coeff;
    const double m, sigma_A, eps_k;
    double kappa1, kappa2;
    PCSAFTPureGrossSadowski2001(const nlohmann::json&j) : coeff((Eigen::Array<double, 7, 6>() << 0.9105631445,-0.3084016918,-0.0906148351,0.7240946941,-0.5755498075,0.0976883116  ,
                 0.6361281449,0.1860531159,0.4527842806,2.2382791861,0.6995095521,-0.2557574982    ,
                 2.6861347891,-2.5030047259,0.5962700728,-4.0025849485,3.8925673390,-9.1558561530  ,
                 -26.547362491,21.419793629,-1.7241829131,-21.003576815,-17.215471648,20.642075974 ,
                 97.759208784,-65.255885330,-4.1302112531,26.855641363,192.67226447,-38.804430052  ,
                 -159.59154087,83.318680481,13.776631870,206.55133841,-161.82646165,93.626774077   ,
                 91.297774084,-33.746922930,-8.6728470368,-355.60235612,-165.20769346,-29.666905585).finished()),
    m(j.at("m")), sigma_A(j.at("sigma / A")), eps_k(j.at("epsilon_over_k")) {
        auto mfac1 = (m-1.0)/m;
        auto mfac2 = (m-2.0)/m*mfac1;
        aim = coeff.col(0) + coeff.col(1)*mfac1 + coeff.col(2)*mfac2;
        bim = coeff.col(3) + coeff.col(4)*mfac1 + coeff.col(5)*mfac2;
        kappa1 = (2.0*pi*eps_k*pow(m, 2)*pow(sigma_A, 3));
        kappa2 = (pi*pow(eps_k, 2)*pow(m, 3)*pow(sigma_A, 3));
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& /*mole_fractions*/) const {
        
        auto rhoN_A3 = forceeval(rhomolar*N_A/1e30); // [A^3]
        
        auto d = forceeval(sigma_A*(1.0-0.12*exp(-3.0*eps_k/T)));
        Eigen::Array<decltype(d), 4, 1> dpowers; dpowers(0) = 1.0; for (auto i = 1U; i <= 3; ++i){ dpowers(i) = d*dpowers(i-1); }
        auto zeta = pi/6.0*rhoN_A3*m*dpowers;
        
        auto zeta2_to2 = zeta[2]*zeta[2];
        auto zeta2_to3 = zeta2_to2*zeta[2];
        auto zeta3_to2 = zeta[3]*zeta[3];
        auto onemineta = forceeval(1.0-zeta[3]);
        auto onemineta_to2 = onemineta*onemineta;
        auto onemineta_to3 = onemineta*onemineta_to2;
        auto onemineta_to4 = onemineta*onemineta_to3;
        
        auto alpha_hs = (3.0*zeta[1]*zeta[2]/onemineta
         + zeta2_to3/(zeta[3]*onemineta_to2)
         + (zeta2_to3/zeta3_to2-zeta[0])*log(1.0-zeta[3]))/zeta[0];
        
        auto fac_g_hs = d/2.0; // d*d/(2*d)
        auto gii = (1.0/onemineta
            + fac_g_hs*3.0*zeta[2]/onemineta_to2
            + (fac_g_hs*fac_g_hs)*2.0*zeta2_to2/onemineta_to3);
        auto alpha_hc = m*alpha_hs - (m-1)*log(gii);
        
        auto eta = zeta[3];
        auto eta2 = eta*eta;
        auto eta3 = eta2*eta;
        auto eta4 = eta2*eta2;
        auto C1 = 1.0+m*(8.0*eta-2.0*eta2)/onemineta_to4+(1.0-m)*(20.0*eta-27.0*eta2+12.0*eta3-2.0*eta4)/onemineta_to2/((2.0-eta)*(2.0-eta));
        
        Eigen::Array<decltype(eta), 7, 1> etapowers; etapowers(0) = 1.0; for (auto i = 1U; i <= 6; ++i){ etapowers(i) = eta*etapowers(i-1); }
        auto I1 = (aim.array().template cast<decltype(eta)>()*etapowers).sum();
        auto I2 = (bim.array().template cast<decltype(eta)>()*etapowers).sum();
        
        auto alpha_disp = -kappa1*rhoN_A3*I1/T - kappa2*rhoN_A3*I2/C1/(T*T);
        
        return forceeval(alpha_hc + alpha_disp);
    }
};

} /* namespace PCSAFT */
}; // namespace teqp
