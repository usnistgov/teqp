/***
 
 \brief This file contains the contributions that can be composed together to form SAFT models

*/

#pragma once

#include "nlohmann/json.hpp"
#include "teqp/types.hpp"
#include "teqp/json_tools.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/constants.hpp"
#include "teqp/math/quadrature.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include <optional>
#include <variant>

namespace teqp {
namespace SAFTVRMie {

/// Coefficients for one fluid
struct SAFTVRMieCoeffs {
    std::string name; ///< Name of fluid
    double m = -1, ///< number of segments
        sigma_m = -1, ///< [m] segment diameter
        epsilon_over_k = -1, ///< [K] depth of pair potential divided by Boltzman constant
        lambda_a = -1, ///< The attractive exponent (the 6 in LJ 12-6 potential)
        lambda_r = -1, ///< The repulsive exponent (the 12 in LJ 12-6 potential)
        mustar2 = 0, ///< nondimensional, the reduced dipole moment squared
        nmu = 0, ///< number of dipolar segments
        Qstar2 = 0, ///< nondimensional, the reduced quadrupole squared
        nQ = 0; ///< number of quadrupolar segments
    std::string BibTeXKey; ///< The BibTeXKey for the reference for these coefficients
};

enum class EpsilonijFlags { kInvalid, kLorentzBerthelot, kLafitte };

// map EpsilonijFlags values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM( EpsilonijFlags, {
    {EpsilonijFlags::kInvalid, nullptr},
    {EpsilonijFlags::kLorentzBerthelot, "Lorentz-Berthelot"},
    {EpsilonijFlags::kLafitte, "Lafitte"},
})

/// Manager class for SAFT-VR-Mie coefficients
class SAFTVRMieLibrary {
    std::map<std::string, SAFTVRMieCoeffs> coeffs;
public:
    SAFTVRMieLibrary() {
        insert_normal_fluid("Methane", 1.0000, 3.7412e-10, 153.36, 12.650, 6, "Lafitte-JCP-2001");
        insert_normal_fluid("Ethane", 1.4373, 3.7257e-10, 206.12, 12.400, 6, "Lafitte-JCP-2001");
        insert_normal_fluid("Propane", 1.6845, 3.9056e-10, 239.89, 13.006, 6, "Lafitte-JCP-2001");
    }
    void insert_normal_fluid(const std::string& name, double m, const double sigma_m, const double epsilon_over_k, const double lambda_r, const double lambda_a, const std::string& BibTeXKey) {
        SAFTVRMieCoeffs coeff;
        coeff.name = name;
        coeff.m = m;
        coeff.sigma_m = sigma_m;
        coeff.epsilon_over_k = epsilon_over_k;
        coeff.lambda_r = lambda_r;
        coeff.lambda_a = lambda_a;
        coeff.BibTeXKey = BibTeXKey;
        coeffs.insert(std::pair<std::string, SAFTVRMieCoeffs>(name, coeff));
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
        std::vector<SAFTVRMieCoeffs> c;
        for (auto n : names){
            c.push_back(get_normal_fluid(n));
        }
        return c;
    }
};

/// Things that only depend on the components themselves, but not on composition, temperature, or density
struct SAFTVRMieChainContributionTerms{
    private:
    
    /// The matrix of coefficients needed to evaluate f_k
    const Eigen::Matrix<double, 7, 7> phi{(Eigen::Matrix<double, 7, 7>() <<
        7.5365557, -359.44,  1550.9, -1.19932,  -1911.28,    9236.9,   10,
        -37.60463, 1825.6,   -5070.1, 9.063632,  21390.175, -129430,   10,
        71.745953, -3168.0,  6534.6, -17.9482,  -51320.7,    357230,   0.57,
        -46.83552, 1884.2,   -3288.7, 11.34027,  37064.54,   -315530,   -6.7,
        -2.467982, -0.82376, -2.7171, 20.52142,  1103.742,    1390.2,   -8,
        -0.50272,  -3.1935,  2.0883, -56.6377,  -3264.61,    -4518.2,   0,
        8.0956883, 3.7090,   0,       40.53683,  2556.181,    4241.6,   0 ).finished()};
    
    /// The matrix used to obtain the parameters c_1, c_2, c_3, and c_4 in Eq. A18
    const Eigen::Matrix<double, 4, 4> A{(Eigen::Matrix<double, 4, 4>() <<
         0.81096,  1.7888, -37.578,  92.284,
         1.0205,  -19.341,  151.26, -463.50,
         -1.9057, 22.845,  -228.14,  973.92,
         1.0885,  -6.1962,  106.98, -677.64).finished()};

    // Eq. A48
    auto get_lambda_k_ij(const Eigen::ArrayXd& lambda_k) const{
        Eigen::ArrayXXd mat(N,N);
        for (auto i = 0; i < lambda_k.size(); ++i){
            for (auto j = i; j < lambda_k.size(); ++j){
                mat(i,j) = 3 + sqrt((lambda_k(i)-3)*(lambda_k(j)-3));
                mat(j,i) = mat(i,j);
            }
        }
        return mat;
    }

    /// Eq. A3
    auto get_C_ij() const{
        Eigen::ArrayXXd C(N,N);
        for (auto i = 0U; i < N; ++i){
            for (auto j = i; j < N; ++j){
                C(i,j) = lambda_r_ij(i,j)/(lambda_r_ij(i,j)-lambda_a_ij(i,j))*pow(lambda_r_ij(i,j)/lambda_a_ij(i,j), lambda_a_ij(i,j)/(lambda_r_ij(i,j)-lambda_a_ij(i,j)));
                C(j,i) = C(i,j); // symmetric
            }
        }
        return C;
    }
    
    // Eq. A26
    auto get_fkij() const{
        std::vector<Eigen::ArrayXXd> f_(8); // 0-th element is present, but not initialized
        for (auto k = 1; k < 8; ++k){
            f_[k].resize(N,N);
        };
        for (auto k = 1; k < 8; ++k){
            auto phik = phi.col(k-1); // phi matrix is indexed to start at 1, but our matrix starts at 0
            Eigen::ArrayXXd num(N,N), den(N,N); num.setZero(), den.setZero();
            for (auto n = 0; n < 4; ++n){
                num += phik[n]*pow(alpha_ij, n);
            }
            for (auto n = 4; n < 7; ++n){
                den += phik[n]*pow(alpha_ij, n-3);
            }
            f_[k] = num/(1 + den);
        }
        return f_;
    }
    
    /// Eq. A45
    auto get_sigma_ij() const{
        Eigen::ArrayXXd sigma(N,N);
        for (auto i = 0U; i < N; ++i){
            for (auto j = i; j < N; ++j){
                sigma(i,j) = (sigma_A(i) + sigma_A(j))/2.0;
                sigma(j,i) = sigma(i,j); // symmetric
            }
        }
        return sigma;
    }
    
    /**
        Build the matrix of epsilon_{ij}, two methods are available based upon the value of the epsilon_ij_flag enumeration
     */
    auto get_epsilon_ij() const{
        if (epsilon_ij_flag == EpsilonijFlags::kLafitte){
            /// Eq. A55 from Lafitte
            Eigen::ArrayXXd eps_(N,N);
            for (auto i = 0U; i < N; ++i){
                for (auto j = i; j < N; ++j){
                    eps_(i,j) = (1.0-kmat(i,j))*sqrt(pow(sigma_ij(i,i),3)*pow(sigma_ij(j,j),3)*epsilon_over_k(i)*epsilon_over_k(j))/pow(sigma_ij(i,j), 3);
                    eps_(j,i) = eps_(i,j); // symmetric
                }
            }
            return eps_;
        }
        else if (epsilon_ij_flag == EpsilonijFlags::kLorentzBerthelot){
            Eigen::ArrayXXd eps_(N,N);
            for (auto i = 0U; i < N; ++i){
                for (auto j = i; j < N; ++j){
                    eps_(i,j) = (1.0-kmat(i,j))*sqrt(epsilon_over_k(i)*epsilon_over_k(j));
                    eps_(j,i) = eps_(i,j); // symmetric
                }
            }
            return eps_;
        }
        else{
            throw std::invalid_argument("epsilon_ij_flag is invalid");
        }
    }
    auto get_N(){
        auto sizes = (Eigen::ArrayX<Eigen::Index>(5) << m.size(), epsilon_over_k.size(), sigma_A.size(), lambda_a.size(), lambda_r.size()).finished();
        if (sizes.maxCoeff() != sizes.minCoeff()){
            throw teqp::InvalidArgument("sizes of pure component arrays are not all the same");
        }
        return sizes[0];
    }

    /// Eq. A18 for the attractive exponents
    auto get_cij(const Eigen::ArrayXXd& lambdaij) const{
        std::vector<Eigen::ArrayXXd> cij(4);
        for (auto n = 0; n < 4; ++n){
            cij[n].resize(N,N);
        };
        for (auto i = 0U; i < N; ++i){
            for (auto j = i; j < N; ++j){
                using CV = Eigen::Vector<double, 4>;
                const CV b{(CV() << 1, 1.0/lambdaij(i,j), 1.0/pow(lambdaij(i,j),2), 1.0/pow(lambdaij(i,j),3)).finished()};
                auto c1234 = (A*b).eval();
                cij[0](i,j) = c1234(0);
                cij[1](i,j) = c1234(1);
                cij[2](i,j) = c1234(2);
                cij[3](i,j) = c1234(3);
            }
        }
        return cij;
    }
        
    /// Eq. A18 for the attractive exponents
    auto get_canij() const{
        return get_cij(lambda_a_ij);
    }
    /// Eq. A18 for 2x the attractive exponents
    auto get_c2anij() const{
        return get_cij(2.0*lambda_a_ij);
    }
    /// Eq. A18 for the repulsive exponents
    auto get_crnij() const{
        return get_cij(lambda_r_ij);
    }
    /// Eq. A18 for the 2x the repulsive exponents
    auto get_c2rnij() const{
        return get_cij(2.0*lambda_r_ij);
    }
    /// Eq. A18 for the 2x the repulsive exponents
    auto get_carnij() const{
        return get_cij(lambda_r_ij + lambda_a_ij);
    }
    
    EpsilonijFlags get_epsilon_ij(const std::optional<nlohmann::json>& flags){
        if (flags){
            const nlohmann::json& j = flags.value();
            if (j.contains("epsilon_ij")){
                return j.at("epsilon_ij").get<EpsilonijFlags>();
            }
        }
        return EpsilonijFlags::kLafitte;
    }
    
    public:
    
    // One entry per component
    const Eigen::ArrayXd m, epsilon_over_k, sigma_A, lambda_a, lambda_r;
    const Eigen::ArrayXXd kmat;

    const Eigen::Index N;
    const EpsilonijFlags epsilon_ij_flag = EpsilonijFlags::kLafitte;

    // Calculated matrices for the ij pair
    const Eigen::ArrayXXd lambda_r_ij, lambda_a_ij, C_ij, alpha_ij, sigma_ij, epsilon_ij; // Matrices of parameters

    const std::vector<Eigen::ArrayXXd> crnij, canij, c2rnij, c2anij, carnij;
    const std::vector<Eigen::ArrayXXd> fkij; // Matrices of parameters

    SAFTVRMieChainContributionTerms(
            const Eigen::ArrayXd& m,
            const Eigen::ArrayXd& epsilon_over_k,
            const Eigen::ArrayXd& sigma_m,
            const Eigen::ArrayXd& lambda_r,
            const Eigen::ArrayXd& lambda_a,
            const Eigen::ArrayXXd& kmat,
            const std::optional<nlohmann::json> & flags = std::nullopt)
    :   m(m), epsilon_over_k(epsilon_over_k), sigma_A(sigma_m*1e10), lambda_a(lambda_a), lambda_r(lambda_r), kmat(kmat),
        N(get_N()),
        epsilon_ij_flag(get_epsilon_ij(flags)),
        lambda_r_ij(get_lambda_k_ij(lambda_r)), lambda_a_ij(get_lambda_k_ij(lambda_a)),
        C_ij(get_C_ij()), alpha_ij(C_ij*(1/(lambda_a_ij-3) - 1/(lambda_r_ij-3))),
        sigma_ij(get_sigma_ij()), epsilon_ij(get_epsilon_ij()),
        crnij(get_crnij()), canij(get_canij()),
        c2rnij(get_c2rnij()), c2anij(get_c2anij()), carnij(get_carnij()),
        fkij(get_fkij())
    {}
    
    /// Get the matrix of \f$\varepsilon_{ij}/k_B\f$ with the entries in K
    auto get_EPSKIJ_K_matrix() const { return epsilon_ij; }
    /// Get the matrix of \f$\sigma_{ij}\f$ with the entries in m
    auto get_SIGMAIJ_m_matrix() const { return sigma_ij/1e10; }
    
    /// Eq. A2 from Lafitte
    template<typename RType>
    auto get_uii_over_kB(std::size_t i, const RType& r) const {
        auto rstarinv = forceeval(sigma_A[i]/r);
        return forceeval(C_ij(i,i)*epsilon_over_k[i]*(pow(rstarinv, lambda_r[i]) - pow(rstarinv, lambda_a[i])));
    }
    
    /// Solve for the value of \f$j=\sigma/r\f$ for which the integrand in \f$d_{ii}\f$ becomes equal to 1 to numerical precision
    template <typename TType>
    auto get_j_cutoff_dii(std::size_t i, const TType &T) const {
        auto lambda_a_ = lambda_a(i), lambda_r_ = lambda_r(i);
        auto EPS = std::numeric_limits<decltype(getbaseval(T))>::epsilon();
        auto K = forceeval(log(-log(EPS)*T/(C_ij(i,i)*epsilon_ij(i,i))));
        auto j0 = forceeval(exp(K/lambda_r_)); // this was proposed by longemen3000 (Andrés Riedemann)
        auto kappa = C_ij(i,i)*epsilon_ij(i,i);
        
        // Function to return residual and its derivatives w.r.t.
        auto fgh = [&kappa, &lambda_r_, &lambda_a_, &T, &EPS](auto j){
            auto jlr = pow(j, lambda_r_), jla = pow(j, lambda_a_);
            auto u = kappa*(jlr - jla);
            auto uprime = kappa*(lambda_r_*jlr - lambda_a_*jla)/j;
            auto uprime2 = kappa*(lambda_r_*(lambda_r_-1.0)*jlr - lambda_a_*(lambda_a_-1.0)*jla)/(j*j);
            return std::make_tuple(forceeval(-u/T-log(EPS)), forceeval(-uprime/T), forceeval(-uprime2/T));
        };
        TType j = j0;
        for (auto counter = 0; counter <= 3; ++counter){
            // Halley's method steps
            auto [R, Rprime, Rprime2] = fgh(j);
            auto denominator = 2.0*Rprime*Rprime-R*Rprime2;
            if (getbaseval(denominator) < EPS){
                break;
            }
            j -= 2.0*R*Rprime/denominator;
        }
        double jbase = getbaseval(j);
        if (jbase < 1.0){
            throw teqp::IterationFailure("Cannot obtain a value of j");
        }
        return j;
    }
    
    /**
     \note Eq. A9 from Lafitte
     
     The calculation of the diameter is based upon
     \f[
     d_{ii} = \int_0^{\sigma_{ii}}(1-\exp(-\beta u_{ii}^{\rm Mie}(r)){\rm d}r
     \f]
     which is broken up into two parts:
    \f[
     d = \int_0^{r_{\rm cut}} 1 {\rm d} r + \int_{r_{\rm cut}}^{\sigma_{ii}} [1-\exp(-\beta u_{ii}^{\rm Mie}(r))] {\rm d}r
     \f]
     but the integrand is basically constant (to numerical precision) from 0 to some cutoff value of \f$r\f$, which we'll call \f$r_{\rm cut}\f$. So first we need to find the value of \f$r_{\rm cut}\f$ that makes the integrand take its constant value, which is explained well in the paper from Aasen (https://github.com/ClapeyronThermo/Clapeyron.jl/issues/152#issuecomment-1480324192).  Finding the cutoff value is obtained when
     \f[
     \exp(-\beta u_{ii}^{\rm Mie}(r)) = EPS
     \f]
     where EPS is the numerical precision of the floating point type. Taking the logs of both sides,
     \f[
     -\beta u_{ii}^{\rm Mie} = \ln(EPS)
     \f]

     To get a starting value, it is first assumed that only the repulsive contribution contributes to the potential, yielding \f$u^{\rm rep} = C\epsilon(\sigma/r)^{\lambda_r}\f$ (with \f$C\f$ the same as the full potential with attraction) which yields
     \f[
     -\beta C\epsilon(\sigma/r)^{\lambda_r} = \ln(EPS)
     \f]
     and
     \f[
     (\sigma/r)_{\rm guess} = (-\ln(EPS)/(\beta C \epsilon))^{1/\lambda_r}
     \f]

     Then we solve for the residual \f$R(r)=0\f$, where \f$R_0=\exp(-u/T)-EPS\f$.  Equivalently we can write the residual in logarithmic terms as \f$R=-u/T-\ln(EPS)\f$. This simplifies the rootfinding as you need \f$R\f$, \f$R'\f$ and \f$R''\f$ to apply Halley's method, which are themselves quite straightforward to obtain because \f$R'=-u'/T\f$, \f$R''=-u''/T\f$, where the primes are derivatives taken with respect to \f$\sigma/r\f$.
    
    */
    template <typename TType>
    TType get_dii(std::size_t i, const TType &T) const{
        std::function<TType(TType)> integrand = [this, i, &T](const TType& r){
            return forceeval(1.0-exp(-this->get_uii_over_kB(i, r)/T));
        };
        
        // Sum of the two integrals, one is constant, the other is from integration
        auto rcut = forceeval(sigma_A[i]/get_j_cutoff_dii(i, T));
        auto integral_contribution = quad<10, TType, TType>(integrand, rcut, sigma_A[i]);
        auto d = forceeval(rcut + integral_contribution);
        
        if (getbaseval(d) > sigma_A[i]){
            throw teqp::IterationFailure("Value of d is larger than sigma; this is impossible");
        }
        return d;
    }
    
    template <typename TType>
    auto get_dmat(const TType &T) const{
        Eigen::Array<TType, Eigen::Dynamic, Eigen::Dynamic> d(N,N);
        // For the pure components, by integration
        for (auto i = 0U; i < N; ++i){
            d(i,i) = get_dii(i, T);
        }
        // The cross terms, using the linear mixing rule
        for (auto i = 0U; i < N; ++i){
            for (auto j = i+1; j < N; ++j){
                d(i,j) = (d(i,i) + d(j,j))/2.0;
                d(j,i) = d(i,j);
            }
        }
        return d;
    }
    // Calculate core parameters that depend on temperature, volume, and composition
    template <typename TType, typename RhoType, typename VecType>
    auto get_core_calcs(const TType& T, const RhoType& rhomolar, const VecType& molefracs) const{
        
        if (molefracs.size() != N){
            throw teqp::InvalidArgument("Length of molefracs of "+std::to_string(molefracs.size()) + " does not match the model size of"+std::to_string(N));
        }
        
        using FracType = std::decay_t<decltype(molefracs[0])>;
        using NumType = std::common_type_t<TType, RhoType, FracType>;
        
        // Things that are easy to calculate
        // ....
        
        auto dmat = get_dmat(T); // Matrix of diameters of pure and cross terms
        auto rhoN = forceeval(rhomolar*N_A); // Number density, in molecules/m^3
        auto mbar = forceeval((molefracs*m).sum()); // Mean number of segments, dimensionless
        auto rhos = forceeval(rhoN*mbar/1e30); // Mean segment number density, in segments/A^3
        auto xs = forceeval((m*molefracs/mbar).eval()); // Segment fractions
        
        constexpr double MY_PI = static_cast<double>(EIGEN_PI);
        auto pi6 = MY_PI/6;
        
        using TRHOType = std::common_type_t<std::decay_t<TType>, std::decay_t<RhoType>, std::decay_t<decltype(molefracs[0])>, std::decay_t<decltype(m[0])>>;
        using DType = std::common_type_t<std::decay_t<TType>, std::decay_t<decltype(molefracs[0])>, std::decay_t<decltype(m[0])>>;
        Eigen::Array<TRHOType, 4, 1> zeta;
        Eigen::Array<DType, 4, 1> D;
        for (auto l = 0; l < 4; ++l){
            DType summer = 0.0;
            for (auto i = 0U; i < N; ++i){
                summer += xs(i)*powi(dmat(i,i), l);
            }
            D(l) = forceeval(pi6*summer);
            zeta(l) = forceeval(D(l)*rhos);
        }
        
        NumType summer_zeta_x = 0.0;
        TRHOType summer_zeta_x_bar = 0.0;
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                summer_zeta_x += xs(i)*xs(j)*powi(dmat(i,j), 3)*rhos;
                summer_zeta_x_bar += xs(i)*xs(j)*powi(sigma_ij(i,j), 3);
            }
        }
        
        auto zeta_x = forceeval(pi6*summer_zeta_x); // Eq. A13
        auto zeta_x_bar = forceeval(pi6*rhos*summer_zeta_x_bar); // Eq. A23
        auto zeta_x_bar5 = forceeval(POW2(zeta_x_bar)*POW3(zeta_x_bar)); // (zeta_x_bar)^5
        auto zeta_x_bar8 = forceeval(zeta_x_bar5*POW3(zeta_x_bar)); // (zeta_x_bar)^8
        
        // Coefficients in the gdHSij term, do not depend on component,
        // so calculate them here
        auto X = forceeval(POW3(1.0 - zeta_x)), X3 = X;
        auto X2 = forceeval(POW2(1.0 - zeta_x));
        auto k0 = forceeval(-log(1.0-zeta_x) + (42.0*zeta_x - 39.0*POW2(zeta_x) + 9.0*POW3(zeta_x) - 2.0*POW4(zeta_x))/(6.0*X3)); // Eq. A30
        auto k1 = forceeval((POW4(zeta_x) + 6.0*POW2(zeta_x) - 12.0*zeta_x)/(2.0*X3));
        auto k2 = forceeval(-3.0*POW2(zeta_x)/(8.0*X2));
        auto k3 = forceeval((-POW4(zeta_x) + 3.0*POW2(zeta_x) + 3.0*zeta_x)/(6.0*X3));
        
        // Pre-calculate the cubes of the diameters
        auto dmat3 = dmat.array().cube().eval();
        
        NumType a1kB = 0.0;
        NumType a2kB2 = 0.0;
        NumType a3kB3 = 0.0;
        NumType alphar_chain = 0.0;
        
        NumType K_HS = get_KHS(zeta_x);
        NumType rho_dK_HS_drho = get_rhos_dK_HS_drhos(zeta_x);
        
        for (auto i = 0U; i < N; ++i){
            for (auto j = i; j < N; ++j){
                NumType x_0_ij = sigma_ij(i,j)/dmat(i, j);
                
                // -----------------------
                // Calculations for a_1/kB
                // -----------------------
                
                auto I = [&x_0_ij](double lambda_ij){
                    return forceeval(-(pow(x_0_ij, 3-lambda_ij)-1.0)/(lambda_ij-3.0)); // Eq. A14
                };
                auto J = [&x_0_ij](double lambda_ij){
                    return forceeval(-(pow(x_0_ij, 4-lambda_ij)*(lambda_ij-3.0)-pow(x_0_ij, 3.0-lambda_ij)*(lambda_ij-4.0)-1.0)/((lambda_ij-3.0)*(lambda_ij-4.0))); // Eq. A15
                };
                auto Bhatij_a = this->get_Bhatij(zeta_x, X, I(lambda_a_ij(i,j)), J(lambda_a_ij(i,j)));
                auto Bhatij_2a = this->get_Bhatij(zeta_x, X, I(2*lambda_a_ij(i,j)), J(2*lambda_a_ij(i,j)));
                auto Bhatij_r = this->get_Bhatij(zeta_x, X, I(lambda_r_ij(i,j)), J(lambda_r_ij(i,j)));
                auto Bhatij_2r = this->get_Bhatij(zeta_x, X, I(2*lambda_r_ij(i,j)), J(2*lambda_r_ij(i,j)));
                auto Bhatij_ar = this->get_Bhatij(zeta_x, X, I(lambda_a_ij(i,j)+lambda_r_ij(i,j)), J(lambda_a_ij(i,j)+lambda_r_ij(i,j)));
                                                 
                auto one_term =  [this, &x_0_ij, &I, &J, &zeta_x, &X](double lambda_ij, const NumType& zeta_x_eff){
                    return forceeval(
                       pow(x_0_ij, lambda_ij)*(
                         this->get_Bhatij(zeta_x, X, I(lambda_ij), J(lambda_ij))
                       + this->get_a1Shatij(zeta_x_eff, lambda_ij)
                       )
                     );
                };
                NumType zeta_x_eff_r = crnij[0](i,j)*zeta_x + crnij[1](i,j)*POW2(zeta_x) + crnij[2](i,j)*POW3(zeta_x) + crnij[3](i,j)*POW4(zeta_x);
                NumType zeta_x_eff_a = canij[0](i,j)*zeta_x + canij[1](i,j)*POW2(zeta_x) + canij[2](i,j)*POW3(zeta_x) + canij[3](i,j)*POW4(zeta_x);
                NumType dzeta_x_eff_dzetax_r = crnij[0](i,j) + crnij[1](i,j)*2*zeta_x + crnij[2](i,j)*3*POW2(zeta_x) + crnij[3](i,j)*4*POW3(zeta_x);
                NumType dzeta_x_eff_dzetax_a = canij[0](i,j) + canij[1](i,j)*2*zeta_x + canij[2](i,j)*3*POW2(zeta_x) + canij[3](i,j)*4*POW3(zeta_x);

                NumType a1ij = 2.0*MY_PI*rhos*dmat3(i,j)*epsilon_ij(i,j)*C_ij(i,j)*(
                    one_term(lambda_a_ij(i,j), zeta_x_eff_a) - one_term(lambda_r_ij(i,j), zeta_x_eff_r)
                ); // divided by k_B
                                    
                NumType contribution = xs(i)*xs(j)*a1ij;
                double factor = (i == j) ? 1.0 : 2.0; // Off-diagonal terms contribute twice
                a1kB += contribution*factor;
                
                // --------------------------
                // Calculations for a_2/k_B^2
                // --------------------------
                
                NumType zeta_x_eff_2r = c2rnij[0](i,j)*zeta_x + c2rnij[1](i,j)*POW2(zeta_x) + c2rnij[2](i,j)*POW3(zeta_x) + c2rnij[3](i,j)*POW4(zeta_x);
                NumType zeta_x_eff_2a = c2anij[0](i,j)*zeta_x + c2anij[1](i,j)*POW2(zeta_x) + c2anij[2](i,j)*POW3(zeta_x) + c2anij[3](i,j)*POW4(zeta_x);
                NumType zeta_x_eff_ar = carnij[0](i,j)*zeta_x + carnij[1](i,j)*POW2(zeta_x) + carnij[2](i,j)*POW3(zeta_x) + carnij[3](i,j)*POW4(zeta_x);
                NumType dzeta_x_eff_dzetax_2r = c2rnij[0](i,j) + c2rnij[1](i,j)*2*zeta_x + c2rnij[2](i,j)*3*POW2(zeta_x) + c2rnij[3](i,j)*4*POW3(zeta_x);
                NumType dzeta_x_eff_dzetax_ar = carnij[0](i,j) + carnij[1](i,j)*2*zeta_x + carnij[2](i,j)*3*POW2(zeta_x) + carnij[3](i,j)*4*POW3(zeta_x);
                NumType dzeta_x_eff_dzetax_2a = c2anij[0](i,j) + c2anij[1](i,j)*2*zeta_x + c2anij[2](i,j)*3*POW2(zeta_x) + c2anij[3](i,j)*4*POW3(zeta_x);
                
                NumType chi_ij = fkij[1](i,j)*zeta_x_bar + fkij[2](i,j)*zeta_x_bar5 + fkij[3](i,j)*zeta_x_bar8;
                auto a2ij = 0.5*K_HS*(1.0+chi_ij)*epsilon_ij(i,j)*POW2(C_ij(i,j))*(2*MY_PI*rhos*dmat3(i,j)*epsilon_ij(i,j))*(
                     one_term(2.0*lambda_a_ij(i,j), zeta_x_eff_2a)
                  -2.0*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), zeta_x_eff_ar)
                    +one_term(2.0*lambda_r_ij(i,j), zeta_x_eff_2r)
                ); // divided by k_B^2
                                    
                NumType contributiona2 = xs(i)*xs(j)*a2ij; // Eq. A19
                a2kB2 += contributiona2*factor;
                
                // --------------------------
                // Calculations for a_3/k_B^3
                // --------------------------
                auto a3ij = -POW3(epsilon_ij(i,j))*fkij[4](i,j)*zeta_x_bar*exp(
                     fkij[5](i,j)*zeta_x_bar + fkij[6](i,j)*POW2(zeta_x_bar)
                ); // divided by k_B^3
                NumType contributiona3 = xs(i)*xs(j)*a3ij; // Eq. A25
                a3kB3 += contributiona3*factor;
                
                if (i == j){
                    // ------------------
                    // Chain contribution
                    // ------------------
                    
                    // Eq. A29
                    auto gdHSii = exp(k0 + k1*x_0_ij + k2*POW2(x_0_ij) + k3*POW3(x_0_ij));
                    
                    // The g1 terms
                    // ....
                    
                    // This is the function for the second part (not the partial) that goes in g_{1,ii},
                    // divided by 2*PI*d_ij^3*epsilon*rhos
                    auto g1_term = [&one_term](double lambda_ij, const NumType& zeta_x_eff){
                        return forceeval(lambda_ij*one_term(lambda_ij, zeta_x_eff));
                    };
                    auto g1_noderivterm = -C_ij(i,i)*(g1_term(lambda_a_ij(i,i), zeta_x_eff_a)-g1_term(lambda_r_ij(i,i), zeta_x_eff_r));
                    
                    // Bhat = B*rho*kappa; diff(Bhat, rho) = Bhat + rho*dBhat/drho; kappa = 2*pi*eps*d^3
                    // This is the function for the partial derivative rhos*(da1ij/drhos),
                    // divided by 2*PI*d_ij^3*epsilon*rhos
                    auto rhosda1iidrhos_term = [this, &x_0_ij, &I, &J, &zeta_x, &X](double lambda_ij, const NumType& zeta_x_eff, const NumType& dzetaxeff_dzetax, const NumType& Bhatij){
                        auto I_ = I(lambda_ij);
                        auto J_ = J(lambda_ij);
                        auto rhosda1Sdrhos = this->get_rhoda1Shatijdrho(zeta_x, zeta_x_eff, dzetaxeff_dzetax, lambda_ij);
                        auto rhosdBdrhos = this->get_rhodBijdrho(zeta_x, X, I_, J_, Bhatij);
                        return forceeval(pow(x_0_ij, lambda_ij)*(rhosda1Sdrhos + rhosdBdrhos));
                    };
                    // This is rhos*d(a_1ij)/drhos/(2*pi*d^3*eps*rhos)
                    auto da1iidrhos_term = C_ij(i,j)*(
                         rhosda1iidrhos_term(lambda_a_ij(i,i), zeta_x_eff_a, dzeta_x_eff_dzetax_a, Bhatij_a)
                        -rhosda1iidrhos_term(lambda_r_ij(i,i), zeta_x_eff_r, dzeta_x_eff_dzetax_r, Bhatij_r)
                    );
                    auto g1ii = 3.0*da1iidrhos_term + g1_noderivterm;
                    
                    // The g2 terms
                    // ....
                    
                    // This is the second part (not the partial deriv.) that goes in g_{2,ii},
                    // divided by 2*PI*d_ij^3*epsilon*rhos
                    auto g2_noderivterm = -POW2(C_ij(i,i))*K_HS*(
                       lambda_a_ij(i,j)*one_term(2*lambda_a_ij(i,j), zeta_x_eff_2a)
                       -(lambda_a_ij(i,j)+lambda_r_ij(i,j))*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), zeta_x_eff_ar)
                       +lambda_r_ij(i,j)*one_term(2*lambda_r_ij(i,j), zeta_x_eff_2r)
                    );
                    // This is [rhos*d(a_2ij/(1+chi_ij))/drhos]/(2*pi*d^3*eps*rhos)
                    auto da2iidrhos_term = 0.5*POW2(C_ij(i,j))*(
                        rho_dK_HS_drho*(
                            one_term(2.0*lambda_a_ij(i,j), zeta_x_eff_2a)
                            -2.0*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), zeta_x_eff_ar)
                            +one_term(2.0*lambda_r_ij(i,j), zeta_x_eff_2r))
                        +K_HS*(
                            rhosda1iidrhos_term(2.0*lambda_a_ij(i,i), zeta_x_eff_2a, dzeta_x_eff_dzetax_2a, Bhatij_2a)
                            -2.0*rhosda1iidrhos_term(lambda_a_ij(i,i)+lambda_r_ij(i,i), zeta_x_eff_ar, dzeta_x_eff_dzetax_ar, Bhatij_ar)
                            +rhosda1iidrhos_term(2.0*lambda_r_ij(i,i), zeta_x_eff_2r, dzeta_x_eff_dzetax_2r, Bhatij_2r)
                            )
                        );
                    auto g2MCAij = 3.0*da2iidrhos_term + g2_noderivterm;
                    
                    auto betaepsilon = epsilon_ij(i,i)/T; // (1/(kB*T))/epsilon
                    auto theta = exp(betaepsilon)-1.0;
                    auto phi7 = phi.col(6);
                    auto gamma_cij = phi7(0)*(-tanh(phi7(1)*(phi7(2)-alpha_ij(i,j)))+1.0)*zeta_x_bar*theta*exp(phi7(3)*zeta_x_bar + phi7(4)*POW2(zeta_x_bar)); // Eq. A37
                    auto g2ii = (1.0+gamma_cij)*g2MCAij;
                    
                    NumType giiMie = gdHSii*exp((betaepsilon*g1ii + POW2(betaepsilon)*g2ii)/gdHSii);
                    alphar_chain -= molefracs[i]*(m[i]-1.0)*log(giiMie);
                }
            }
        }
        
        auto ahs = get_a_HS(rhos, zeta, D);
        // Eq. A5 from Lafitte, multiplied by mbar
        auto alphar_mono = forceeval(mbar*(ahs + a1kB/T + a2kB2/(T*T) + a3kB3/(T*T*T)));
        
        using dmat_t = decltype(dmat);
        using rhos_t = decltype(rhos);
        using rhoN_t = decltype(rhoN);
        using mbar_t = decltype(mbar);
        using xs_t = decltype(xs);
        using zeta_t = decltype(zeta);
        using zeta_x_t = decltype(zeta_x);
        using zeta_x_bar_t = decltype(zeta_x_bar);
        using alphar_mono_t = decltype(alphar_mono);
        using a1kB_t = decltype(a1kB);
        using a2kB2_t = decltype(a2kB2);
        using a3kB3_t = decltype(a3kB3);
        using alphar_chain_t = decltype(alphar_chain);
        struct vals{
            dmat_t dmat;
            rhos_t rhos;
            rhoN_t rhoN;
            mbar_t mbar;
            xs_t xs;
            zeta_t zeta;
            zeta_x_t zeta_x;
            zeta_x_bar_t zeta_x_bar;
            alphar_mono_t alphar_mono;
            a1kB_t a1kB;
            a2kB2_t a2kB2;
            a3kB3_t a3kB3;
            alphar_chain_t alphar_chain;
        };
        return vals{dmat, rhos, rhoN, mbar, xs, zeta, zeta_x, zeta_x_bar, alphar_mono, a1kB, a2kB2, a3kB3, alphar_chain};
    }
    
    /// Eq. A21 from Lafitte
    template<typename RhoType>
    auto get_KHS(const RhoType& pf) const {
        return forceeval(pow(1.0-pf,4)/(1.0 + 4.0*pf + 4.0*pf*pf - 4.0*pf*pf*pf + pf*pf*pf*pf));
    }
    
    /**
     \f[
      \rho_s\frac{\partial K_{HS}}{\partial \rho_s} = \zeta\frac{\partial K_{HS}}{\partial \zeta}
     \f]
     */
    template<typename RhoType>
    auto get_rhos_dK_HS_drhos(const RhoType& zeta_x) const {
        auto num = -4.0*POW3(zeta_x - 1.0)*(POW2(zeta_x) - 5.0*zeta_x - 2.0);
        auto den = POW2(POW4(zeta_x) - 4.0*POW3(zeta_x) + 4.0*POW2(zeta_x) + 4.0*zeta_x + 1.0);
        return forceeval(num/den*zeta_x);
    }
    
    /// Eq. A6 from Lafitte, accounting for the case of rho_s=0, for which the limit is zero
    template<typename RhoType, typename ZetaType, typename DType>
    auto get_a_HS(const RhoType& rhos, const Eigen::Array<ZetaType, 4, 1>& zeta, const Eigen::Array<DType, 4, 1>& D) const{
        constexpr double MY_PI = static_cast<double>(EIGEN_PI);
        if (getbaseval(rhos) == 0){
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
             
             The updated approach is simpler; start off with the expression for alphar_hs, and simplify
             ratios where 0/0 for which the l'hopital limit would be ok until you remove all the terms
             in the denominator, allowing it to be evaluated. All terms have a zeta_x/zeta_0 and a few
             have zeta_3*zeta_0 in the denominator
             
            */
            auto Upsilon = 1.0 - zeta[3];
            return forceeval(
                 3.0*D[1]/D[0]*zeta[2]/Upsilon
                 + D[2]*D[2]*zeta[2]/(D[3]*D[0]*Upsilon*Upsilon)
                 - log(Upsilon)
                 + (D[2]*D[2]*D[2])/(D[3]*D[3]*D[0])*log(Upsilon)
            );
        }
        else{
            return forceeval(6.0/(MY_PI*rhos)*(3.0*zeta[1]*zeta[2]/(1.0-zeta[3]) + POW3(zeta[2])/(zeta[3]*POW2(1.0-zeta[3])) + (POW3(zeta[2])/POW2(zeta[3])-zeta[0])*log(1.0-zeta[3])));
        }
    }
    
    /**
    \note Starting from Eq. A12 from Lafitte
     
    Defining:
    \f[
    \hat B_{ij} \equiv \frac{B_{ij}}{2\pi\epsilon_{ij}d^3_{ij}\rho_s} = \frac{1-\zeta_x/2}{(1-\zeta_x)^3}I-\frac{9\zeta_x(1+\zeta_x)}{2(1-\zeta_x)^3}J
    \f]
    */
    template<typename ZetaType, typename IJ>
    auto get_Bhatij(const ZetaType& zeta_x, const ZetaType& one_minus_zeta_x3, const IJ& I, const IJ& J) const{
        return forceeval(
             (1.0-zeta_x/2.0)/one_minus_zeta_x3*I - 9.0*zeta_x*(1.0+zeta_x)/(2.0*one_minus_zeta_x3)*J
        );
    }
    
    /**
    \f[
    B = \hat B_{ij}\kappa \rho_s
    \f]
     \f[
     \left(\frac{\partial B_{ij}}{\partial \rho_s}\right)_{T,\vec{z}} = \kappa\left(\hat B + \zeta_x \frac{\partial \hat B}{\partial \zeta_x}\right)
     \f]
    and thus
     \f[
    \rho_s \left(\frac{\partial B_{ij}}{\partial \rho_s}\right)_{T,\vec{z}} = \hat B + \zeta_x \frac{\partial \hat B}{\partial \zeta_x}
     \f]
    */
    template<typename ZetaType, typename IJ>
    auto get_rhodBijdrho(const ZetaType& zeta_x, const ZetaType& /*one_minus_zeta_x3*/, const IJ& I, const IJ& J, const ZetaType& Bhatij) const{
        auto dBhatdzetax = (-3.0*I*(zeta_x - 2.0) - 27.0*J*zeta_x*(zeta_x + 1.0) + (zeta_x - 1.0)*(I + 9.0*J*zeta_x + 9.0*J*(zeta_x + 1.0)))/(2.0*POW4(1.0-zeta_x));
        return forceeval(Bhatij + dBhatdzetax*zeta_x);
    }
    
    /**
     \note Starting from Eq. A16 from Lafitte
     
     \f[
     \hat a^S_{1,ii} = \frac{a^S_{1,ii}}{2\pi\epsilon_{ij}d^3_{ij}\rho_s}
     \f]
     so
     \f[
     a^S_{1,ii} = \kappa\rho_s\hat a^S_{1,ii}
     \f]
    */
    template<typename ZetaType>
    auto get_a1Shatij(const ZetaType& zeta_x_eff, double lambda_ij) const{
        return forceeval(
            -1.0/(lambda_ij-3.0)*(1.0-zeta_x_eff/2.0)/POW3(forceeval(1.0-zeta_x_eff))
        );
    }
    
    /**
     \f[
     \left(\frac{\partial a^S_{1,ii}}{\partial \rho_s}\right)_{T,\vec{z}} = \kappa\left(\hat a^S_{1,ii} + \rho_s\frac{\partial \hat a^S_{1,ii}}{\partial \rho_s} \right)
     \f]
  
     \f[
     \left(\frac{\partial a^S_{1,ii}}{\partial \rho_s}\right)_{T,\vec{z}} = \kappa\left(\hat a^S_{1,ii} + \rho_s\frac{\partial \hat a^S_{1,ii}}{\partial \zeta_{x,eff}}\frac{\partial \zeta_{x,eff}}{\partial \zeta_x}\frac{\partial \zeta_x}{\partial \rho_s} \right)
     \f]
     
     since \f$\rho_s\frac{\partial \zeta_x}{\partial \rho_s}  = \zeta_x\f$
     */
    template<typename ZetaType>
    auto get_rhoda1Shatijdrho(const ZetaType& zeta_x, const ZetaType& zeta_x_eff, const ZetaType& dzetaxeffdzetax, double lambda_ij) const{
        auto zetaxda1Shatdzetax = ((2.0*zeta_x_eff - 5.0)*dzetaxeffdzetax)/(2.0*(lambda_ij-3)*POW4(zeta_x_eff-1.0))*zeta_x;
        return forceeval(get_a1Shatij(zeta_x_eff, lambda_ij) + zetaxda1Shatdzetax);
    }
};

/**
 \brief A class used to evaluate mixtures using the SAFT-VR-Mie model
*/
class SAFTVRMieNonpolarMixture {
private:
    
    std::vector<std::string> names, bibtex;
    const SAFTVRMieChainContributionTerms terms;

    static void check_kmat(const Eigen::ArrayXXd& kmat, Eigen::Index N) {
        if (kmat.size() == 0){
            return;
        }
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows and columns are not identical");
        }
        if (kmat.cols() != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components");
        }
    };
    static auto get_coeffs_from_names(const std::vector<std::string> &names){
        SAFTVRMieLibrary library;
        return library.get_coeffs(names);
    }
    auto get_names(const std::vector<SAFTVRMieCoeffs> &coeffs){
        std::vector<std::string> names_;
        for (auto c : coeffs){
            names_.push_back(c.name);
        }
        return names_;
    }
    auto get_bibtex(const std::vector<SAFTVRMieCoeffs> &coeffs){
        std::vector<std::string> keys_;
        for (auto c : coeffs){
            keys_.push_back(c.BibTeXKey);
        }
        return keys_;
    }
public:
    static auto build_chain(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd>& kmat, const std::optional<nlohmann::json>& flags = std::nullopt){
        if (kmat){
            check_kmat(kmat.value(), coeffs.size());
        }
        const std::size_t N = coeffs.size();
        Eigen::ArrayXd m(N), epsilon_over_k(N), sigma_m(N), lambda_r(N), lambda_a(N);
        auto i = 0;
        for (const auto &coeff : coeffs) {
            m[i] = coeff.m;
            epsilon_over_k[i] = coeff.epsilon_over_k;
            sigma_m[i] = coeff.sigma_m;
            lambda_r[i] = coeff.lambda_r;
            lambda_a[i] = coeff.lambda_a;
            i++;
        }
        if (kmat){
            return SAFTVRMieChainContributionTerms(m, epsilon_over_k, sigma_m, lambda_r, lambda_a, std::move(kmat.value()), flags);
        }
        else{
            auto mat = Eigen::ArrayXXd::Zero(N,N);
            return SAFTVRMieChainContributionTerms(m, epsilon_over_k, sigma_m, lambda_r, lambda_a, std::move(mat), flags);
        }
    }
    
public:
    SAFTVRMieNonpolarMixture(const std::vector<std::string> &names, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt, const std::optional<nlohmann::json>&flags = std::nullopt) : SAFTVRMieNonpolarMixture(get_coeffs_from_names(names), kmat, flags){};
    SAFTVRMieNonpolarMixture(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd> &kmat = std::nullopt, const std::optional<nlohmann::json>&flags = std::nullopt) : names(get_names(coeffs)), bibtex(get_bibtex(coeffs)), terms(build_chain(coeffs, kmat, flags)) {};
    SAFTVRMieNonpolarMixture(SAFTVRMieChainContributionTerms&& terms, const std::vector<SAFTVRMieCoeffs> &coeffs) : names(get_names(coeffs)), bibtex(get_bibtex(coeffs)), terms(std::move(terms)) {};
    
    SAFTVRMieNonpolarMixture& operator=( const SAFTVRMieNonpolarMixture& ) = delete; // non copyable
    
    auto chain_factory(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd>& kmat){
        return SAFTVRMieNonpolarMixture::build_chain(coeffs, kmat);
    }
    
    const auto& get_terms() const { return terms; }
    auto get_core_calcs(double T, double rhomolar, const Eigen::ArrayXd& mole_fractions) const {
        auto val = terms.get_core_calcs(T, rhomolar, mole_fractions);
        
        auto fromArrayX = [](const Eigen::ArrayXd &x){std::valarray<double>n(x.size()); for (auto i = 0U; i < n.size(); ++i){ n[i] = x[i];} return n;};
        auto fromArrayXX = [](const Eigen::ArrayXXd &x){
            std::size_t N = x.rows();
            std::vector<std::vector<double>> n; n.resize(N);
            for (auto i = 0U; i < N; ++i){
                n[i].resize(N);
                for (auto j = 0U; j < N; ++j){
                    n[i][j] = x(i,j);
                }
            }
            return n;
        };
        return nlohmann::json{
            {"dmat", fromArrayXX(val.dmat)},
            {"rhos", val.rhos},
            {"rhoN", val.rhoN},
            {"mbar", val.mbar},
            {"xs", fromArrayX(val.xs)},
            {"zeta", fromArrayX(val.zeta)},
            {"zeta_x", val.zeta_x},
            {"zeta_x_bar", val.zeta_x_bar},
            {"alphar_mono", val.alphar_mono},
            {"a1kB", val.a1kB},
            {"a2kB2", val.a2kB2},
            {"a3kB3", val.a3kB3},
            {"alphar_chain", val.alphar_chain}
        };
    }
    auto get_names() const { return names; }
    auto get_BibTeXKeys() const { return bibtex; }
    auto get_m() const { return terms.m; }
    auto get_sigma_Angstrom() const { return (terms.sigma_A).eval(); }
    auto get_sigma_m() const { return terms.sigma_A/1e10; }
    auto get_epsilon_over_k_K() const { return terms.epsilon_over_k; }
    auto get_kmat() const { return terms.kmat; }
    auto get_lambda_r() const { return terms.lambda_r; }
    auto get_lambda_a() const { return terms.lambda_a; }
    auto get_EPSKIJ_matrix() const { return terms.get_EPSKIJ_K_matrix(); }
    auto get_SIGMAIJ_matrix() const { return terms.get_SIGMAIJ_m_matrix(); }
    
    // template<typename VecType>
    // double max_rhoN(const double T, const VecType& mole_fractions) const {
    //     auto N = mole_fractions.size();
    //     Eigen::ArrayX<double> d(N);
    //     for (auto i = 0; i < N; ++i) {
    //         d[i] = sigma_Angstrom[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
    //     }
    //     return 6 * 0.74 / EIGEN_PI / (mole_fractions*m*powvec(d, 3)).sum()*1e30; // particles/m^3
    // }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        // First values for the Mie chain with dispersion (always included)
        error_if_expr(T); error_if_expr(rhomolar);
        auto vals = terms.get_core_calcs(T, rhomolar, mole_fractions);
        using type = std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])>;
        type alphar = vals.alphar_mono + vals.alphar_chain;
        
        return forceeval(alphar);
    }
};

/**
 \brief A class used to evaluate mixtures using the SAFT-VR-Mie model
*/
class SAFTVRMieMixture {
private:
    
    std::vector<std::string> names, bibtex;
    const SAFTVRMieChainContributionTerms terms;
    const std::optional<SAFTpolar::multipolar_contributions_variant> polar; // Can be present or not

    static void check_kmat(const Eigen::ArrayXXd& kmat, Eigen::Index N) {
        if (kmat.size() == 0){
            return;
        }
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows and columns are not identical");
        }
        if (kmat.cols() != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components");
        }
    };
    static auto get_coeffs_from_names(const std::vector<std::string> &names){
        SAFTVRMieLibrary library;
        return library.get_coeffs(names);
    }
    auto get_names(const std::vector<SAFTVRMieCoeffs> &coeffs){
        std::vector<std::string> names_;
        for (auto c : coeffs){
            names_.push_back(c.name);
        }
        return names_;
    }
    auto get_bibtex(const std::vector<SAFTVRMieCoeffs> &coeffs){
        std::vector<std::string> keys_;
        for (auto c : coeffs){
            keys_.push_back(c.BibTeXKey);
        }
        return keys_;
    }
public:
    static auto build_chain(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd>& kmat, const std::optional<nlohmann::json>& flags = std::nullopt){
        if (kmat){
            check_kmat(kmat.value(), coeffs.size());
        }
        const std::size_t N = coeffs.size();
        Eigen::ArrayXd m(N), epsilon_over_k(N), sigma_m(N), lambda_r(N), lambda_a(N);
        auto i = 0;
        for (const auto &coeff : coeffs) {
            m[i] = coeff.m;
            epsilon_over_k[i] = coeff.epsilon_over_k;
            sigma_m[i] = coeff.sigma_m;
            lambda_r[i] = coeff.lambda_r;
            lambda_a[i] = coeff.lambda_a;
            i++;
        }
        if (kmat){
            return SAFTVRMieChainContributionTerms(m, epsilon_over_k, sigma_m, lambda_r, lambda_a, std::move(kmat.value()), flags);
        }
        else{
            auto mat = Eigen::ArrayXXd::Zero(N,N);
            return SAFTVRMieChainContributionTerms(m, epsilon_over_k, sigma_m, lambda_r, lambda_a, std::move(mat), flags);
        }
    }
    auto build_polar(const std::vector<SAFTVRMieCoeffs> &coeffs) -> decltype(this->polar){
        Eigen::ArrayXd mustar2(coeffs.size()), nmu(coeffs.size()), Qstar2(coeffs.size()), nQ(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            mustar2[i] = coeff.mustar2;
            nmu[i] = coeff.nmu;
            Qstar2[i] = coeff.Qstar2;
            nQ[i] = coeff.nQ;
            i++;
        }
        bool has_dipolar = ((mustar2*nmu).cwiseAbs().sum() == 0);
        bool has_quadrupolar = ((Qstar2*nQ).cwiseAbs().sum() == 0);
        if (!has_dipolar && !has_quadrupolar){
            return std::nullopt; // No dipolar or quadrupolar contribution is present
        }
        else{
            // The dispersive and hard chain initialization has already happened at this point
            return saft::polar_terms::GrossVrabec::MultipolarContributionGrossVrabec(terms.m, terms.sigma_A, terms.epsilon_over_k, mustar2, nmu, Qstar2, nQ);
        }
    }
    
public:
    SAFTVRMieMixture(const std::vector<std::string> &names, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt, const std::optional<nlohmann::json>&flags = std::nullopt) : SAFTVRMieMixture(get_coeffs_from_names(names), kmat, flags){};
    SAFTVRMieMixture(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd> &kmat = std::nullopt, const std::optional<nlohmann::json>&flags = std::nullopt) : names(get_names(coeffs)), bibtex(get_bibtex(coeffs)), terms(build_chain(coeffs, kmat, flags)), polar(build_polar(coeffs)) {};
    SAFTVRMieMixture(SAFTVRMieChainContributionTerms&& terms, const std::vector<SAFTVRMieCoeffs> &coeffs, std::optional<SAFTpolar::multipolar_contributions_variant> &&polar = std::nullopt) : names(get_names(coeffs)), bibtex(get_bibtex(coeffs)), terms(std::move(terms)), polar(std::move(polar)) {};
    
    
//    PCSAFTMixture( const PCSAFTMixture& ) = delete; // non construction-copyable
    SAFTVRMieMixture& operator=( const SAFTVRMieMixture& ) = delete; // non copyable
    
    auto chain_factory(const std::vector<SAFTVRMieCoeffs> &coeffs, const std::optional<Eigen::ArrayXXd>& kmat){
        SAFTVRMieMixture::build_chain(coeffs, kmat);
    }
    
    const auto& get_polar() const { return polar; }
    
    // Checker for whether a polar term is present
    bool has_polar() const{ return polar.has_value(); }
    
    const auto& get_terms() const { return terms; }
    auto get_core_calcs(double T, double rhomolar, const Eigen::ArrayXd& mole_fractions) const {
        auto val = terms.get_core_calcs(T, rhomolar, mole_fractions);
        
        auto fromArrayX = [](const Eigen::ArrayXd &x){std::valarray<double>n(x.size()); for (auto i = 0U; i < n.size(); ++i){ n[i] = x[i];} return n;};
        auto fromArrayXX = [](const Eigen::ArrayXXd &x){
            std::size_t N = x.rows();
            std::vector<std::vector<double>> n; n.resize(N);
            for (auto i = 0U; i < N; ++i){
                n[i].resize(N);
                for (auto j = 0U; j < N; ++j){
                    n[i][j] = x(i,j);
                }
            }
            return n;
        };
        return nlohmann::json{
            {"dmat", fromArrayXX(val.dmat)},
            {"rhos", val.rhos},
            {"rhoN", val.rhoN},
            {"mbar", val.mbar},
            {"xs", fromArrayX(val.xs)},
            {"zeta", fromArrayX(val.zeta)},
            {"zeta_x", val.zeta_x},
            {"zeta_x_bar", val.zeta_x_bar},
            {"alphar_mono", val.alphar_mono},
            {"a1kB", val.a1kB},
            {"a2kB2", val.a2kB2},
            {"a3kB3", val.a3kB3},
            {"alphar_chain", val.alphar_chain}
        };
    }
    auto get_names() const { return names; }
    auto get_BibTeXKeys() const { return bibtex; }
    auto get_m() const { return terms.m; }
    auto get_sigma_Angstrom() const { return (terms.sigma_A).eval(); }
    auto get_sigma_m() const { return terms.sigma_A/1e10; }
    auto get_epsilon_over_k_K() const { return terms.epsilon_over_k; }
    auto get_kmat() const { return terms.kmat; }
    auto get_lambda_r() const { return terms.lambda_r; }
    auto get_lambda_a() const { return terms.lambda_a; }
    auto get_EPSKIJ_matrix() const { return terms.get_EPSKIJ_K_matrix(); }
    auto get_SIGMAIJ_matrix() const { return terms.get_SIGMAIJ_m_matrix(); }
    
    // template<typename VecType>
    // double max_rhoN(const double T, const VecType& mole_fractions) const {
    //     auto N = mole_fractions.size();
    //     Eigen::ArrayX<double> d(N);
    //     for (auto i = 0; i < N; ++i) {
    //         d[i] = sigma_Angstrom[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
    //     }
    //     return 6 * 0.74 / EIGEN_PI / (mole_fractions*m*powvec(d, 3)).sum()*1e30; // particles/m^3
    // }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        // First values for the Mie chain with dispersion (always included)
        error_if_expr(T); error_if_expr(rhomolar);
        auto vals = terms.get_core_calcs(T, rhomolar, mole_fractions);
        using type = std::common_type_t<TTYPE, RhoType, decltype(mole_fractions[0])>;
        type alphar = vals.alphar_mono + vals.alphar_chain;
        type packing_fraction = vals.zeta[3];
        
       if (polar){ // polar term is present
           using mas = SAFTpolar::multipolar_argument_spec;
           auto visitor = [&T, &rhomolar, &mole_fractions, &packing_fraction](const auto& contrib) -> type {
               
               constexpr mas arg_spec = std::decay_t<decltype(contrib)>::arg_spec;
               if constexpr(arg_spec == mas::TK_rhoNA3_packingfraction_molefractions){
                   RhoType rho_A3 = rhomolar*N_A*1e-30;
                   type alpha = contrib.eval(T, rho_A3, packing_fraction, mole_fractions).alpha;
                   return alpha;
               }
               else if constexpr(arg_spec == mas::TK_rhoNm3_rhostar_molefractions){
                   RhoType rhoN_m3 = rhomolar*N_A;
                   auto rhostar = contrib.get_rhostar(rhoN_m3, packing_fraction, mole_fractions);
                   return contrib.eval(T, rhoN_m3, rhostar, mole_fractions).alpha;
               }
               else{
                   throw teqp::InvalidArgument("Don't know how to handle this kind of arguments in polar term");
               }
           };
           alphar += std::visit(visitor, polar.value());
       }
        
        return forceeval(alphar);
    }
};

inline auto SAFTVRMieNonpolarfactory(const nlohmann::json & spec){

    using klass = SAFTVRMieNonpolarMixture;
    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat") && spec.at("kmat").is_array() && spec.at("kmat").size() > 0){
        kmat = build_square_matrix(spec["kmat"]);
    }
    
    if (spec.contains("names")){
        std::vector<std::string> names = spec["names"];
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != names.size()){
            throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return klass(names, kmat);
    }
    else if (spec.contains("coeffs")){
        std::vector<SAFTVRMieCoeffs> coeffs;
        for (auto j : spec["coeffs"]) {
            SAFTVRMieCoeffs c;
            c.name = j.at("name");
            c.m = j.at("m");
            c.sigma_m = (j.contains("sigma_m")) ? j.at("sigma_m").get<double>() : j.at("sigma_Angstrom").get<double>()/1e10;
            c.epsilon_over_k = j.at("epsilon_over_k");
            c.lambda_r = j.at("lambda_r");
            c.lambda_a = j.at("lambda_a");
            c.BibTeXKey = j.at("BibTeXKey");
            coeffs.push_back(c);
        }
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != coeffs.size()){
            throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " +  std::to_string(kmat.value().rows()));
        }
        return klass(klass::build_chain(coeffs, kmat), coeffs);
    }
    else{
        throw std::invalid_argument("you must provide names or coeffs, but not both");
    }
}
                                                                                                                                                         
inline auto SAFTVRMiefactory(const nlohmann::json & spec){

    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat") && spec.at("kmat").is_array() && spec.at("kmat").size() > 0){
        kmat = build_square_matrix(spec["kmat"]);
    }
    
    if (spec.contains("names")){
        std::vector<std::string> names = spec["names"];
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != names.size()){
            throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return SAFTVRMieMixture(names, kmat);
    }
    else if (spec.contains("coeffs")){
        bool something_polar = false;
        std::vector<SAFTVRMieCoeffs> coeffs;
        for (auto j : spec["coeffs"]) {
            SAFTVRMieCoeffs c;
            c.name = j.at("name");
            c.m = j.at("m");
            c.sigma_m = (j.contains("sigma_m")) ? j.at("sigma_m").get<double>() : j.at("sigma_Angstrom").get<double>()/1e10;
            c.epsilon_over_k = j.at("epsilon_over_k");
            c.lambda_r = j.at("lambda_r");
            c.lambda_a = j.at("lambda_a");
            c.BibTeXKey = j.at("BibTeXKey");
            
            // These are legacy definitions of the polar moments
            if (j.contains("(mu^*)^2") && j.contains("nmu")){
                c.mustar2 = j.at("(mu^*)^2");
                c.nmu = j.at("nmu");
                something_polar = true;
            }
            if (j.contains("(Q^*)^2") && j.contains("nQ")){
                c.Qstar2 = j.at("(Q^*)^2");
                c.nQ = j.at("nQ");
                something_polar = true;
            }
            if (j.contains("Q_Cm2") || j.contains("Q_DA") || j.contains("mu_Cm") || j.contains("mu_D")){
                something_polar = true;
            }
            coeffs.push_back(c);
        }
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != coeffs.size()){
            throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " +  std::to_string(kmat.value().rows()));
        }
        
        if (!something_polar){
            // Nonpolar, just m, epsilon, sigma and possibly a kmat matrix with kij coefficients
            return SAFTVRMieMixture(SAFTVRMieMixture::build_chain(coeffs, kmat), coeffs);
        }
        else{
            // Polar term is also provided, along with the chain terms
            std::string polar_model = "GrossVrabec"; // This is the default, as it was the first one implemented
            if (spec.contains("polar_model")){
                polar_model = spec["polar_model"];
            }
            std::optional<nlohmann::json> SAFTVRMie_flags = std::nullopt;
            if (spec.contains("SAFTVRMie_flags")){
                SAFTVRMie_flags = spec["SAFTVRMie_flags"];
            }
            std::optional<nlohmann::json> polar_flags = std::nullopt;
            if (spec.contains("polar_flags")){
                polar_flags = spec["polar_flags"];
            }
            else{
                // Set to the default value
                polar_flags = nlohmann::json{{"approach", "use_packing_fraction"}};
            }
            
            // Go back and extract the dipolar and quadrupolar terms from
            // the JSON, in base SI units
            const double D_to_Cm = 3.33564e-30; // C m/D
            const double mustar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23); // factor=1/(4*pi*epsilon_0*k_B), such that (mu^*)^2 := factor*mu[Cm]^2/((epsilon/kB)[K]*sigma[m]^3)
            const double Qstar2factor = 1.0/(4*static_cast<double>(EIGEN_PI)*8.8541878128e-12*1.380649e-23); // same as mustar2factor
            auto N = coeffs.size();
            Eigen::ArrayXd ms(N), epsks(N), sigma_ms(N), mu_Cm(N), Q_Cm2(N), nQ(N), nmu(N);
            Eigen::Index i = 0;
            for (auto j : spec["coeffs"]) {
                double m = j.at("m");
                double sigma_m = (j.contains("sigma_m")) ? j.at("sigma_m").get<double>() : j.at("sigma_Angstrom").get<double>()/1e10;
                double epsilon_over_k = j.at("epsilon_over_k");
                auto get_dipole_Cm = [&]() -> double {
                    if (j.contains("(mu^*)^2") && j.contains("nmu")){
                        // Terms defined like in Gross&Vrabec; backwards-compatibility
                        double mustar2 = j.at("(mu^*)^2");
                        return sqrt(mustar2*(m*epsilon_over_k*pow(sigma_m, 3))/mustar2factor);
                    }
                    else if (j.contains("mu_Cm")){
                        return j.at("mu_Cm");
                    }
                    else if (j.contains("mu_D")){
                        return j.at("mu_D").get<double>()*D_to_Cm;
                    }
                    else{
                        return 0.0;
                    }
                };
                auto get_quadrupole_Cm2 = [&]() -> double{
                    if (j.contains("(Q^*)^2") && j.contains("nQ")){
                        // Terms defined like in Gross&Vrabec; backwards-compatibility
                        double Qstar2 = j.at("(Q^*)^2");
                        return sqrt(Qstar2*(m*epsilon_over_k*pow(sigma_m, 5))/Qstar2factor);
                    }
                    else if (j.contains("Q_Cm2")){
                        return j.at("Q_Cm2");
                    }
                    else if (j.contains("Q_DA")){
                        return j.at("Q_DA").get<double>()*D_to_Cm/1e10;
                    }
                    else{
                        return 0.0;
                    }
                };
                ms(i) = m; sigma_ms(i) = sigma_m; epsks(i) = epsilon_over_k; mu_Cm(i) = get_dipole_Cm(); Q_Cm2(i) = get_quadrupole_Cm2();
                nmu(i) = (j.contains("nmu") ? j["nmu"].get<double>() : 0.0);
                nQ(i) = (j.contains("nQ") ? j["nQ"].get<double>() : 0.0);
                i++;
            };
            nlohmann::json SAFTVRMieFlags;
            
            auto chain = SAFTVRMieMixture::build_chain(coeffs, kmat, SAFTVRMie_flags);
            auto EPSKIJ = chain.get_EPSKIJ_K_matrix(); // In units of K
            auto SIGMAIJ = chain.get_SIGMAIJ_m_matrix(); // In units of m
            
            using namespace SAFTpolar;
            if (polar_model == "GrossVrabec"){
                auto mustar2 = (mustar2factor*mu_Cm.pow(2)/(ms*epsks*sigma_ms.pow(3))).eval();
                auto Qstar2 = (Qstar2factor*Q_Cm2.pow(2)/(ms*epsks*sigma_ms.pow(5))).eval();
                auto polar = saft::polar_terms::GrossVrabec::MultipolarContributionGrossVrabec(ms, sigma_ms*1e10, epsks, mustar2, nmu, Qstar2, nQ);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            if (polar_model == "GubbinsTwu+Luckas"){
                using MCGTL = MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>;
                auto mubar2 = (mustar2factor*mu_Cm.pow(2)/(epsks*sigma_ms.pow(3))).eval();
                auto Qbar2 = (Qstar2factor*Q_Cm2.pow(2)/(epsks*sigma_ms.pow(5))).eval();
                auto polar = MCGTL(sigma_ms, epsks, mubar2, Qbar2, multipolar_rhostar_approach::use_packing_fraction);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            if (polar_model == "GubbinsTwu+GubbinsTwu"){
                using MCGG = MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>;
                auto mubar2 = (mustar2factor*mu_Cm.pow(2)/(epsks*sigma_ms.pow(3))).eval();
                auto Qbar2 = (Qstar2factor*Q_Cm2.pow(2)/(epsks*sigma_ms.pow(5))).eval();
                auto polar = MCGG(sigma_ms, epsks, mubar2, Qbar2, multipolar_rhostar_approach::use_packing_fraction);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            if (polar_model == "GubbinsTwu+Gottschalk"){
                using MCGG = MultipolarContributionGubbinsTwu<GottschalkJIntegral, GottschalkKIntegral>;
                auto mubar2 = (mustar2factor*mu_Cm.pow(2)/(epsks*sigma_ms.pow(3))).eval();
                auto Qbar2 = (Qstar2factor*Q_Cm2.pow(2)/(epsks*sigma_ms.pow(5))).eval();
                auto polar = MCGG(sigma_ms, epsks, mubar2, Qbar2, multipolar_rhostar_approach::use_packing_fraction);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            
            if (polar_model == "GrayGubbins+GubbinsTwu"){
                using MCGG = MultipolarContributionGrayGubbins<GubbinsTwuJIntegral, GubbinsTwuKIntegral>;
                auto polar = MCGG(sigma_ms, epsks, SIGMAIJ, EPSKIJ, mu_Cm, Q_Cm2, polar_flags);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
//            if (polar_model == "GrayGubbins+Gottschalk"){
//                using MCGG = MultipolarContributionGrayGubbins<GottschalkJIntegral, GottschalkKIntegral>;
//                auto polar = MCGG(sigma_ms, epsks, mu_Cm, Q_Cm2, polar_flags);
//                return SAFTVRMieMixture(std::move(chain), std::move(polar));
//            }
            if (polar_model == "GrayGubbins+Luckas"){
                using MCGG = MultipolarContributionGrayGubbins<LuckasJIntegral, LuckasKIntegral>;
                auto polar = MCGG(sigma_ms, epsks, SIGMAIJ, EPSKIJ, mu_Cm, Q_Cm2, polar_flags);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            
            
            if (polar_model == "GubbinsTwu+Luckas+GubbinsTwuRhostar"){
                using MCGTL = MultipolarContributionGubbinsTwu<LuckasJIntegral, LuckasKIntegral>;
                auto mubar2 = (mustar2factor*mu_Cm.pow(2)/(epsks*sigma_ms.pow(3))).eval();
                auto Qbar2 = (Qstar2factor*Q_Cm2.pow(2)/(epsks*sigma_ms.pow(5))).eval();
                auto polar = MCGTL(sigma_ms, epsks, mubar2, Qbar2, multipolar_rhostar_approach::calculate_Gubbins_rhostar);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            if (polar_model == "GubbinsTwu+GubbinsTwu+GubbinsTwuRhostar"){
                using MCGG = MultipolarContributionGubbinsTwu<GubbinsTwuJIntegral, GubbinsTwuKIntegral>;
                auto mubar2 = (mustar2factor*mu_Cm.pow(2)/(epsks*sigma_ms.pow(3))).eval();
                auto Qbar2 = (Qstar2factor*Q_Cm2.pow(2)/(epsks*sigma_ms.pow(5))).eval();
                auto polar = MCGG(sigma_ms, epsks, mubar2, Qbar2, multipolar_rhostar_approach::calculate_Gubbins_rhostar);
                return SAFTVRMieMixture(std::move(chain), coeffs, std::move(polar));
            }
            throw teqp::InvalidArgument("didn't understand this polar_model:"+polar_model);
        }
    }
    else{
        throw std::invalid_argument("you must provide names or coeffs, but not both");
    }
}

} /* namespace SAFTVRMie */
}; // namespace teqp
