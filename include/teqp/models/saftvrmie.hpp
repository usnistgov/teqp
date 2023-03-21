/***
 
 \brief This file contains the contributions that can be composed together to form SAFT models

*/

#pragma once

#include "nlohmann/json.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/constants.hpp"
#include "teqp/math/quadrature.hpp"
#include <optional>

namespace teqp {
namespace SAFTVRMie {

/// Coefficients for one fluid
struct SAFTVRMieCoeffs {
    std::string name; ///< Name of fluid
    double m = -1, ///< number of segments
        sigma_Angstrom = -1, ///< [A] segment diameter
        epsilon_over_k = -1, ///< [K] depth of pair potential divided by Boltzman constant
        lambda_a = -1,
        lambda_r = -1;
    std::string BibTeXKey; ///< The BibTeXKey for the reference for these coefficients
};

/// Manager class for SAFT-VR-Mie coefficients
class SAFTVRMieLibrary {
    std::map<std::string, SAFTVRMieCoeffs> coeffs;
public:
    SAFTVRMieLibrary() {
        insert_normal_fluid("Methane", 1.0000, 3.7039, 150.03, 12, 6, "Lafitte-JCP-2001");
        insert_normal_fluid("Ethane", 1.6069, 3.5206, 191.42, 12, 6, "Lafitte-JCP-2001");
        insert_normal_fluid("Propane", 2.0020, 3.6184, 208.11, 12, 6, "Lafitte-JCP-2001");
    }
    void insert_normal_fluid(const std::string& name, double m, const double sigma_Angstrom, const double epsilon_over_k, const double lambda_a, const double lambda_r, const std::string& BibTeXKey) {
        SAFTVRMieCoeffs coeff;
        coeff.name = name;
        coeff.m = m;
        coeff.sigma_Angstrom = sigma_Angstrom;
        coeff.epsilon_over_k = epsilon_over_k;
        coeff.lambda_a = lambda_a;
        coeff.lambda_r = lambda_r;
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
    auto get_lambda_k_ij(const Eigen::ArrayXd& lambda_k){
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
    auto get_C_ij(){
        Eigen::ArrayXXd C(N,N);
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                C(i,j) = lambda_r_ij(i,j)/(lambda_r_ij(i,j)-lambda_a_ij(i,j))*pow(lambda_r_ij(i,j)/lambda_a_ij(i,j), lambda_a_ij(i,j)/(lambda_r_ij(i,j)-lambda_a_ij(i,j)));
                C(j,i) = C(i,j); // symmetric
            }
        }
        return C;
    }
    
    // Eq. A26
    auto get_fkij(){
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
    auto get_sigma_ij(){
        Eigen::ArrayXXd sigma(N,N);
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                sigma(i,j) = (sigma_m(i) + sigma_m(j))/2.0;
                sigma(j,i) = sigma(i,j); // symmetric
            }
        }
        return sigma;
    }
    /// Eq. A45
    auto get_epsilon_ij(){
        Eigen::ArrayXXd eps_(N,N);
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                eps_(i,j) = (1.0-kmat(i,j))*sqrt(pow(sigma_m(i,i),3)*pow(sigma_m(j,j),3)*epsilon_over_k(i)*epsilon_over_k(j))/pow(sigma_ij(i,j), 3);
                eps_(j,i) = eps_(i,j); // symmetric
            }
        }
        return eps_;
    }
    std::size_t get_N(){
        auto sizes = (Eigen::ArrayXd(5) << m.size(), epsilon_over_k.size(), sigma_m.size(), lambda_a.size(), lambda_r.size()).finished();
        if (sizes.mean() != sizes.minCoeff()){
            throw teqp::InvalidArgument("sizes of pure component arrays are not all the same");
        }
        return sizes[0];
    }

    /// Eq. A18 for the attractive exponents
    auto get_cij(const Eigen::ArrayXd& lambdaij){
        std::vector<Eigen::ArrayXXd> cij(4);
        for (auto n = 0; n < 4; ++n){
            cij[n].resize(N,N);
        };
        for (auto i = 0; i < N; ++i){
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
    auto get_canij(){
        return get_cij(lambda_a_ij);
    }
    /// Eq. A18 for 2x the attractive exponents
    auto get_c2anij(){
        return get_cij(2.0*lambda_a_ij);
    }
    /// Eq. A18 for the repulsive exponents
    auto get_crnij(){
        return get_cij(lambda_r_ij);
    }
    /// Eq. A18 for the 2x the repulsive exponents
    auto get_c2rnij(){
        return get_cij(2.0*lambda_r_ij);
    }
    /// Eq. A18 for the 2x the repulsive exponents
    auto get_carnij(){
        return get_cij(lambda_r_ij + lambda_a_ij);
    }
    
    template<typename X>
    auto POW2(const X& x) const{
        return forceeval(x*x);
    };
    template<typename X>
    auto POW3(const X& x) const{
        return forceeval(x*POW2(x));
    };
    template<typename X>
    auto POW4(const X& x) const{
        return forceeval(POW2(x)*POW2(x));
    };
    
    public:
    
    // One entry per component
    const Eigen::ArrayXd m, epsilon_over_k, sigma_m, lambda_a, lambda_r;
    const Eigen::ArrayXXd kmat;

    const std::size_t N;

    // Calculated matrices for the ij pair
    const Eigen::ArrayXXd lambda_r_ij, lambda_a_ij, C_ij, alpha_ij, sigma_ij, epsilon_ij; // Matrices of parameters

    const std::vector<Eigen::ArrayXXd> canij, crnij, c2anij, c2rnij, carnij;
    const std::vector<Eigen::ArrayXXd> fkij; // Matrices of parameters

    SAFTVRMieChainContributionTerms(
            const Eigen::ArrayXd& m,
            const Eigen::ArrayXd& epsilon_over_k,
            const Eigen::ArrayXd& sigma_m,
            const Eigen::ArrayXd& lambda_r,
            const Eigen::ArrayXd& lambda_a,
            const Eigen::ArrayXd& kmat)
    :   m(m), epsilon_over_k(epsilon_over_k), sigma_m(sigma_m), lambda_a(lambda_a), lambda_r(lambda_r), kmat(kmat),
        N(get_N()),
        lambda_r_ij(get_lambda_k_ij(lambda_r)), lambda_a_ij(get_lambda_k_ij(lambda_a)),
        C_ij(get_C_ij()), alpha_ij(C_ij*(1/(lambda_a_ij-3) - 1/(lambda_r_ij-3))),
        sigma_ij(get_sigma_ij()), epsilon_ij(get_epsilon_ij()),
        crnij(get_crnij()), canij(get_canij()),
        c2rnij(get_c2rnij()), c2anij(get_c2anij()), carnij(get_carnij()),
        fkij(get_fkij())
    {}
    
    template<typename R>
    auto get_uii_over_kB(std::size_t i, const R& r) const {
        auto rstarinv = sigma_m[i]/r;
        return C_ij(i,i)*epsilon_over_k[i]*(pow(rstarinv, lambda_r[i]) - pow(rstarinv, lambda_a[i]));
    }
    
    template <typename TType>
    auto get_dii(std::size_t i, const TType &T) const{
        std::function<double(double)> integrand = [this, i, &T](const double& r){
            return 1.0-exp(-this->get_uii_over_kB(i, r)/T);
        };
        return quad<30, double>(integrand, 0.0, sigma_m[i]);
    }
    
    template <typename TType>
    auto get_dmat(const TType &T) const{
        Eigen::ArrayXXd d(N,N);
        // For the pure components, by integration
        for (auto i = 0; i < N; ++i){
            d(i,i) = get_dii(i, T);
        }
        // The cross terms, using the linear mixing rule
        for (auto i = 0; i < N; ++i){
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
        
        // Things that are easy to calculate
        // ....
        
        auto dmat = get_dmat(T); // Matrix of diameters of pure and cross terms
        auto rhoN = rhomolar*N_A; // Number density, in molecules/m^3
        auto mbar = forceeval((molefracs*m).sum()); // Mean number of segments, dimensionless
        auto rhos = forceeval(rhoN*mbar); // Mean segment number density
        auto xs = forceeval(m*molefracs/mbar); // Segment fractions
        using NumType = std::common_type_t<TType, RhoType, std::decay_t<decltype(molefracs[0])>>;
        Eigen::Array<NumType, 4, 1> xi;
        for (auto l = 0; l < xi.size(); ++l){
            NumType summer = 0.0;
            for (auto i = 0; i < N; ++i){
                summer += xs(i)*pow(dmat(i,i), l);
            }
            xi(l) = static_cast<double>(EIGEN_PI)*rhos/6.0*summer;
        }
        NumType summer_xi_x = 0.0, summer_xi_x_bar = 0.0;
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                summer_xi_x += xs(i)*xs(j)*pow(dmat(i,j), 3);
                summer_xi_x_bar += xs(i)*xs(j)*pow(sigma_ij(i,j), 3);
            }
        }
        constexpr double MY_PI = static_cast<double>(EIGEN_PI);
        NumType xi_x = MY_PI*rhos/6.0*summer_xi_x; // Eq. A13
        NumType xi_x_bar = MY_PI*rhos/6.0*summer_xi_x_bar; // Eq. A23
        NumType xi_x_bar5 = POW2(xi_x_bar)*POW3(xi_x_bar); // (xi_x_bar)^5
        NumType xi_x_bar8 = xi_x_bar5*POW3(xi_x_bar); // (xi_x_bar)^8
        
        // Coefficients in the gdHSij term, do not depend on component,
        // so calculate them here
        NumType X = POW3(1.0 - xi_x), X3=X;
        NumType X2 = POW2(1.0 - xi_x);
        auto k0 = -log(1-xi_x) + (42*xi_x - 39*POW2(xi_x) + 9*POW3(xi_x) - 2*POW4(xi_x))/(6*X3); // Eq. A30
        auto k1 = (POW4(xi_x) + 6*POW2(xi_x) - 12*xi_x)/(2*X3);
        auto k2 = -3*POW2(xi_x)/(8*X2);
        auto k3 = (-POW4(xi_x) + 3*POW2(xi_x) + 3*xi_x)/(6*X3);
        
        // Pre-calculate the cubes of the diameters
        auto dmat3 = dmat.array().pow(3).eval();
        
        NumType a1kB = 0.0;
        NumType a2kB2 = 0.0;
        NumType a3kB3 = 0.0;
        NumType a_chain = 0.0;
        
        auto K_HS = get_KHS(xi_x);
        auto rho_dK_HS_drho = get_rhos_dK_HS_drhos(xi_x);
        
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                NumType x_0_ij = sigma_ij(i,j)/dmat(i, j);
                
                // -----------------------
                // Calculations for a_1/kB
                // -----------------------
                
                auto I = [&x_0_ij](double lambda_ij){
                    return forceeval(-(pow(x_0_ij, 3-lambda_ij)-1.0)/(lambda_ij-3.0)); // Eq. A14
                };
                auto J = [&x_0_ij](double lambda_ij){
                    return forceeval(-(pow(x_0_ij, 4-lambda_ij)*(lambda_ij-3)-pow(x_0_ij, 3-lambda_ij)*(lambda_ij-4)-1)/((lambda_ij-3)*(lambda_ij-4))); // Eq. A15
                };
                auto Bhatij_a = this->get_Bhatij(xi_x, X, I(lambda_a_ij(i,j)), J(lambda_a_ij(i,j)));
                auto Bhatij_2a = this->get_Bhatij(xi_x, X, I(2*lambda_a_ij(i,j)), J(2*lambda_a_ij(i,j)));
                auto Bhatij_r = this->get_Bhatij(xi_x, X, I(lambda_r_ij(i,j)), J(lambda_r_ij(i,j)));
                auto Bhatij_2r = this->get_Bhatij(xi_x, X, I(2*lambda_r_ij(i,j)), J(2*lambda_r_ij(i,j)));
                auto Bhatij_ar = this->get_Bhatij(xi_x, X, I(lambda_a_ij(i,j)+lambda_r_ij(i,j)), J(lambda_a_ij(i,j)+lambda_r_ij(i,j)));
                                                 
                auto one_term =  [this, &x_0_ij, &I, &J, &xi_x, &X](double lambda_ij, const NumType& xi_x_eff){
                    return pow(x_0_ij, lambda_ij)*(this->get_Bhatij(xi_x, X, I(lambda_ij), J(lambda_ij))
                                                   + this->get_a1Shatij(xi_x_eff, lambda_ij));
                };
                NumType xi_x_eff_r = crnij[0](i,j)*xi_x + crnij[1](i,j)*POW2(xi_x) + crnij[2](i,j)*POW3(xi_x) + crnij[3](i,j)*POW4(xi_x);
                NumType xi_x_eff_a = canij[0](i,j)*xi_x + canij[1](i,j)*POW2(xi_x) + canij[2](i,j)*POW3(xi_x) + canij[3](i,j)*POW4(xi_x);
                NumType dxi_x_eff_dxix_r = crnij[0](i,j) + crnij[1](i,j)*2*xi_x + crnij[2](i,j)*3*POW2(xi_x) + crnij[3](i,j)*4*POW3(xi_x);
                NumType dxi_x_eff_dxix_a = canij[0](i,j) + canij[1](i,j)*2*xi_x + canij[2](i,j)*3*POW2(xi_x) + canij[3](i,j)*4*POW3(xi_x);

                NumType a1ij = forceeval(2*MY_PI*rhos*dmat3(i,j)*epsilon_ij(i,j)*C_ij(i,j)*(
                    one_term(lambda_a_ij(i,j), xi_x_eff_a) - one_term(lambda_r_ij(i,j), xi_x_eff_r)
                )); // divided by k_B
                                    
                NumType contribution = xs(i)*xs(j)*a1ij;
                double factor = (i == j) ? 1 : 2; // Off-diagonal terms contribute twice
                a1kB += contribution*factor;
                
                // --------------------------
                // Calculations for a_2/k_B^2
                // --------------------------
                
                NumType xi_x_eff_2r = c2rnij[0](i,j)*xi_x + c2rnij[1](i,j)*POW2(xi_x) + c2rnij[2](i,j)*POW3(xi_x) + c2rnij[3](i,j)*POW4(xi_x);
                NumType xi_x_eff_2a = c2anij[0](i,j)*xi_x + c2anij[1](i,j)*POW2(xi_x) + c2anij[2](i,j)*POW3(xi_x) + c2anij[3](i,j)*POW4(xi_x);
                NumType xi_x_eff_ar = carnij[0](i,j)*xi_x + carnij[1](i,j)*POW2(xi_x) + carnij[2](i,j)*POW3(xi_x) + carnij[3](i,j)*POW4(xi_x);
                NumType dxi_x_eff_dxix_2r = c2rnij[0](i,j) + c2rnij[1](i,j)*2*xi_x + c2rnij[2](i,j)*3*POW2(xi_x) + c2rnij[3](i,j)*4*POW3(xi_x);
                NumType dxi_x_eff_dxix_ar = carnij[0](i,j) + carnij[1](i,j)*2*xi_x + carnij[2](i,j)*3*POW2(xi_x) + carnij[3](i,j)*4*POW3(xi_x);
                NumType dxi_x_eff_dxix_2a = c2anij[0](i,j) + c2anij[1](i,j)*2*xi_x + c2anij[2](i,j)*3*POW2(xi_x) + c2anij[3](i,j)*4*POW3(xi_x);
                
                double chi_ij = fkij[1](i,j)*xi_x_bar + fkij[2](i,j)*xi_x_bar5 + fkij[3](i,j)*xi_x_bar8;
                auto a2ij = 0.5*K_HS*(1.0+chi_ij)*epsilon_ij(i,j)*POW2(C_ij(i,j))*(2*MY_PI*rhos*dmat3(i,j)*epsilon_ij(i,j))*(
                     one_term(2*lambda_a_ij(i,j), xi_x_eff_2a)
                  -2*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), xi_x_eff_ar)
                    +one_term(2*lambda_r_ij(i,j), xi_x_eff_2r)
                ); // divided by k_B^2
                                    
                NumType contributiona2 = xs(i)*xs(j)*a2ij; // Eq. A19
                a2kB2 += contributiona2*factor;
                
                // --------------------------
                // Calculations for a_3/k_B^3
                // --------------------------
                auto a3ij = -POW3(epsilon_ij(i,j))*fkij[4](i,j)*xi_x_bar*exp(
                     fkij[5](i,j)*xi_x_bar + fkij[6](i,j)*POW2(xi_x_bar)
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
                    auto g1_term = [&one_term](double lambda_ij, const NumType& xi_x_eff){
                        return lambda_ij*one_term(lambda_ij, xi_x_eff);
                    };
                    auto g1_noderivterm = -C_ij(i,i)*(g1_term(lambda_a_ij(i,i), xi_x_eff_a)-g1_term(lambda_r_ij(i,i), xi_x_eff_r));
                    
                    // Bhat = B*rho*kappa; diff(Bhat, rho) = Bhat + rho*dBhat/drho; kappa = 2*pi*eps*d^3
                    // This is the function for the partial derivative rhos*(da1ij/drhos),
                    // divided by 2*PI*d_ij^3*epsilon*rhos
                    auto rhosda1iidrhos_term = [this, &x_0_ij, &I, &J, &xi_x, &X](double lambda_ij, const NumType& xi_x_eff, const NumType& dxixeff_dxix, const NumType& Bhatij){
                        auto I_ = I(lambda_ij);
                        auto J_ = J(lambda_ij);
                        auto rhosda1Sdrhos = this->get_rhoda1Shatijdrho(xi_x, xi_x_eff, dxixeff_dxix, lambda_ij);
                        auto rhosdBdrhos = this->get_rhodBijdrho(xi_x, X, I_, J_, Bhatij);
                        return pow(x_0_ij, lambda_ij)*(rhosda1Sdrhos + rhosdBdrhos);
                    };
                    // This is rhos*d(a_1ij)/drhos/(2*pi*d^3*eps*rhos)
                    auto da1iidrhos_term = C_ij(i,j)*(
                         rhosda1iidrhos_term(lambda_a_ij(i,i), xi_x_eff_a, dxi_x_eff_dxix_a, Bhatij_a)
                        -rhosda1iidrhos_term(lambda_r_ij(i,i), xi_x_eff_r, dxi_x_eff_dxix_r, Bhatij_r)
                    );
                    auto g1ii = 3.0*da1iidrhos_term + g1_noderivterm;
                    
                    // The g2 terms
                    // ....
                    
                    // This is the second part (not the partial deriv.) that goes in g_{2,ii},
                    // divided by 2*PI*d_ij^3*epsilon*rhos
                    auto g2_noderivterm = -POW2(C_ij(i,i))*K_HS*(
                       lambda_a_ij(i,j)*one_term(2*lambda_a_ij(i,j), xi_x_eff_2a)
                       -(lambda_a_ij(i,j)+lambda_r_ij(i,j))*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), xi_x_eff_ar)
                       +lambda_r_ij(i,j)*one_term(2*lambda_r_ij(i,j), xi_x_eff_2r)
                    );
                    // This is [rhos*d(a_2ij/(1+chi_ij))/drhos]/(2*pi*d^3*eps*rhos)
                    auto da2iidrhos_term = 0.5*POW2(C_ij(i,j))*(
                        rho_dK_HS_drho*(
                            one_term(2*lambda_a_ij(i,j), xi_x_eff_2a)
                            -2*one_term(lambda_a_ij(i,j)+lambda_r_ij(i,j), xi_x_eff_ar)
                            +one_term(2*lambda_r_ij(i,j), xi_x_eff_2r))
                        +K_HS*(
                            rhosda1iidrhos_term(2*lambda_a_ij(i,i), xi_x_eff_2a, dxi_x_eff_dxix_2a, Bhatij_2a)
                            -2*rhosda1iidrhos_term(lambda_a_ij(i,i)+lambda_r_ij(i,i), xi_x_eff_ar, dxi_x_eff_dxix_ar, Bhatij_ar)
                            +rhosda1iidrhos_term(2*lambda_r_ij(i,i), xi_x_eff_2r, dxi_x_eff_dxix_2r, Bhatij_2r)
                            )
                        );
                    auto g2MCAij = 3.0*da2iidrhos_term + g2_noderivterm;
                    
                    auto betaepsilon = epsilon_ij(i,i)/T; // (1/(kB*T))/epsilon
                    auto theta = exp(betaepsilon)-1.0;
                    auto phi7 = phi.col(6);
                    auto gamma_cij = phi7(0)*(-tanh(phi7(1)*(phi7(2)-alpha_ij(i,j)))+1)*xi_x_bar*theta*exp(phi7(3)*xi_x_bar + phi7(4)*POW2(xi_x_bar)); // Eq. A37
                    auto g2ii = (1+gamma_cij)*g2MCAij;
                    
                    NumType giiMie = gdHSii*exp((betaepsilon*g1ii + POW2(betaepsilon)*g2ii)/gdHSii);
                    a_chain -= molefracs[i]*(m[i]-1)*log(giiMie);
                }
            }
        }
        
        auto ahs = a_HS(rhos, xi);
        auto a_mono = mbar*(ahs + a1kB/T + a2kB2/(T*T) + a3kB3/(T*T*T));
        
        struct vals{
            decltype(dmat) dmat;
            decltype(rhos) rhos;
            decltype(mbar) mbar;
            decltype(xs) xs;
            decltype(xi) xi;
            decltype(xi_x) xi_x;
            decltype(xi_x_bar) xi_x_bar;
            decltype(a_mono) a_mono;
            decltype(a1kB) a1kB;
            decltype(a2kB2) a2kB2;
            decltype(a3kB3) a3kB3;
            decltype(a_chain) a_chain;
        };
        return vals{dmat, rhos, mbar, xs, xi, xi_x, xi_x_bar, a_mono, a1kB, a2kB2, a3kB3, a_chain};
    }
    
    template<typename RhoType>
    auto get_KHS(const RhoType& pf) const {
        return pow(1-pf,4)/(1 + 4*pf + 4*pf*pf - 4*pf*pf*pf + pf*pf*pf*pf);
    }
    
    /**
     \f[
      rhos\frac{\partial K_{HS}}{\partial \rho_s} = \xi\frac{\partial K_{HS}}{\partial \xi}
     \f]
     */
    template<typename RhoType>
    auto get_rhos_dK_HS_drhos(const RhoType& xi_x) const {
        auto num = -4*POW3(xi_x - 1)*(POW2(xi_x) - 5*xi_x - 2);
        auto den = POW2(POW4(xi_x) - 4*POW3(xi_x) + 4*POW2(xi_x) + 4*xi_x + 1);
        return num/den*xi_x;
    }
    
    template<typename RhoType, typename XiType>
    auto a_HS(const RhoType& rhos, const Eigen::Array<XiType, 4, 1>& xi) const{
        constexpr double MY_PI = static_cast<double>(EIGEN_PI);
        return 6/(MY_PI*rhos)*(3*xi[1]*xi[2]/(1-xi[3]) + POW3(xi[2])/(xi[3]*POW2(1-xi[3])) + (POW3(xi[2])/POW2(xi[3])-xi[0])*log(1-xi[3]));
    }
    
    /***
    Defining:
    \f[
    \hat B_{ij} \equiv \frac{B_{ij}}{2\pi\epsilon_{ij}d^3_{ij}\rho_s} = \frac{1-\xi_x/2}{(1-xi_x)^3}I-\frac{9\xi_x(1+\xi_x)}{2(1-\xi_x)^3}J
    \f]
    \note Eq. A12
    */
    template<typename XiType>
    auto get_Bhatij(const XiType& xi_x, const XiType& one_minus_xi_x3, double I, double J) const{
        return (1-xi_x/2.0)/one_minus_xi_x3*I - 9.0*xi_x*(1.0+xi_x)/(2*one_minus_xi_x3)*J;
    }
    
    /***
    \f[
    B = \hat B_{ij}\kappa \rho_s
    \f]
     \f[
     \frac{\partial B_{ij}}{\partial \rho_s}\right|_{T,\vec{z}} = \kappa\left(\hat B + \xi_x \frac{\partial \hat B}{\partial \xi_x}\right)
     \f]
    and thus
     \f[
    \rho_s \frac{\partial B_{ij}}{\partial \rho_s}\right|_{T,\vec{z}} = \hat B + \xi_x \frac{\partial \hat B}{\partial \xi_x}
     \f]
    */
    template<typename XiType>
    auto get_rhodBijdrho(const XiType& xi_x, const XiType& one_minus_xi_x3, double I, double J, const XiType& Bhatij) const{
        auto dBhatdxix = (-3*I*(xi_x - 2) - 27*J*xi_x*(xi_x + 1) + (xi_x - 1)*(I + 9*J*xi_x + 9*J*(xi_x + 1)))/(2*POW4(1-xi_x));
        return forceeval(Bhatij + dBhatdxix*xi_x);
    }
    
    template<typename XiType>
    auto get_a1Shatij(const XiType& xi_x_eff, double lambda_ij) const{
        return -1.0/(lambda_ij-3.0)*(1.0-xi_x_eff/2.0)/POW3(forceeval(1.0-xi_x_eff));
    }
    
    template<typename XiType>
    auto get_rhoda1Shatijdrho(const XiType& xi_x, const XiType& xi_x_eff, const XiType& dxixeffdxix, double lambda_ij) const{
        auto one_minus_xi_x_eff3 = POW3(1.0-xi_x_eff);
        auto da1Shatdxix = ((2.0*xi_x_eff - 5.0)*dxixeffdxix)/(2.0*(lambda_ij-3)*POW4(xi_x_eff-1.0))*xi_x;
        return forceeval(get_a1Shatij(xi_x_eff, lambda_ij) + da1Shatdxix);
    }
};

///***
// * \brief This class provides the evaluation of the chain contribution from SAFT-VR-Mie
// */
//class SAFTVRMieChainContribution{
//    
//private:
//    const SAFTVRMieChainContributionTerms terms;
//    const Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix
//public:
//    SAFTVRMieChainContribution(const SAFTVRMieChainContributionTerms &terms, const Eigen::ArrayXXd &kmat)
//    : terms(terms), kmat(kmat) {}
//    
//    SAFTVRMieChainContribution& operator=( const SAFTVRMieChainContribution& ) = delete; // non copyable
//    
//    template<typename TTYPE, typename RhoType, typename VecType>
//    auto eval(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
//        
////        std::size_t N = m.size();
////
////        if (mole_fractions.size() != N) {
////            throw std::invalid_argument("Length of mole_fractions (" + std::to_string(mole_fractions.size()) + ") is not the length of components (" + std::to_string(N) + ")");
////        }
////
////        using TRHOType = std::common_type_t<std::decay_t<TTYPE>, std::decay_t<RhoType>, std::decay_t<decltype(mole_fractions[0])>, std::decay_t<decltype(m[0])>>;
////
////        SAFTCalc<TTYPE, TRHOType> c;
////        c.m2_epsilon_sigma3_bar = static_cast<TRHOType>(0.0);
////        c.m2_epsilon2_sigma3_bar = static_cast<TRHOType>(0.0);
////        c.d.resize(N);
////        for (std::size_t i = 0; i < N; ++i) {
////            c.d[i] = sigma_Angstrom[i]*(1.0 - 0.12 * exp(-3.0*epsilon_over_k[i]/T)); // [A]
////            for (std::size_t j = 0; j < N; ++j) {
////                // Eq. A.5
////                auto sigma_ij = 0.5 * sigma_Angstrom[i] + 0.5 * sigma_Angstrom[j];
////                auto eij_over_k = sqrt(epsilon_over_k[i] * epsilon_over_k[j]) * (1.0 - kmat(i,j));
////                c.m2_epsilon_sigma3_bar = c.m2_epsilon_sigma3_bar + mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * eij_over_k / T * pow(sigma_ij, 3);
////                c.m2_epsilon2_sigma3_bar = c.m2_epsilon2_sigma3_bar + mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * pow(eij_over_k / T, 2) * pow(sigma_ij, 3);
////            }
////        }
////        auto mbar = (mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()).sum();
////
////        /// Convert from molar density to number density in molecules/Angstrom^3
////        RhoType rho_A3 = rhomolar * N_A * 1e-30; //[molecules (not moles)/A^3]
////
////        constexpr double MY_PI = EIGEN_PI;
////        double pi6 = (MY_PI / 6.0);
////
////        /// Evaluate the components of zeta
////        using ta = std::common_type_t<decltype(pi6), decltype(m[0]), decltype(c.d[0]), decltype(rho_A3)>;
////        std::vector<ta> zeta(4);
////        for (std::size_t n = 0; n < 4; ++n) {
////            // Eqn A.8
////            auto dn = c.d.pow(n).eval();
////            TRHOType xmdn = forceeval((mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()*dn.template cast<TRHOType>().array()).sum());
////            zeta[n] = forceeval(pi6*rho_A3*xmdn);
////        }
////
////        /// Packing fraction is the 4-th value in zeta, at index 3
////        auto eta = zeta[3];
////
////        auto [I1, etadI1deta] = get_I1(eta, mbar);
////        auto [I2, etadI2deta] = get_I2(eta, mbar);
////
////        // Hard chain contribution from G&S
////        using tt = std::common_type_t<decltype(zeta[0]), decltype(c.d[0])>;
////        Eigen::ArrayX<tt> lngii_hs(mole_fractions.size());
////        for (auto i = 0; i < lngii_hs.size(); ++i) {
////            lngii_hs[i] = log(gij_HS(zeta, c.d, i, i));
////        }
////        auto alphar_hc = forceeval(0.0);
//        
//        auto eta = 0.0;
//        auto alphar_mono = forceeval(0.0);
//        
//        // Dispersive contribution
//        auto alphar_chain = forceeval(0.0);
//                                     
////        using eta_t = decltype(eta);
//        using mono_t = decltype(alphar_mono);
//        using chain_t = decltype(alphar_chain);
//        struct SAFTVChainContributionTerms{
////            eta_t eta;
//            mono_t alphar_mono;
//            chain_t alphar_chain;
//        };
//        return SAFTVChainContributionTerms{forceeval(alphar_mono), forceeval(alphar_chain)};
//    }
//};
//
///** A class used to evaluate mixtures using PC-SAFT model
//
//This is the classical Gross and Sadowski model from 2001: https://doi.org/10.1021/ie0003887
//*/
//class SAFTVRMie {
//private:
//    
//    std::vector<std::string> names;
//    Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix
//    
//    SAFTVRMieChainContribution hardchain;
//
//    void check_kmat(std::size_t N) {
//        if (kmat.cols() != kmat.rows()) {
//            throw teqp::InvalidArgument("kmat rows and columns are not identical");
//        }
//        if (kmat.cols() == 0) {
//            kmat.resize(N, N); kmat.setZero();
//        }
//        else if (kmat.cols() != N) {
//            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components");
//        }
//    };
//    auto get_coeffs_from_names(const std::vector<std::string> &names){
//        SAFTVRMieLibrary library;
//        return library.get_coeffs(names);
//    }
//    auto build_chain(const std::vector<SAFTVRMieCoeffs> &coeffs){
//        check_kmat(coeffs.size());
//        SAFTVRMieChainContributionTerms terms;
//        auto i = 0;
//        for (const auto &coeff : coeffs) {
//            terms.m[i] = coeff.m;
//            terms.mminus1[i] = m[i] - 1;
//            sigma_Angstrom[i] = coeff.sigma_Angstrom;
//            epsilon_over_k[i] = coeff.epsilon_over_k;
//            names[i] = coeff.name;
//            i++;
//        }
//        return SAFTVRMieChainContribution(std::move(terms), std::move(kmat));
//    }
//public:
//    // SAFTVRMieMixture(const std::vector<std::string> &names, const Eigen::ArrayXXd& kmat = {}) : PCSAFTMixture(get_coeffs_from_names(names), kmat){};
//    SAFTVRMieMixture(const std::vector<SAFTVRMieCoeffs> &coeffs, const Eigen::ArrayXXd &kmat = {}) : kmat(kmat), hardchain(build_chain(coeffs)) {};
//    
////    PCSAFTMixture( const PCSAFTMixture& ) = delete; // non construction-copyable
//    SAFTVRMieMixture& operator=( const SAFTVRMieMixture& ) = delete; // non copyable
//    
//    auto get_m() const { return m; }
//    auto get_sigma_Angstrom() const { return sigma_Angstrom; }
//    auto get_epsilon_over_k_K() const { return epsilon_over_k; }
//    auto get_kmat() const { return kmat; }
//    auto get_lambda_a() const { return lambda_a; }
//    auto get_lambda_r() const { return lambda_r; }
//    
//    // template<typename VecType>
//    // double max_rhoN(const double T, const VecType& mole_fractions) const {
//    //     auto N = mole_fractions.size();
//    //     Eigen::ArrayX<double> d(N);
//    //     for (auto i = 0; i < N; ++i) {
//    //         d[i] = sigma_Angstrom[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
//    //     }
//    //     return 6 * 0.74 / EIGEN_PI / (mole_fractions*m*powvec(d, 3)).sum()*1e30; // particles/m^3
//    // }
//    
//    template<class VecType>
//    auto R(const VecType& molefrac) const {
//        return get_R_gas<decltype(molefrac[0])>();
//    }
//
//    template<typename TTYPE, typename RhoType, typename VecType>
//    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
//        // First values for the chain with dispersion (always included)
//        auto vals = hardchain.eval(T, rhomolar, mole_fractions);
//        auto alphar = forceeval(vals.alphar_hc + vals.alphar_disp);
//        return forceeval(alphar);
//    }
//};

} /* namespace SAFTVRMie */
}; // namespace teqp
