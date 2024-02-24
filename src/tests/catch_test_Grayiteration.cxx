#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include "nlohmann/json.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include "teqp/derivs.hpp"
#include "teqp/constants.hpp"
#include "teqp/cpp/teqpcpp.hpp"

using namespace teqp;
using namespace teqp::SAFTpolar;

/**
 \tparam JIntegral A type that can be indexed with a single integer n to give the J^{(n)} integral
 \tparam KIntegral A type that can be indexed with a two integers a and b to give the K(a,b) integral
 
 The flexibility was added to include J and K integrals from either Luckas et al. or Gubbins and Twu (or any others following the interface)
 */
template<class JIntegral, class KIntegral>
class GrayGubbins {
    
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNm3_rhostar_molefractions;
private:
    const Eigen::ArrayXd sigma_m, epsilon_over_k;
    Eigen::MatrixXd SIGMAIJ, EPSKIJ;
    const Eigen::ArrayXd mu, Q, mu2, Q2, Q3;
    const bool has_a_polar;
    const Eigen::ArrayXd sigma_m3, sigma_m5;
    
    const JIntegral J6{6};
    const KIntegral K222_333{222, 333};
    const Eigen::ArrayXd alpha_symm, alpha_asymm;
    
    const double PI_ = static_cast<double>(EIGEN_PI);
    
public:
    GrayGubbins(const Eigen::ArrayX<double> &sigma_m, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::MatrixXd& SIGMAIJ, const Eigen::MatrixXd& EPSKIJ, const Eigen::ArrayX<double> &mu, const Eigen::ArrayX<double> &Q, const Eigen::ArrayX<double>& alpha_symm, const Eigen::ArrayX<double>& alpha_asymm, const std::optional<nlohmann::json>& flags) : sigma_m(sigma_m), epsilon_over_k(epsilon_over_k), SIGMAIJ(SIGMAIJ), EPSKIJ(EPSKIJ), mu(mu), Q(Q), mu2(mu.pow(2)), Q2(Q.pow(2)), Q3(Q.pow(3)), has_a_polar(Q.cwiseAbs().sum() > 0 || mu.cwiseAbs().sum() > 0), sigma_m3(sigma_m.pow(3)), sigma_m5(sigma_m.pow(5)), alpha_symm(alpha_symm), alpha_asymm(alpha_asymm) {
        // Check lengths match
        if (sigma_m.size() != mu.size()){
            throw teqp::InvalidArgument("bad size of mu");
        }
        if (sigma_m.size() != Q.size()){
            throw teqp::InvalidArgument("bad size of Q");
        }
    }
    GrayGubbins& operator=( const GrayGubbins& ) = delete; // non copyable
    
    /// Appendix B of Gray and Gubbins
    template<typename Jintegral, typename TTYPE, typename RhoStarType>
    auto get_In(const Jintegral& J, int n, double sigmaij, const TTYPE& Tstar, const RhoStarType& rhostar) const{
        return 4.0*PI_/pow(sigmaij, n-3)*J.get_J(Tstar, rhostar);
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MUPRIME>
    auto get_omegao2(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MUPRIME& muprime) const{
        const auto& x = mole_fractions; // concision
        
        const std::size_t N = mole_fractions.size();
        
        std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])> summer = 0.0;
        
        using namespace teqp::constants;
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const auto muprime2 = muprime.pow(2);
        const auto alpha_isotropic = (alpha_symm + 2*alpha_asymm)/3.0; // m^3
        const auto z1 = forceeval(1.0/3.0*muprime2 * beta + alpha_isotropic/k_e); // C^2m^2/J
        const auto z2 = alpha_isotropic/k_e; // C^2m^2/J
        
        // Each factor of \rho^\alpha gives x^\alpha*rhoN
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                TTYPE Tstarij = forceeval(T/EPSKIJ(i, j));
                double sigmaij = SIGMAIJ(i,j);
                summer += x[i]*x[j]*(
                     3.0/2.0*(z1[i]*z1[j] - z2[i]*z2[j])*get_In(J6, 6, sigmaij, Tstarij, rhostar)
                );
            }
        }
        // At this point, summer is in units of C^4/J^2/m^5
        return forceeval(rhoN*rhoN*k_e*k_e*summer); // The factor of k_e^2 takes us from CGS to SI units
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MUPRIME>
    auto get_omegao3(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MUPRIME& muprime) const{
        const VecType& x = mole_fractions; // concision
        const std::size_t N = mole_fractions.size();
        using type = std::common_type_t<TTYPE, RhoType, RhoStarType, decltype(mole_fractions[0]), decltype(muprime[0])>;
        type summer_a = 0.0, summer_b = 0.0;
        
        using namespace teqp::constants;
        const TTYPE beta = forceeval(1.0/(k_B*T));
        const auto muprime2 = muprime.pow(2);
        const auto alpha_isotropic = ((alpha_symm + 2.0*alpha_asymm)/3.0).eval();
        const auto z1 = forceeval((1.0/3.0*muprime2)*beta + alpha_isotropic/k_e);
        const auto z2 = alpha_isotropic/k_e;
        
        /// Following Appendix B of Gray and Gubbins
        const double PI3 = POW3(PI_);
        auto Immm = [&](std::size_t i, std::size_t j, std::size_t k, const auto& T, const auto& rhostar){
            auto Tstarij = T/EPSKIJ(i,j), Tstarik = T/EPSKIJ(i,k), Tstarjk = T/EPSKIJ(j, k);
            const double coeff = 64.0*PI3/5.0*sqrt(14*PI_/5.0)/SIGMAIJ(i,j)/SIGMAIJ(i,k)/SIGMAIJ(j,k);
            return coeff*get_Kijk(K222_333, rhostar, Tstarij, Tstarik, Tstarjk);
        };
        
        for (std::size_t i = 0; i < N; ++i){
            for (std::size_t j = 0; j < N; ++j){
                for (std::size_t k = 0; k < N; ++k){
                    auto b_ijk = 1.0/2.0*(z1[i]*z1[j]*z1[k] - z2[i]*z2[j]*z2[k])*Immm(i, j, k, T, rhostar);
                    summer_b += x[i]*x[j]*x[k]*b_ijk;
                }
            }
        }

        return forceeval((-rhoN*rhoN*summer_a - rhoN*rhoN*rhoN*summer_b)*k_e*k_e*k_e); // The factor of k_e^3 takes us from CGS to SI units
    }
    
    template<typename TTYPE, typename RhoType, typename RhoStarType, typename VecType, typename MUPRIME>
    auto get_perturbation(const TTYPE& T, const RhoType& rhoN, const RhoStarType& rhostar, const VecType& mole_fractions, const MUPRIME& muprime) const{
        auto w2 = get_omegao2(T, rhoN, rhostar, mole_fractions, muprime);
        auto w3 = get_omegao3(T, rhoN, rhostar, mole_fractions, muprime);
        return forceeval(-w2/(1.0-w3/w2));
    }
};

TEST_CASE("Test calculation of polarized dipole moment with polarizable Lennard-Jones", "[polarizability]")
{
    // These values don't matter, just pick something reasonable
    double sigma_m = 1e-10; // one Angstrom, in meters
    double epsilon_over_kB = 100.0; // K
    
    // T^*, rho^*, mu^*, alpha^*, mu^*_eff, reference
    std::vector<std::tuple<double, double, double, double, double, std::string>> db = {
        {1.002, 0.8344, 1.0, 0.03, 1.385, "Gray-MP-1985"},
        {1.002, 0.8344, 1.5, 0.03, 2.305, "Gray-MP-1985"},
        {1.15, 0.7, 1.0, 0.05/3, 1.117, "Kriebel-MP-1996,Table1"}, // Typos: Kriebel always gives mu^*, never (mu^*)^2 or (mu^*_{eff})^2
        {1.15, 0.7, 1.0, 0.1/3, 1.361, "Kriebel-MP-1996,Table1"},
        {1.35, 0.8, 0.5, 0.05/3, 0.53, "Kriebel-MP-1996,Table2"},
        {1.15, 0.822, 1.732, 0.05/3, 2.14, "Kriebel-MP-1996,Table3"},
    };
    
    for (const auto& [Tstar, rhostar, mustar, alphastar_iso, mustareff, ref] : db){
        auto alphastar_symm = alphastar_iso*3;
        auto alphastar_asymm = 0.0;
        
        double rhoN = rhostar/pow(sigma_m, 3);
        double alpha_symm = alphastar_symm*pow(sigma_m, 3);
        double alpha_asymm = alphastar_asymm*pow(sigma_m, 3);
        
        // Convert variables to SI units
        double T = Tstar*epsilon_over_kB;
        double Q = 0.0;
        
        using namespace teqp::constants;
        const double mu_Cm = mustar*sqrt(epsilon_over_kB*k_B*pow(sigma_m,3))/sqrt(k_e);
        
        const auto sigma_m_ = (Eigen::ArrayXd(1) << sigma_m).finished();
        const auto epsilon_over_kB_ = (Eigen::ArrayXd(1) << epsilon_over_kB).finished();
        const auto mu_ = (Eigen::ArrayXd(1) << mu_Cm).finished();
        const auto Q_ = (Eigen::ArrayXd(1) << Q).finished();
        const auto molefracs_ = (Eigen::ArrayXd(1) << 1.0).finished();
        const auto alpha_symm_ = (Eigen::ArrayXd(1) << alpha_symm).finished();
        const auto alpha_asymm_ = (Eigen::ArrayXd(1) << alpha_asymm).finished();
        
        auto N = mu_.size();
        Eigen::ArrayXXd SIGMAIJ(N,N);
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                SIGMAIJ(i,j) = (sigma_m_(i) + sigma_m_(j))/2.0;
                SIGMAIJ(j,i) = SIGMAIJ(i,j); // symmetric
            }
        }
        Eigen::ArrayXXd EPSKBIJ(N,N);
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                EPSKBIJ(i,j) = sqrt(epsilon_over_kB_(i)*epsilon_over_kB_(j));
                EPSKBIJ(j,i) = EPSKBIJ(i,j); // symmetric
            }
        }
        
        GrayGubbins<GubbinsTwuJIntegral, GubbinsTwuKIntegral> GG{sigma_m_, epsilon_over_kB_, SIGMAIJ, EPSKBIJ, mu_, Q_, alpha_symm_, alpha_asymm_, {}};
        
        // Successive substitution to update mu^*_eff
        Eigen::ArrayXd muprime = mu_;
        for (auto counter = 0; counter < 20; ++counter){
            double rhostar_ = rhostar;
            auto f = [&](const auto& muprime){
                return forceeval(GG.get_perturbation(T, rhoN, rhostar_, molefracs_, muprime));
            };
            ArrayXreal muprimead = muprime.cast<autodiff::real>();
            Eigen::ArrayXd dperturb_dmuprimead = autodiff::gradient(f, wrt(muprimead), at(muprimead));  // units of 1/(m^3)/(C m)
            
            Eigen::ArrayXd Eprime = -k_B*T/rhoN*dperturb_dmuprimead; // units of J /(C m) because perturb has units of 1/m^3
            // alpha*Eprime has units of J m^3/(C m), divide by k_e (has units of J m / C^2) to get C m
            muprime = mu_ + alpha_symm_/k_e*Eprime; // Units of C m
//            std::cout << muprime/(sqrt(epsilon_over_kB*k_B*pow(sigma_m,3))/sqrt(k_e)) << std::endl;
        }
        CAPTURE(ref);
        CAPTURE(Tstar);
        CAPTURE(mustar);
        CHECK(muprime[0]/(sqrt(epsilon_over_kB*k_B*pow(sigma_m,3))/sqrt(k_e)) == Approx(mustareff).margin(0.02));
        
        {
            auto flags = R"(
            {
            "polarizable": {
            "alpha_symm / m^3": [0.09e-30],
            "alpha_asymm / m^3": [0.00e-30]
            }
            }
            )"_json;
            flags["polarizable"]["alpha_symm / m^3"][0] = alpha_symm;
            MultipolarContributionGrayGubbins<GubbinsTwuJIntegral, GubbinsTwuKIntegral> GG{sigma_m_, epsilon_over_kB_, SIGMAIJ, EPSKBIJ, mu_, Q_, flags};
            double muprime = GG.iterate_muprime_SS(T, rhoN, rhostar, molefracs_, mu_, 20)[0];
            double mustareffcalc = muprime/(sqrt(epsilon_over_kB*k_B*pow(sigma_m,3))/sqrt(k_e));
            CAPTURE(ref);
            CAPTURE(Tstar);
            CAPTURE(mustar);
            CHECK(mustareffcalc == Approx(mustareff).margin(0.02) );
        }
    }
}

TEST_CASE("Test critical points against Kiyohara results", "[polarizability]"){
    auto j = R"({"kind": "SAFT-VR-Mie", "model": {"polar_model": "GrayGubbins+GubbinsTwu", "polar_flags": {"polarizable": {"alpha_symm / m^3": [18e-32], "alpha_asymm / m^3": [0.0]}}, "coeffs": [{"name": "PolarizableStockmayer", "BibTeXKey": "me", "m": 1.0, "epsilon_over_k": 100, "sigma_m": 1e-10, "lambda_r": 12.0, "lambda_a": 6.0, "mu_Cm": 3.919412183483701e-31, "nmu": 1.0}]}} )"_json;
    
    using namespace teqp::constants;
    
    double sigma_m = 1e-10, sigma3 = pow(sigma_m, 3);
    double epsilon_over_kB = 100.0;
    double epsilon = epsilon_over_kB*k_B;
    auto rhoNguess = 0.33/pow(sigma_m, 3);
    auto Tguess = 1.32*100;
    
    // From Kiyohara JCP 1997
    for (double mustar : {1, 2}){
        for (double alphastar: {0.0, 0.03, 0.06}){
            j["model"]["coeffs"][0]["mu_Cm"] = mustar*sqrt(epsilon*sigma3/k_e);
            j["model"]["polar_flags"]["polarizable"]["alpha_symm / m^3"][0] = alphastar*sigma3;
//            std::cout << j["model"]["polar_flags"]["polarizable"]["alpha_symm / m^3"][0] << std::endl;
            
            auto model = teqp::cppinterface::make_model(j);
            auto [Tc, rhoc] = model->solve_pure_critical(Tguess, rhoNguess/N_A);
            double Tstar = Tc/epsilon_over_kB;
            double rhostar = rhoc*N_A*pow(sigma_m,3);
            std::cout << mustar << "," << alphastar << "," << Tstar << "," << rhostar << std::endl;
        }
    }
}
