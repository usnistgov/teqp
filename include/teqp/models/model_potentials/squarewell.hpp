#ifndef squarewell_h
#define squarewell_h

#include <valarray>
#include <map>

namespace teqp{
namespace squarewell{

#include "teqp/types.hpp"

/**
 Rodolfo Espíndola-Heredia, Fernando del Río and Anatol Malijevsky
 Optimized equation of the state of the
 square-well fluid of variable range based on
 a fourth-order free-energy expansion
 J. Chem. Phys. 130, 024509 (2009); https://doi.org/10.1063/1.3054361

 \f[
     V(r) = \left\lbrace \begin{array}{cc}
         \infty & r < \sigma \\
         -\varepsilon & \sigma < r < \lambda\sigma \\
         0 & r > \lambda \sigma
         \end{array}\right.
\f]
 
 Note: if needed, all the terms that don't depend on T or rho could be pre-calculated at model
 initialization for a small speed boost

 */
class EspindolaHeredia2009{
private:
    
    const double m_pi = 3.1415926535897932384626433;
    
    double __factorial(int i) const{ return tgamma(i+1); }
    
    const double lambda;
    
    const std::map<int, std::valarray<double>> phivals = {
        {1, {-1320.19, 5124.1, -8145.37, 6895.8, -3381.42, 968.739, -151.255, 9.98592}},
        {2, {1049.76, -4023.29, 6305.95, -5265.42, 2553.84, -727.3, 113.631, -7.56266}}
    };
    const std::map<int, std::valarray<double>> thetavals = {
        {3, {0.0, -945.597, 1326.61, -471.688, 0.0, 23.2271, -2.63477, 0.0}},
        {4, {0.0, 4131.09,-10501.1,8909.18,-2521.96,-16.7882,19.5315,-1.27373}}
    };
    
    const std::map<int, std::valarray<double>> gammanvals = {
        {1, {0, -59.0464, 26.098, 26.4454, 7.40136, 11.0743, -5.49152, 0.781823, -0.0319751, 0.827621, 0.605635, -0.254959, 0.0377111, -0.00210896 , 0.0000452328}},
        {2, {0, 214.316, -88.1394, 273.3, 95.9759, 71.1228, -40.2656, 5.94069,  -0.23842, -2.17558,  -1.29255,  0.554993, -0.0857543, 0.00492511, -0.000107067 }},
        {3, {0, -225.479, 88.8202, 250.472, 90.2606, 57.0274, -33.2376, 4.99527, -0.195714, 1.84677, 0.99813, -0.440314, 0.0708793, -0.00416274, 0.0000917291 }},
        {4, {0, 65.0504, -25.096, 74.3095, 26.2153, 18.4397, -10.0891, 1.50243, -0.057694, -1.87154, -1.01682, 0.445247, -0.0725107, 0.00427862, -0.0000949723}}
    };
    
    template<typename GType>
    double Rn(const GType &gn, double lambda_) const{
        auto o = gn[3];
        for (auto j = 4; j < 9; ++j){
            o += gn[j]*pow(pow(lambda_,3)-1, j-2);
        }
        return o;
    }
    
    template<typename GType>
    double Qn(const GType &gn, double lambda_) const{
        auto o = gn[9];
        for (auto j = 10; j < 15; ++j){
            o += gn[j]*pow(pow(lambda_,3)-1, j-7);
        }
        return o;
    }

    double gamman(int n, double lambda_) const{
        const auto& gn = gammanvals.at(n);
        return gn[1]*lambda_ + gn[2]*pow(lambda_,2) + Rn(gn, lambda_)/Qn(gn, lambda_);
    }
    
    double phii(int i, double lambda_) const{
        const auto& phivalsi = phivals.at(i);
        double o = 0.0;
        for (auto n = 0; n < 8; ++n){
            o += phivalsi[n]*pow(lambda_, n);
        }
        return o;
    };
    
    double P1(double lambda_) const{return pow(lambda_,6) - 18*pow(lambda_,4) + 32*pow(lambda_,3) - 15;}
    double P2(double lambda_) const{return -2*pow(lambda_,6) + 36*pow(lambda_,4) - 32*pow(lambda_,3) - 18*pow(lambda_,2) + 16;}
    double P3(double lambda_) const{return 6*pow(lambda_,6) - 18*pow(lambda_,4) + 18*pow(lambda_,2)-6;}
    double P4(double lambda_) const{return 32*pow(lambda_,3) - 18*pow(lambda_,2) - 48;}
    double P5(double lambda_) const{return 5*pow(lambda_,6) - 32*pow(lambda_,3) + 18*pow(lambda_,2) + 26;};
    
    double a2i(int i, double lambda_) const{ return -2*m_pi/(3*__factorial(i))*(pow(lambda_, 3)-1); };
    
    double a31(double lambda_) const{ return -pow(m_pi/6, 2)*(P1(std::min(lambda_, 2.0)));};
    
    double a32(double lambda_) const {
        if (lambda_ <= 2)
            return pow(m_pi/6,2)*(P2(lambda_) - P1(lambda_)/2);
        else
            return pow(m_pi/6,2)*(-17/2 + P4(lambda_));
    }

    double a33(double lambda_) const {
        if (lambda_ <= 2)
            return pow(m_pi/6,2)*(P2(lambda_) - P1(lambda_)/6 - P3(lambda_));
        else
            return pow(m_pi/6,2)*(-17/6 + P4(lambda_) - P5(lambda_));
    }

    auto a34(double lambda_) const{
        if (lambda_ <= 2)
            return pow(m_pi/6,2)*(-P1(lambda_)/24 + 7*P2(lambda_)/12 - 3*P3(lambda_)/2);
        else
            return pow(m_pi/6,2)*(-17/24 + 7*P4(lambda_)/12 - 3*P5(lambda_)/2);
    }
        
    double xi2(double lambda_) const{ return a32(lambda_)/a2i(2, lambda_); }
    double xi3(double lambda_) const{ return a33(lambda_)/a2i(3, lambda_); }
    double xi4(double lambda_) const{ return a34(lambda_)/a2i(4, lambda_); }
    
    template<typename RhoType>
    auto Ki(int i, const RhoType & rhostar, double lambda_) const{
        const auto & thetai = thetavals.at(i);
        RhoType num = 0.0;
        for (auto n = 1; n < 5; ++n){
            num += thetai[n]*pow(lambda_, n);
        }
        num *= powi(rhostar, 2);
        RhoType den = 0;
        for (auto n = 5; n < 8; ++n){
            den += thetai[n]*pow(lambda_, n-4);
        }
        den = 1.0 + rhostar*den;
        return forceeval(num/den);
    }
    
    template<typename RhoType>
    auto Chi(const RhoType & rhostar, double lambda_) const { return forceeval(a2i(2, lambda_)*rhostar*(1.0-powi(rhostar,2)/1.5129)); }
    
    template<typename RhoType>
    auto aHS(const RhoType & rhostar) const{
        return forceeval(-3.0*m_pi*rhostar*(m_pi*rhostar-8.0)/powi(forceeval(m_pi*rhostar-6.0), 2));
    }
    
    template<typename RhoType>
    auto get_a1(const RhoType & rhostar, double lambda_) const{
        RhoType o = a2i(1, lambda_)*powi(rhostar, 2-1) + a31(lambda_)*powi(rhostar, 3-1);
        for (auto i = 1; i < 5; ++i){
            o = o + gamman(i, lambda_)*powi(rhostar, i+2);
        }
        return forceeval(o);
    }
    
    template<typename RhoType>
    auto get_a2(const RhoType & rhostar, double lambda_) const{
        return forceeval(Chi(rhostar, lambda_)*exp(xi2(lambda_)*rhostar + phii(1, lambda_)*powi(rhostar,3) + phii(2,lambda_)*powi(rhostar,4)));
    }
    
    template<typename RhoType>
    auto get_a3(const RhoType & rhostar, double lambda_) const {
        return forceeval(a2i(3, lambda_)*rhostar*exp(xi3(lambda_)*rhostar + Ki(3, rhostar, lambda_)));
    }
    
    template<typename RhoType>
    auto get_a4(const RhoType & rhostar, double lambda_) const {
        return forceeval(a2i(4, lambda_)*rhostar*exp(xi4(lambda_)*rhostar + Ki(4, rhostar, lambda_)));
    }
    
public:
    EspindolaHeredia2009(double lambda) : lambda(lambda){};
    
    // We are in "simulation units", so R is 1.0, and T and rho are T^* and rho^*
    template<typename MoleFracType>
    double R(const MoleFracType &) const { return 1.0; }
    
    /// Return the lambda parameter
    auto get_lambda() const {
        return lambda;
    }
    
    /**
        \param Tstar: \f$T^*=T/(\epsilon/k) \f$
        \param rhostar: \f$\rho^*=\rho_{\rm N}\sigma^3 \f$
        \note mole fractions must be provided, but are ignored in this case
     */
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& Tstar,
        const RhoType& rhostar,
        const MoleFracType& /*molefrac*/) const
    {
        auto a1 = get_a1(rhostar, lambda);
        auto a2 = get_a2(rhostar, lambda);
        auto a3 = get_a3(rhostar, lambda);
        auto a4 = get_a4(rhostar, lambda);
        
        return forceeval(aHS(rhostar)
            + a1/Tstar
            + a2/pow(Tstar, 2)
            + a3/pow(Tstar, 3)
            + a4/pow(Tstar, 4));
    }
};

}
}

#endif /* squarewell_h */
