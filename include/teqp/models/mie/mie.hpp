#pragma once

#include "teqp/types.hpp"
#include <Eigen/Dense>

namespace teqp {
namespace Mie{

    /**
     Equation of state for the Mie (\f$\lambda_{\mathrm{r}}\f$,6) fluid with a repulsive exponent from 11 to 13
     Pohl, S.; Fingerhut, R., Thol, M.; Vrabec, J.; Span, R.
     */
    class Mie6Pohl2023 {
        
    private:
        using EArray6 = Eigen::Array<double, 6, 1>;
        using EArray4 = Eigen::Array<double, 4, 1>;
        
        const EArray6 c1_pol = (EArray6() << -0.0192944,1.38,-2.2653,1.6291,-1.974,0.40412 ).finished();
        const EArray6 c1_exp = (EArray6() <<  0.1845,-0.3227,1.1351,2.232,-2.344,-0.4238 ).finished();
        const EArray4 c1_gbs = (EArray4() <<  -4.367,0.0371,1.3895,2.835 ).finished();
        const EArray6 c2_pol = (EArray6() <<  0.26021,-5.525,8.329,-19.492,25.8,-3.8133 ).finished();
        const EArray6 c2_exp = (EArray6() <<  -5.05,2.7842,-9.523,-30.383,17.902,2.2264 ).finished();
        const EArray4 c2_gbs = (EArray4() <<  48.445,-5.506,-11.643,-24.36 ).finished();
        const EArray6 t_pol = (EArray6() <<  1,0.236,0.872,0.313,0.407,0.703 ).finished();
        const EArray6 t_exp = (EArray6() <<  1.78,2.99,2.866,1.2,3.06,1.073 ).finished();
        const EArray4 t_gbs = (EArray4() <<  1.50,1.03,4.02,1.57 ).finished();
        const EArray6 d_pol = (EArray6() <<  4,1,1,2,2,3 ).finished();
        const EArray6 d_exp = (EArray6() <<  1,1,3,2,2,5 ).finished();
        const EArray4 d_gbs = (EArray4() <<  2,3,2,2 ).finished();
        const EArray6 p = (EArray6() <<  1,2,2,1,2,1 ).finished();
        const EArray4 eta = (EArray4() <<  0.362,0.313,1.17,0.957 ).finished();
        const EArray4 beta = (EArray4() <<  0.0761,0.143,0.63,1.32 ).finished();
        const EArray4 gam = (EArray4() <<  1.55,-0.0826,1.505,1.07 ).finished();
        const EArray4 eps = (EArray4() <<  -1,-1,-0.195,-0.287 ).finished();
        
        const double m_lambda_r;
        const EArray6 n_pol, n_exp;
        const EArray4 n_gbs;
        const double Tc, rhoc; // In simulation units
    public:
        
        Mie6Pohl2023(double lambda_r) : m_lambda_r(lambda_r),
        n_pol(c1_pol + c2_pol / m_lambda_r),
        n_exp(c1_exp + c2_exp / m_lambda_r),
        n_gbs(c1_gbs + c2_gbs / m_lambda_r),
        Tc(0.668 + 6.84 / m_lambda_r + 145 / pow(m_lambda_r, 3)), // T^*
        rhoc(0.2516 + 0.049 * log10(m_lambda_r)) // rho^*
        {}
        
        auto get_lambda_r() const { return m_lambda_r; }

        // We are in "simulation units", so R is 1.0, and T and rho that
        // go into alphar are actually T^* and rho^*
        template<typename MoleFracType>
        double R(const MoleFracType &) const { return 1.0; }

        template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
        auto alphar(const TTYPE& Tstar, const RHOTYPE& rhostar, const MoleFracType& /*molefrac*/) const {
            auto tau = forceeval(Tc / Tstar); auto delta = forceeval(rhostar / rhoc);
            using _t = std::decay_t<std::common_type_t<TTYPE, RHOTYPE>>;
            _t s1 = 0; for (auto i = 0; i < n_pol.size(); ++i){ s1 += n_pol[i] * pow(tau, t_pol[i]) * powi(delta, static_cast<int>(d_pol[i])); }
            _t s2 = 0; for (auto i = 0; i < n_exp.size(); ++i){ s2 += n_exp[i] * pow(tau, t_exp[i]) * powi(delta, static_cast<int>(d_exp[i])) * exp(-powi(delta, static_cast<int>(p[i]))); }
            _t s3 = 0; for (auto i = 0; i < n_gbs.size(); ++i){ s3 += n_gbs[i] * pow(tau, t_gbs[i]) * powi(delta, static_cast<int>(d_gbs[i])) * exp(-eta[i] * powi(forceeval(delta - eps[i]), 2) - beta[i] * powi(forceeval(tau - gam[i]), 2)); }
            return forceeval(s1 + s2 + s3);
        }

    };

};

namespace FEANN{

    namespace FEANNMatrices{
        extern const Eigen::MatrixXd kernel_0, kernel_1, kernel_2, kernel_3, kernel_helmholtz;
        extern const Eigen::ArrayXd bias_0, bias_1, bias_2, bias_3;
    }

class ChaparroJCP2023 {
    
private:
    const double m_lambda_r, m_lambda_a, m_alpha;
    
    auto alpha_helper(double lambda_r, double lambda_a){
        auto c_alpha = lambda_r / (lambda_r-lambda_a) * pow(lambda_r/lambda_a, lambda_a/(lambda_r-lambda_a));
        auto alpha = c_alpha*(1.0/(lambda_a-3) - 1.0/(lambda_r-3));
        return alpha;
    }
public:
    
    ChaparroJCP2023(double lambda_r, double lambda_a) : m_lambda_r(lambda_r), m_lambda_a(lambda_a), m_alpha(alpha_helper(m_lambda_r, m_lambda_a)){}
    
    auto get_lambda_r() const { return m_lambda_r; }
    auto get_lambda_a() const { return m_lambda_a; }
    auto get_alpha() const { return m_alpha; }

    // We are in "simulation units", so R is 1.0, and T and rho that
    // go into alphar are actually T^* and rho^*
    template<typename MoleFracType>
    double R(const MoleFracType &) const { return 1.0; }

    template<typename TTYPE, typename RHOTYPE, typename MoleFracType>
    auto alphar(const TTYPE& Tstar, const RHOTYPE& rhostar, const MoleFracType& /*molefrac*/) const {
        using namespace FEANNMatrices;
        
        using Type = std::decay_t<std::common_type_t<TTYPE, RHOTYPE>>;
        Eigen::RowVectorX<Type> x = (Eigen::ArrayX<Type>(3) << m_alpha, rhostar, 1.0/Tstar).finished();
        Eigen::RowVectorX<Type> x_rhoad0 = (Eigen::ArrayX<Type>(3) << m_alpha, 0.0, 1.0/Tstar).finished();
        
        x = tanh(((x*kernel_0.cast<Type>()).reshaped().array() + bias_0.cast<Type>()).array());
        x_rhoad0 = tanh(((x_rhoad0*kernel_0.cast<Type>()).reshaped().array() + bias_0.cast<Type>()).array());
        
        x = tanh(((x*kernel_1.cast<Type>()).reshaped().array() + bias_1.cast<Type>()).array());
        x_rhoad0 = tanh(((x_rhoad0*kernel_1.cast<Type>()).reshaped().array() + bias_1.cast<Type>()).array());
        
        x = tanh(((x*kernel_2.cast<Type>()).reshaped().array() + bias_2.cast<Type>()).array());
        x_rhoad0 = tanh(((x_rhoad0*kernel_2.cast<Type>()).reshaped().array() + bias_2.cast<Type>()).array());
        
        x = tanh(((x*kernel_3.cast<Type>()).reshaped().array() + bias_3.cast<Type>()).array());
        x_rhoad0 = tanh(((x_rhoad0*kernel_3.cast<Type>()).reshaped().array() + bias_3.cast<Type>()).array());
        
        // The last layer doesn't have bias
        x = x*kernel_helmholtz.cast<Type>();
        x_rhoad0 = x_rhoad0*kernel_helmholtz.cast<Type>();
        
        return forceeval((x - x_rhoad0).array()[0]/(Tstar));
    }
};

}
};
