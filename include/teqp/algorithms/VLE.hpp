#pragma once

#include <optional>
#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include <Eigen/Dense>

// Imports from boost for numerical integration
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/euler.hpp>

namespace teqp{

template<typename Model, typename TYPE = double>
class IsothermPureVLEResiduals  {
    typedef Eigen::Array<TYPE, 2, 1> EigenArray;
    typedef Eigen::Array<TYPE, 1, 1> EigenArray1;
    typedef Eigen::Array<TYPE, 2, 2> EigenMatrix;
private:
    const Model& m_model;
    TYPE m_T;
    EigenMatrix J;
    EigenArray y;
    
public:
    std::size_t icall = 0;
    double Rr, R0;

    IsothermPureVLEResiduals(const Model& model, TYPE T) : m_model(model), m_T(T) {
        std::valarray<double> molefrac = { 1.0 };
        Rr = m_model.R(molefrac);
        R0 = m_model.R(molefrac);
    };

    const auto& get_errors() { return y; };

    auto call(const EigenArray& rhovec) {
        assert(rhovec.size() == 2);

        const EigenArray1 rhovecL = rhovec.head(1);
        const EigenArray1 rhovecV = rhovec.tail(1);
        const auto rhomolarL = rhovecL.sum(), rhomolarV = rhovecV.sum();
        const auto molefracs = (EigenArray1() << 1.0).finished();

        using tdx = TDXDerivatives<Model,TYPE,EigenArray1>;

        const TYPE &T = m_T;
        const TYPE R = m_model.R(molefracs);
        double R0_over_Rr = R0 / Rr;
        
        auto derL = tdx::template get_Ar0n<2>(m_model, T, rhomolarL, molefracs);
        auto pRTL = rhomolarL*(R0_over_Rr + derL[1]); // p/(R*T)
        auto dpRTLdrhoL = R0_over_Rr + 2*derL[1] + derL[2];
        auto hatmurL = derL[1] + derL[0] + R0_over_Rr*log(rhomolarL);
        auto dhatmurLdrho = (2*derL[1] + derL[2])/rhomolarL + R0_over_Rr/rhomolarL;

        auto derV = tdx::template get_Ar0n<2>(m_model, T, rhomolarV, molefracs);
        auto pRTV = rhomolarV*(R0_over_Rr + derV[1]); // p/(R*T)
        auto dpRTVdrhoV = R0_over_Rr + 2*derV[1] + derV[2];
        auto hatmurV = derV[1] + derV[0] + R0_over_Rr *log(rhomolarV);
        auto dhatmurVdrho = (2*derV[1] + derV[2])/rhomolarV + R0_over_Rr/rhomolarV;

        y(0) = pRTL - pRTV;
        J(0, 0) = dpRTLdrhoL;
        J(0, 1) = -dpRTVdrhoV;

        y(1) = hatmurL - hatmurV;
        J(1, 0) = dhatmurLdrho;
        J(1, 1) = -dhatmurVdrho;

        icall++;
        return y;
    }
    auto Jacobian(const EigenArray& rhovec){
        return J;
    }
    //auto numJacobian(const EigenArray& rhovec) {
    //    EigenArray plus0 = rhovec, plus1 = rhovec;
    //    double dr = 1e-6 * rhovec[0];
    //    plus0[0] += dr; plus1[1] += dr;
    //    EigenMatrix J;
    //    J.col(0) = (call(plus0) - call(rhovec))/dr;
    //    J.col(1) = (call(plus1) - call(rhovec))/dr;
    //    return J;
    //}
};

template<typename Residual, typename Scalar>
Eigen::ArrayXd do_pure_VLE_T(Residual &resid, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto rhovec = (Eigen::ArrayXd(2) << rhoL, rhoV).finished();
    auto r0 = resid.call(rhovec);
    auto J = resid.Jacobian(rhovec);
    for (int iter = 0; iter < maxiter; ++iter){
        if (iter > 0) {
            r0 = resid.call(rhovec);
            J = resid.Jacobian(rhovec); 
        }
        auto v = J.matrix().colPivHouseholderQr().solve(-r0.matrix()).array().eval();
        auto rhovecnew = (rhovec + v).eval();
        
        // If the solution has stopped improving, stop. The change in rhovec is equal to v in infinite precision, but 
        // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
        // the values are done changing
        if (((rhovecnew - rhovec).cwiseAbs() < std::numeric_limits<Scalar>::min()).all()) {
            break;
        }
        rhovec = rhovecnew;
    }
    return (Eigen::ArrayXd(2) << rhovec[0], rhovec[1]).finished();
}

template<typename Model, typename Scalar>
Eigen::ArrayXd pure_VLE_T(const Model& model, Scalar T, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto res = IsothermPureVLEResiduals(model, T);
    return do_pure_VLE_T(res, rhoL, rhoV, maxiter);
}

enum class VLE_return_code { unset, xtol_satisfied, functol_satisfied, maxiter_met };

/***
* \brief Do a vapor-liquid phase equilibrium problem for a mixture (binary only for now) with mole fractions specified in the liquid phase
* \param model The model to operate on
* \param T Temperature
* \param rhovecL0 Initial values for liquid mole concentrations
* \param rhovecV0 Initial values for vapor mole concentrations
* \param xspec Specified mole fractions for all components
* \param atol Absolute tolerance on function values
* \param reltol Relative tolerance on function values
* \param axtol Absolute tolerance on steps in independent variables
* \param relxtol Relative tolerance on steps in independent variables
* \param maxiter Maximum number of iterations permitted
*/
template<typename Model, typename Scalar, typename Vector>
auto mix_VLE_Tx(const Model& model, Scalar T, const Vector& rhovecL0, const Vector& rhovecV0, const Vector& xspec, double atol, double reltol, double axtol, double relxtol, int maxiter) {

    const Eigen::Index N = rhovecL0.size();
    Eigen::MatrixXd J(2 * N, 2 * N), r(2 * N, 1), x(2 * N, 1);
    x.col(0).array().head(N) = rhovecL0;
    x.col(0).array().tail(N) = rhovecV0;
    using isochoric = IsochoricDerivatives<Model, Scalar, Vector>;

    Eigen::Map<Eigen::ArrayXd> rhovecL(&(x(0)), N);
    Eigen::Map<Eigen::ArrayXd> rhovecV(&(x(0 + N)), N);
    auto RT = model.R(xspec) * T;

    VLE_return_code return_code = VLE_return_code::unset;

    for (int iter = 0; iter < maxiter; ++iter) {

        auto [PsirL, PsirgradL, hessianL] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
        auto [PsirV, PsirgradV, hessianV] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecV);
        auto rhoL = rhovecL.sum();
        auto rhoV = rhovecV.sum();
        Scalar pL = rhoL * RT - PsirL + (rhovecL.array() * PsirgradL.array()).sum(); // The (array*array).sum is a dot product
        Scalar pV = rhoV * RT - PsirV + (rhovecV.array() * PsirgradV.array()).sum();
        auto dpdrhovecL = RT + (hessianL * rhovecL.matrix()).array();
        auto dpdrhovecV = RT + (hessianV * rhovecV.matrix()).array();

        r(0) = PsirgradL(0) + RT * log(rhovecL(0)) - (PsirgradV(0) + RT * log(rhovecV(0)));
        r(1) = PsirgradL(1) + RT * log(rhovecL(1)) - (PsirgradV(1) + RT * log(rhovecV(1)));
        r(2) = pL - pV;
        r(3) = rhovecL(0) / rhovecL.sum() - xspec(0);

        // Chemical potential contributions in Jacobian
        J(0, 0) = hessianL(0, 0) + RT / rhovecL(0);
        J(0, 1) = hessianL(0, 1);
        J(1, 0) = hessianL(1, 0); // symmetric, so same as above
        J(1, 1) = hessianL(1, 1) + RT / rhovecL(1);
        J(0, 2) = -(hessianV(0, 0) + RT / rhovecV(0));
        J(0, 3) = -(hessianV(0, 1));
        J(1, 2) = -(hessianV(1, 0)); // symmetric, so same as above
        J(1, 3) = -(hessianV(1, 1) + RT / rhovecV(1));
        // Pressure contributions in Jacobian
        J(2, 0) = dpdrhovecL(0);
        J(2, 1) = dpdrhovecL(1);
        J(2, 2) = -dpdrhovecV(0);
        J(2, 3) = -dpdrhovecV(1);
        // Mole fraction composition specification in Jacobian
        J.row(3).array() = 0.0;
        J(3, 0) = (rhoL - rhovecL(0)) / (rhoL * rhoL); // dxi/drhoj (j=i)
        J(3, 1) = -rhovecL(0) / (rhoL * rhoL); // dxi/drhoj (j!=i)

        // Solve for the step
        Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-r);
        x.array() += dx;

        auto xtol_threshold = (axtol + relxtol * x.array().cwiseAbs()).eval();
        if ((dx.array() < xtol_threshold).all()) {
            return_code = VLE_return_code::xtol_satisfied;
            break;
        }

        auto error_threshold = (atol + reltol * r.array().cwiseAbs()).eval();
        if ((r.array().cwiseAbs() < error_threshold).all()) {
            return_code = VLE_return_code::functol_satisfied;
            break;
        }

        // If the solution has stopped improving, stop. The change in x is equal to dx in infinite precision, but 
        // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
        // the values are done changing
        if (((x.array() - dx.array()).cwiseAbs() < std::numeric_limits<Scalar>::min()).all()) {
            return_code = VLE_return_code::xtol_satisfied;
            break;
        }
        if (iter == maxiter - 1){
            return_code = VLE_return_code::maxiter_met;
        }
    }
    Eigen::ArrayXd rhovecLfinal = rhovecL, rhovecVfinal = rhovecV;
    return std::make_tuple(return_code, rhovecLfinal, rhovecVfinal);
}

template<typename Model, typename Scalar>
Eigen::ArrayXd extrapolate_from_critical(const Model& model, const Scalar Tc, const Scalar rhoc, const Scalar T) {
    
    using tdx = TDXDerivatives<Model>;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto R = model.R(z);
    auto ders = tdx::template get_Ar0n<4>(model, Tc, rhoc, z);
    auto dpdrho = R*Tc*(1 + 2 * ders[1] + ders[2]); // Should be zero
    auto d2pdrho2 = R*Tc/rhoc*(2 * ders[1] + 4 * ders[2] + ders[3]); // Should be zero
    auto d3pdrho3 = R*Tc/(rhoc*rhoc)*(6 * ders[2] + 6 * ders[3] + ders[4]);
    auto Ar11 = tdx::template get_Ar11(model, Tc, rhoc, z);
    auto Ar12 = tdx::template get_Ar12(model, Tc, rhoc, z);
    auto d2pdrhodT = R * (1 + 2 * ders[1] + ders[2] - 2 * Ar11 - Ar12);
    auto Brho = sqrt(6*d2pdrhodT*Tc/d3pdrho3);

    auto drhohat_dT = Brho / Tc;
    auto dT = T - Tc;

    auto drhohat = dT * drhohat_dT;
    auto rholiq = -drhohat/sqrt(1 - T/Tc) + rhoc;
    auto rhovap = drhohat/sqrt(1 - T/Tc) + rhoc;
    return (Eigen::ArrayXd(2) << rholiq, rhovap).finished();
}

template<class A, class B>
auto linsolve(const A &a, const B& b) {
    return a.matrix().colPivHouseholderQr().solve(b.matrix()).array().eval();
}

template<class Model, class Scalar, class VecType>
auto get_drhovecdp_Tsat(const Model& model, const Scalar &T, const VecType& rhovecL, const VecType& rhovecV) {
    //tic = timeit.default_timer();
    using id = IsochoricDerivatives<Model, Scalar, VecType>;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hliq = id::build_Psi_Hessian_autodiff(model, T, rhovecL).eval();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hvap = id::build_Psi_Hessian_autodiff(model, T, rhovecV).eval();
    //Hvap[~np.isfinite(Hvap)] = 1e20;
    //Hliq[~np.isfinite(Hliq)] = 1e20;

    auto N = rhovecL.size();
    Eigen::MatrixXd A = decltype(Hliq)::Zero(N, N);
    auto b = decltype(Hliq)::Ones(N, 1);
    decltype(Hliq) drhodp_liq, drhodp_vap;
    assert(rhovecL.size() == rhovecV.size());
    if ((rhovecL != 0).all() && (rhovecV != 0).all()) {
        // Normal treatment for all concentrations not equal to zero
        A(0, 0) = Hliq.row(0).dot(rhovecV.matrix());
        A(0, 1) = Hliq.row(1).dot(rhovecV.matrix());
        A(1, 0) = Hliq.row(0).dot(rhovecL.matrix());
        A(1, 1) = Hliq.row(1).dot(rhovecL.matrix());

        drhodp_liq = linsolve(A, b);
        drhodp_vap = linsolve(Hvap, Hliq*drhodp_liq);
    }
    else{
        // Special treatment for infinite dilution
        auto murL = id::build_Psir_gradient_autodiff(model, T, rhovecL);
        auto murV = id::build_Psir_gradient_autodiff(model, T, rhovecV);
        auto RL = model.R(rhovecL / rhovecL.sum());
        auto RV = model.R(rhovecV / rhovecV.sum());

        // First, for the liquid part
        for (auto i = 0; i < N; ++i) {
            for (auto j = 0; j < N; ++j) {
                if (rhovecL[j] == 0) {
                    // Analysis is special if j is the index that is a zero concentration.If you are multiplying by the vector
                    // of liquid concentrations, a different treatment than the case where you multiply by the vector
                    // of vapor concentrations is required
                    // ...
                    // Initial values
                    auto Aij = (Hliq.row(j).array().cwiseProduct(((i == 0) ? rhovecV : rhovecL).array().transpose())).eval(); // coefficient - wise product
                    // A throwaway boolean for clarity
                    bool is_liq = (i == 1);
                    // Apply correction to the j term (RT if liquid, RT*phi for vapor)
                    Aij[j] = (is_liq) ? RL*T : RL*T*exp(-(murV[j] - murL[j])/(RL*T));
                    // Fill in entry
                    A(i, j) = Aij.sum();
                }
                else{
                    // Normal
                    A(i, j) = Hliq.row(j).dot(((i==0) ? rhovecV : rhovecL).matrix());
                }
            }
        }
        drhodp_liq = linsolve(A, b);

        // Then, for the vapor part, also requiring special treatment
        // Left - multiplication of both sides of equation by diagonal matrix with liquid concentrations along diagonal, all others zero
        auto diagrhovecL = rhovecL.matrix().asDiagonal();
        auto PSIVstar = (diagrhovecL*Hvap).eval();
        auto PSILstar = (diagrhovecL*Hliq).eval();
        for (auto j = 0; j < N; ++j) {
            if (rhovecL[j] == 0) {
                PSILstar(j, j) = RL*T;
                PSIVstar(j, j) = RV*T/exp(-(murV[j] - murL[j]) / (RV * T));
            }
        }
        drhodp_vap = linsolve(PSIVstar, PSILstar*drhodp_liq);
    }
    return std::make_tuple(drhodp_liq, drhodp_vap);
}

/***
* \brief Derivative of molar concentration vectors w.r.t. T along an isobar of the phase envelope for binary mixtures
*
* \f[
* \left(\frac{d \vec\rho' }{d T}\right)_{p,\sigma}
* \f]
* 
* See Eq 13 and 14 of Deiters and Bell, AICHEJ: https://doi.org/10.1002/aic.16730
*/
template<class Model, class Scalar, class VecType>
auto get_drhovecdT_psat(const Model& model, const Scalar& T, const VecType& rhovecL, const VecType& rhovecV) {
    using id = IsochoricDerivatives<Model, Scalar, VecType>;
    if (rhovecL.size() != 2) { throw std::invalid_argument("Binary mixtures only"); }
    assert(rhovecL.size() == rhovecV.size());

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hliq = id::build_Psi_Hessian_autodiff(model, T, rhovecL).eval();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hvap = id::build_Psi_Hessian_autodiff(model, T, rhovecV).eval();

    auto N = rhovecL.size();
    Eigen::MatrixXd A = decltype(Hliq)::Zero(N, N);
    Eigen::MatrixXd R = decltype(Hliq)::Ones(N, 1);
    Eigen::MatrixXd drhodT_liq, drhodT_vap;
    
    if ((rhovecL != 0).all() && (rhovecV != 0).all()) {
        // Normal treatment for all concentrations not equal to zero
        A(0, 0) = Hliq.row(0).dot(rhovecV.matrix());
        A(0, 1) = Hliq.row(1).dot(rhovecV.matrix());
        A(1, 0) = Hliq.row(0).dot(rhovecL.matrix());
        A(1, 1) = Hliq.row(1).dot(rhovecL.matrix());

        VecType deltas = (id::get_dchempotdT_autodiff(model, T, rhovecV) - id::get_dchempotdT_autodiff(model, T, rhovecL)).eval();

        R(0,0) = (deltas.matrix().dot(rhovecV.matrix())-id::get_dpdT_constrhovec(model, T, rhovecV));
        R(1,0) = -id::get_dpdT_constrhovec(model, T, rhovecL);

        drhodT_liq = linsolve(A, R);
        drhodT_vap = linsolve(Hvap, ((Hliq*drhodT_liq).array() - deltas.array()).eval());
    }
    else {
        std::invalid_argument("Infinite dilution not yet supported");
    }
    return std::make_tuple(drhodT_liq, drhodT_vap);
}

/***
* \brief Derivative of molar concentration vectors w.r.t. T along an isopleth of the phase envelope for binary mixtures
* 
* The liquid phase will have its mole fractions held constant
* 
* \f[
* \left(\frac{d \vec\rho' }{d T}\right)_{x',\sigma} = \frac{\Delta s\dot \rho''-\Delta\beta_{\rho}}{(\Psi'\Delta\rho)\dot x')}x'
* \f]
* 
* See Eq 15 and 16 of Deiters and Bell, AICHEJ: https://doi.org/10.1002/aic.16730
* 
* To keep the vapor mole fraction constant, just swap the input molar concentrations to this function, the first concentration 
* vector is always the one with fixed mole fractions
*/
template<class Model, class Scalar, class VecType>
auto get_drhovecdT_xsat(const Model& model, const Scalar& T, const VecType& rhovecL, const VecType& rhovecV) {
    using id = IsochoricDerivatives<Model, Scalar, VecType>;

    if (rhovecL.size() != 2) { throw std::invalid_argument("Binary mixtures only"); }
    assert(rhovecL.size() == rhovecV.size());

    VecType molefracL = rhovecL / rhovecL.sum();
    VecType deltas = (id::get_dchempotdT_autodiff(model, T, rhovecV) - id::get_dchempotdT_autodiff(model, T, rhovecL)).eval();
    Scalar deltabeta = (id::get_dpdT_constrhovec(model, T, rhovecV)- id::get_dpdT_constrhovec(model, T, rhovecL));
    VecType deltarho = (rhovecV - rhovecL).eval();

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hliq = id::build_Psi_Hessian_autodiff(model, T, rhovecL).eval();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hvap = id::build_Psi_Hessian_autodiff(model, T, rhovecV).eval();
    
    Eigen::MatrixXd drhodT_liq, drhodT_vap;
    if ((rhovecL != 0).all() && (rhovecV != 0).all()) {
        auto num = (deltas.matrix().dot(rhovecV.matrix()) - deltabeta); // numerator, a scalar
        auto den = (Hliq*(deltarho.matrix())).dot(molefracL.matrix()); // denominator, a scalar
        drhodT_liq = num/den*molefracL;
        drhodT_vap = linsolve(Hvap, ((Hliq * drhodT_liq).array() - deltas.array()).eval());
    }
    else {
        throw std::invalid_argument("Infinite dilution not yet supported");
    }
    return std::make_tuple(drhodT_liq, drhodT_vap);
}

/**
* \brief Derivative of pressure w.r.t. temperature along the isopleth of a phase envelope (at constant composition of the bulk phase with the first concentration array)
* 
* Express \f$p(T,\rho,\vec x)\f$, so its total differential is
* \f[
* dp = \left(\frac{\partial p}{\partial T}\right)_{T,\vec x}dT + \left(\frac{\partial p}{\partial \rho}\right)_{T,\vec x} d\rho + \sum_{k} \left(\frac{\partial p}{\partial x_k}\right)_{T,\rho,x_{j\neq k}} dx_k
* \f]
* And for the derivative taken along the phase envelope at constant composition (along an isopleth so the composition part drops out):
* \f[
* \left(\frac{dp}{dT}\right)_{x, \sigma} = \left(\frac{\partial p}{\partial T}\right)_{T,\vec x}\frac{dT}{dT} + \left(\frac{\partial p}{\partial \rho}\right)_{T,\vec x} \left(\frac{d\rho}{dT}\right)_{\vec x,\sigma}
* \f]
* where
* \f[
\left(\frac{d\rho}{dT}\right)_{\vec x,\sigma} = \sum_k\left(\frac{d\rho_k}{dT}\right)_{\vec x,\sigma}
* \f]
*
* In the isochoric framework, a similar analysis would apply, which yields the identical result. Express \f$p(T,\vec\rho)\f$, so its total differential is
* \f[
* dp = \left(\frac{\partial p}{\partial T}\right)_{\vec\rho}dT + \sum_k \left(\frac{\partial p}{\partial \rho_k}\right)_{T,\rho_{j\neq k}} d\rho_k
* \f]
* And for the derivative taken along the phase envelope at constant composition (along an isopleth):
* \f[
* \left(\frac{dp}{dT}\right)_{x, \sigma} = \left(\frac{\partial p}{\partial T}\right)_{\vec\rho}\frac{dT}{dT} + \sum_k \left(\frac{\partial p}{\partial \rho_k}\right)_{T,\rho_{j\neq k}} \left(\frac{\partial \rho_k}{\partial T}\right)_{x,\sigma}
* \f]
*/
template<class Model, class Scalar, class VecType>
auto get_dpsat_dTsat_isopleth(const Model& model, const Scalar& T, const VecType& rhovecL, const VecType& rhovecV) {

    // Derivative along phase envelope at constant composition (correct, tested)
    auto [drhovecLdT_xsat, drhovecVdT_xsat] = get_drhovecdT_xsat(model, T, rhovecL, rhovecV);
    // And the derivative of the total density 
    auto drhoLdT_sat = drhovecLdT_xsat.sum();
    
    using tdx = TDXDerivatives<Model, Scalar, VecType>;
    double rhoL = rhovecL.sum();
    auto molefracL = rhovecL / rhoL;
    auto RT = model.R(molefracL) * T;
    auto derivs = tdx::template get_Ar0n<2>(model, T, rhoL, molefracL);
    auto dpdrho = RT*(1 + 2 * derivs[1] + derivs[2]);
    Scalar dpdT = model.R(molefracL) * rhoL * (1 + derivs[1] - tdx::get_Ar11(model, T, rhoL, molefracL));
    auto der = dpdT + dpdrho * drhoLdT_sat;
    return der;

    // How to do this derivative with isochoric formalism
    //using iso = IsochoricDerivatives<Model, Scalar, VecType>;
    //auto [PsirL, PsirgradL, hessianL] = iso::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
    //auto dpdrhovecL = (RT + (hessianL * rhovecL.matrix()).array()).eval();
    //auto der = (dpdrhovecL * drhovecLdT_xsat.array()).sum() + dpdT;
    //return der;
}

struct TVLEOptions {
    double init_dt = 1e-5, abs_err = 1e-8, rel_err = 1e-8, max_dt = 100000, init_c = 1.0;
    int max_steps = 1000, integration_order = 5;
    bool polish = true;
    bool calc_criticality = false;
};

/***
* \brief Trace an isotherm with parametric tracing
*/
template<typename Model, typename Scalar, typename VecType>
auto trace_VLE_isotherm_binary(const Model &model, Scalar T, VecType rhovecL0, VecType rhovecV0, const std::optional<TVLEOptions>& options = std::nullopt) 
{
    // Get the options, or the default values if not provided
    TVLEOptions opt = options.value_or(TVLEOptions{});
    auto N = rhovecL0.size();
    if (N != 2) {
        throw InvalidArgument("Size must be 2");
    }
    if (rhovecL0.size() != rhovecV0.size()) {
        throw InvalidArgument("Both molar concentration arrays must be of the same size");
    }

    auto norm = [](const auto& v) { return (v * v).sum(); };

    // Define datatypes and functions for tracing tools
    auto JSONdata = nlohmann::json::array();

    // Typedefs for the types
    using namespace boost::numeric::odeint;
    using state_type = std::vector<double>;

    // Class for simple Euler integration
    euler<state_type> eul;
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

    // Define the tolerances
    double abs_err = opt.abs_err, rel_err = opt.rel_err, a_x = 1.0, a_dxdt = 1.0;
    controlled_stepper_type controlled_stepper(default_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt));

    // Start off with the direction determined by c
    double c = opt.init_c;

    // Set up the initial state vector
    state_type x0(2 * N), last_drhodt(2 * N), previous_drhodt(2 * N);
    auto set_init_state = [&](state_type& X) {
        auto rhovecL = Eigen::Map<Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<Eigen::ArrayXd>(&(X[0]) + N, N);
        rhovecL = rhovecL0;
        rhovecV = rhovecV0;
    };
    set_init_state(x0);

    // The function to be integrated by odeint
    auto xprime = [&](const state_type& X, state_type& Xprime, double /*t*/) {
        // Memory maps into the state vector for inputs and their derivatives
        // These are views, not copies!
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[0]), N);
        auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[0]) + N, N);
        auto drhovecdtL = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), N);
        auto drhovecdtV = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]) + N, N);
        // Get the derivatives with respect to pressure along the isotherm of the phase envelope
        auto [drhovecdpL, drhovecdpV] = get_drhovecdp_Tsat(model, T, rhovecL, rhovecV);
        // Get the derivative of p w.r.t. parameter
        auto dpdt = 1.0/sqrt(norm(drhovecdpL.array()) + norm(drhovecdpV.array()));
        // And finally the derivatives with respect to the tracing variable
        drhovecdtL = c*drhovecdpL*dpdt;
        drhovecdtV = c*drhovecdpV*dpdt;

        if (previous_drhodt.empty()) {
            return;
        }

        // Flip the step if it changes direction from the smooth continuation of previous steps
        auto get_const_view = [&](const auto& v, int N) {
            return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), N);
        };
        if (get_const_view(Xprime, N).matrix().dot(get_const_view(previous_drhodt, N).matrix()) < 0) {
            auto Xprimeview = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), 2*N);
            Xprimeview *= -1;
        }
    };
    
    // Figure out which direction to trace initially
    double t = 0, dt = opt.init_dt;
    {
        auto dxdt = x0;
        xprime(x0, dxdt, -1.0);
        const auto dXdt = Eigen::Map<const Eigen::ArrayXd>(&(dxdt[0]), dxdt.size());
        const auto X = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), x0.size());

        const Eigen::ArrayXd step = X + dXdt * dt;
        Eigen::ArrayX<bool> negativestepvals = (step < 0).eval();
        // Flip the sign if the first step would yield any negative concentrations
        if (negativestepvals.any()) {
            c *= -1;
        }
    }
    std::string termination_reason;
    
    // Then trace...
    int retry_count = 0;
    for (auto istep = 0; istep < opt.max_steps; ++istep) {

        auto store_point = [&]() {
            //// Calculate some other parameters, for debugging
            auto N = x0.size() / 2;
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + N, N);
            auto rhototL = rhovecL.sum(), rhototV = rhovecV.sum();
            using id = IsochoricDerivatives<decltype(model), Scalar, VecType>;
            double pL = rhototL * model.R(rhovecL / rhovecL.sum())*T + id::get_pr(model, T, rhovecL);
            double pV = rhototV * model.R(rhovecV / rhovecV.sum())*T + id::get_pr(model, T, rhovecV);

            // Store the derivative
            try {
                xprime(x0, last_drhodt, -1.0);
            }
            catch (...) {
                std::cout << "Something bad happened; couldn't calculate xprime in store_point" << std::endl;
            }

            // Store the data in a JSON structure
            nlohmann::json point = {
                {"t", t},
                {"dt", dt},
                {"T / K", T},
                {"pL / Pa", pL},
                {"pV / Pa", pV},
                {"c", c},
                {"rhoL / mol/m^3", rhovecL},
                {"rhoV / mol/m^3", rhovecV},
                {"xL_0 / mole frac.", rhovecL[0]/rhovecL.sum()},
                {"xV_0 / mole frac.", rhovecV[0]/rhovecV.sum()},
                {"drho/dt", last_drhodt}
            };
            if (opt.calc_criticality) {
                using ct = CriticalTracing<Model, Scalar, VecType>;
                point["crit. conditions L"] = ct::get_criticality_conditions(model, T, rhovecL);
                point["crit. conditions V"] = ct::get_criticality_conditions(model, T, rhovecV);
            }
            JSONdata.push_back(point);
            //std::cout << JSONdata.back().dump() << std::endl;
        };
        if (istep == 0 && retry_count == 0) {
            store_point();
        }

        double dtold = dt;
        auto x0_previous = x0;

        if (opt.integration_order == 5) {
            controlled_step_result res = controlled_step_result::fail;
            try {
                res = controlled_stepper.try_step(xprime, x0, t, dt);
            }
            catch (...) {
                break;
            }

            if (res != controlled_step_result::success) {
                // Try again, with a smaller step size
                istep--;
                retry_count++;
                continue;
            }
            else {
                retry_count = 0;
            }
            // Reduce step size if greater than the specified max step size
            dt = std::min(dt, opt.max_dt);
        }
        else if (opt.integration_order == 1) {
            try {
                eul.do_step(xprime, x0, t, dt);
                t += dt;
            }
            catch (...) {
                break;
            }
        }
        else {
            throw InvalidArgument("integration order is invalid:" + std::to_string(opt.integration_order));
        }
        auto stop_requested = [&]() {
            //// Calculate some other parameters, for debugging
            auto N = x0.size() / 2;
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]) + N, N);
            auto x = rhovecL / rhovecL.sum();
            auto y = rhovecV / rhovecV.sum();
            if ((x < 0).any() || (x > 1).any() || (y < 0).any() || (y > 1).any()) {
                return true;
            }
            else {
                return false;
            }
        };
        if (stop_requested()) {
            break;
        }
        // Polish the solution
        if (opt.polish) {
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[0]), N).eval();
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[0 + N]), N).eval();
            auto x = (Eigen::ArrayXd(2) << rhovecL(0) / rhovecL.sum(), rhovecL(1) / rhovecL.sum()).finished(); // Mole fractions in the liquid phase (to be kept constant)
            auto [return_code, rhovecLnew, rhovecVnew] = mix_VLE_Tx(model, T, rhovecL, rhovecV, x, 1e-10, 1e-8, 1e-10, 1e-8, 10);

            // If the step is accepted, copy into x again ...
            auto rhovecLview = Eigen::Map<Eigen::ArrayXd>(&(x0[0]), N);
            auto rhovecVview = Eigen::Map<Eigen::ArrayXd>(&(x0[0]) + N, N);
            rhovecLview = rhovecLnew;
            rhovecVview = rhovecVnew;
            //std::cout << "[polish]: " << static_cast<int>(return_code) << ": " << rhovecLnew.sum() / rhovecL.sum() << " " << rhovecVnew.sum() / rhovecV.sum() << std::endl;
        }

        std::swap(previous_drhodt, last_drhodt);
        store_point(); // last_drhodt is updated;
        
    }
    return JSONdata;
}

}; /* namespace teqp*/