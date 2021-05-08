#pragma once

#include "teqp/derivs.hpp"
#include <Eigen/Dense>

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
        Rr = m_model.R;
        R0 = m_model.R;
    };

    const auto& get_errors() { return y; };

    auto call(const EigenArray& rhovec) {
        assert(rhovec.size() == 2);

        const EigenArray1 rhovecL = rhovec.head(1);
        const EigenArray1 rhovecV = rhovec.tail(1);
        const auto rhomolarL = rhovecL.sum(), rhomolarV = rhovecV.sum();
        const auto molefracs = (EigenArray1() << 1.0).finished();

        using id = IsochoricDerivatives<Model,TYPE,EigenArray1>;
        using tdx = TDXDerivatives<Model,TYPE,EigenArray1>;

        const TYPE &T = m_T;
        const TYPE R = m_model.R;
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
auto do_pure_VLE_T(Residual &resid, Scalar rhoL, Scalar rhoV, int maxiter) {
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
auto pure_VLE_T(const Model& model, Scalar T, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto res = IsothermPureVLEResiduals(model, T);
    return do_pure_VLE_T(res, rhoL, rhoV, maxiter);
}

template<typename Model, typename Scalar>
auto extrapolate_from_critical(const Model& model, const Scalar Tc, const Scalar rhoc, const Scalar T) {
    
    using tdx = TDXDerivatives<Model>;
    auto z = (Eigen::ArrayXd(1) << 1.0).finished();
    auto R = model.R;
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