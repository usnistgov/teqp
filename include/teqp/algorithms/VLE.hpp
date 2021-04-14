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

    IsothermPureVLEResiduals(const Model& model, TYPE T) : m_model(model), m_T(T) {};

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
        
        auto derL = tdx::template get_Ar0n<2>(m_model, T, rhomolarL, molefracs);
        auto pL = rhomolarL*R*T*(1+derL[1]);
        auto dpLdrho = R*T*(1 + 2*derL[1] + derL[2]);
        auto hatmurL = id::build_Psir_gradient_autodiff(m_model, T, rhovecL)[0] + R*T*log(rhomolarL);
        auto dhatmurLdrho = id::build_Psir_Hessian_autodiff(m_model, T, rhovecL)(0,0) + R*T/rhomolarL;

        auto derV = tdx::template get_Ar0n<2>(m_model, T, rhomolarV, molefracs);
        auto pV = rhomolarV*R*T*(1 + derV[1]);
        auto dpVdrho = R*T*(1 + 2*derV[1] + derV[2]);
        auto hatmurV = id::build_Psir_gradient_autodiff(m_model, T, rhovecV)[0] + R*T*log(rhomolarV);
        auto dhatmurVdrho = id::build_Psir_Hessian_autodiff(m_model, T, rhovecV)(0,0) + R*T/rhomolarV;

        y(0) = pL - pV;
        J(0, 0) = dpLdrho;
        J(0, 1) = -dpVdrho;

        y(1) = hatmurL - hatmurV;
        J(1, 0) = dhatmurLdrho;
        J(1, 1) = -dhatmurVdrho;

        icall++;
        return y;
    }
    auto Jacobian(const EigenArray& rhovec){
        return J;
    }
};

template<typename Residual, typename Scalar>
auto do_pure_VLE_T(Residual &resid, Scalar T, Scalar rhoL, Scalar rhoV, int maxiter) {
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
        rhovec += v;
    }
    return std::make_tuple(rhovec[0], rhovec[1]);
}