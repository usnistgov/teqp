#pragma once

#include <optional>
#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/critical_pure.hpp"
#include <Eigen/Dense>

// Imports from boost for numerical integration
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/euler.hpp>


// Imports from Eigen unsupported for hybrj method
#include <unsupported/Eigen/NonLinearOptimization>

namespace teqp{

    using namespace Eigen;

    // Generic functor
    template<typename _Scalar, int NX = Dynamic, int NY = Dynamic>
    struct Functor
    {
        typedef _Scalar Scalar;
        enum {
            InputsAtCompileTime = NX,
            ValuesAtCompileTime = NY
        };
        typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
        typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
        typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

        const int m_inputs, m_values;

        Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
        Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

        int inputs() const { return m_inputs; }
        int values() const { return m_values; }

        // you should define that in the subclass :
      //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    };

// A convenience method to make linear system solving more concise with Eigen datatypes
/*** 
* All arguments are converted to matrices, the solve is done, and an array is returned
*/
template<class A, class B>
auto linsolve(const A& a, const B& b) {
    return a.matrix().colPivHouseholderQr().solve(b.matrix()).array().eval();
}

template<typename Model, typename TYPE = double, ADBackends backend = ADBackends::autodiff>
class IsothermPureVLEResiduals  {
public:
    using EigenArray = Eigen::Array<TYPE, 2, 1>;
    using EigenArray1 = Eigen::Array<TYPE, 1, 1>;
    using EigenMatrix = Eigen::Array<TYPE, 2, 2>;
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
        //const TYPE R = m_model.R(molefracs);
        double R0_over_Rr = R0 / Rr;
        
        auto derL = tdx::template get_Ar0n<2, backend>(m_model, T, rhomolarL, molefracs);
        auto pRTL = rhomolarL*(R0_over_Rr + derL[1]); // p/(R*T)
        auto dpRTLdrhoL = R0_over_Rr + 2*derL[1] + derL[2];
        auto hatmurL = derL[1] + derL[0] + R0_over_Rr*log(rhomolarL);
        auto dhatmurLdrho = (2*derL[1] + derL[2])/rhomolarL + R0_over_Rr/rhomolarL;

        auto derV = tdx::template get_Ar0n<2, backend>(m_model, T, rhomolarV, molefracs);
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
    using EArray = Eigen::Array<Scalar, 2, 1>;
    auto rhovec = (EArray() << rhoL, rhoV).finished();
    auto r0 = resid.call(rhovec);
    auto J = resid.Jacobian(rhovec);
    for (int iter = 0; iter < maxiter; ++iter){
        if (iter > 0) {
            r0 = resid.call(rhovec);
            J = resid.Jacobian(rhovec); 
        }
        auto v = J.matrix().colPivHouseholderQr().solve(-r0.matrix()).array().eval();
        auto rhovecnew = (rhovec + v).eval();
        //double r00 = static_cast<double>(r0[0]);
        //double r01 = static_cast<double>(r0[1]);
        
        // If the solution has stopped improving, stop. The change in rhovec is equal to v in infinite precision, but 
        // not when finite precision is involved, use the minimum non-denormal float as the determination of whether
        // the values are done changing
        auto minval = std::numeric_limits<Scalar>::epsilon();
        //double minvaldbl = static_cast<double>(minval);
        if (((rhovecnew - rhovec).cwiseAbs() < minval).all()) {
            break;
        }
        if ((r0.cwiseAbs() < minval).all()) {
            break;
        }
        rhovec = rhovecnew;
    }
    return rhovec;
}

template<typename Model, typename Scalar, ADBackends backend = ADBackends::autodiff>
auto pure_VLE_T(const Model& model, Scalar T, Scalar rhoL, Scalar rhoV, int maxiter) {
    auto res = IsothermPureVLEResiduals<Model, Scalar, backend>(model, T);
    return do_pure_VLE_T(res, rhoL, rhoV, maxiter);
}

enum class VLE_return_code { unset, xtol_satisfied, functol_satisfied, maxfev_met, maxiter_met, notfinite_step };

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
* 
* Note: if a mole fraction is zero in the provided vector, the molar concentrations in 
* this component will not be allowed to change (they will stay zero, avoiding the possibility that 
* they go to a negative value, which can cause trouble for some EOS)
*/
template<typename Model, typename Scalar, typename Vector>
auto mix_VLE_Tx(const Model& model, Scalar T, const Vector& rhovecL0, const Vector& rhovecV0, const Vector& xspec, double atol, double reltol, double axtol, double relxtol, int maxiter) {

    const Eigen::Index N = rhovecL0.size();
    auto lengths = (Eigen::ArrayXi(3) << rhovecL0.size(), rhovecV0.size(), xspec.size()).finished();
    if (lengths.minCoeff() != lengths.maxCoeff()){
        throw InvalidArgument("lengths of rhovecs and xspec must be the same in mix_VLE_Tx");
    }
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
        
        bool index0nonzero = rhovecL(0) > 0 && rhovecV(0) > 0;
        bool index1nonzero = rhovecL(1) > 0 && rhovecV(1) > 0;

        if (index0nonzero) {
            r(0) = PsirgradL(0) + RT * log(rhovecL(0)) - (PsirgradV(0) + RT * log(rhovecV(0)));
        } else {
            r(0) = PsirgradL(0) - PsirgradV(0);
        }
        if (index1nonzero){
            r(1) = PsirgradL(1) + RT * log(rhovecL(1)) - (PsirgradV(1) + RT * log(rhovecV(1)));
        } else {
            r(1) = PsirgradL(1) - PsirgradV(1);
        }
        r(2) = pL - pV;
        r(3) = rhovecL(0) / rhovecL.sum() - xspec(0);

        // Chemical potential contributions in Jacobian
        J(0, 0) = hessianL(0, 0) + (index0nonzero ? RT / rhovecL(0) : 0);
        J(0, 1) = hessianL(0, 1);
        J(1, 0) = hessianL(1, 0); // symmetric, so same as above
        J(1, 1) = hessianL(1, 1) + (index1nonzero ? RT / rhovecL(1) : 0);
        J(0, 2) = -(hessianV(0, 0) + (index0nonzero ? RT / rhovecV(0) : 0));
        J(0, 3) = -(hessianV(0, 1));
        J(1, 2) = -(hessianV(1, 0)); // symmetric, so same as above
        J(1, 3) = -(hessianV(1, 1) + (index1nonzero ? RT / rhovecV(1) : 0));
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

        if ((!dx.isFinite()).all()) {
            return_code = VLE_return_code::notfinite_step;
            break;
        }

        // Constrain the step to yield only positive densities
        if ((x.array() + dx.array() < 0).any()) {
            // The step that would take all the concentrations to zero
            Eigen::ArrayXd dxmax = -x;
            // Most limiting variable is the smallest allowed
            // before going negative
            auto f = (dx / dxmax).minCoeff();
            dx *= f / 2; // Only allow a step half the way to most constraining molar concentrations at most
        }

        // Don't allow changes to components with input zero mole fractions
        for (auto i = 0; i < 2; ++i) {
            if (xspec[i] == 0) {
                dx[i] = 0;
                dx[i+2] = 0;
            }
        }

        x.array() += dx;

        auto xtol_threshold = (axtol + relxtol * x.array().cwiseAbs()).eval();
        if ((dx.array().cwiseAbs() < xtol_threshold).all()) {
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

struct MixVLETpFlags {
    double atol = 1e-10,
        reltol = 1e-10,
        axtol = 1e-10,
        relxtol = 1e-10,
        relaxation = 1.0;
    int maxiter = 10;
};

struct MixVLEReturn {
    bool success = false;
    std::string message = "";
    Eigen::ArrayXd rhovecL, rhovecV;
    VLE_return_code return_code;
    int num_iter=-1, num_fev=-1;
    double T=-1;
    Eigen::ArrayXd r, initial_r;
};

template<typename Model>
struct hybrj_functor__mix_VLE_Tp : Functor<double>
{
    const Model& model;
    const double T, p;

    hybrj_functor__mix_VLE_Tp(const Model& model, const double T, const double p) : Functor<double>(4, 4), model(model), T(T), p(p) {}

    int operator()(const VectorXd& x, VectorXd& r)
    {
        const VectorXd::Index n = x.size() / 2;
        Eigen::Map<const Eigen::ArrayXd> rhovecL(&(x(0)), n);
        Eigen::Map<const Eigen::ArrayXd> rhovecV(&(x(0 + n)), n);
        auto RT = model.R((rhovecL / rhovecL.sum()).eval()) * T;
        using isochoric = IsochoricDerivatives<Model, Scalar, VectorXd>;
        auto [PsirL, PsirgradL, hessianL] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
        auto [PsirV, PsirgradV, hessianV] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecV);
        auto rhoL = rhovecL.sum();
        auto rhoV = rhovecV.sum();
        Scalar pL = rhoL * RT - PsirL + (rhovecL.array() * PsirgradL.array()).sum(); // The (array*array).sum is a dot product
        Scalar pV = rhoV * RT - PsirV + (rhovecV.array() * PsirgradV.array()).sum();

        bool index0nonzero = rhovecL(0) > 0 && rhovecV(0) > 0;
        bool index1nonzero = rhovecL(1) > 0 && rhovecV(1) > 0;

        if (index0nonzero) {
            r(0) = PsirgradL(0) + RT * log(rhovecL(0)) - (PsirgradV(0) + RT * log(rhovecV(0)));
        }
        else {
            r(0) = PsirgradL(0) - PsirgradV(0);
        }
        if (index1nonzero) {
            r(1) = PsirgradL(1) + RT * log(rhovecL(1)) - (PsirgradV(1) + RT * log(rhovecV(1)));
        }
        else {
            r(1) = PsirgradL(1) - PsirgradV(1);
        }
        r(2) = (pV - p) / p;
        r(3) = (pL - p) / p;
        return 0;
    }
    int df(const VectorXd& x, MatrixXd& J)
    {
        const VectorXd::Index n = x.size() / 2;
        Eigen::Map<const Eigen::ArrayXd> rhovecL(&(x(0)), n);
        Eigen::Map<const Eigen::ArrayXd> rhovecV(&(x(0 + n)), n);
        assert(J.rows() == 2*n);
        assert(J.cols() == 2*n);

        auto RT = model.R((rhovecL / rhovecL.sum()).eval()) * T;
        using isochoric = IsochoricDerivatives<Model, Scalar, VectorXd>;
        auto [PsirL, PsirgradL, hessianL] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
        auto [PsirV, PsirgradV, hessianV] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecV);
        auto dpdrhovecL = RT + (hessianL * rhovecL.matrix()).array();
        auto dpdrhovecV = RT + (hessianV * rhovecV.matrix()).array();

        bool index0nonzero = rhovecL(0) > 0 && rhovecV(0) > 0;
        bool index1nonzero = rhovecL(1) > 0 && rhovecV(1) > 0;

        // Chemical potential contributions in Jacobian
        J(0, 0) = hessianL(0, 0) + (index0nonzero ? RT / rhovecL(0) : 0);
        J(0, 1) = hessianL(0, 1);
        J(1, 0) = hessianL(1, 0); // symmetric, so same as above
        J(1, 1) = hessianL(1, 1) + (index1nonzero ? RT / rhovecL(1) : 0);
        J(0, 2) = -(hessianV(0, 0) + (index0nonzero ? RT / rhovecV(0) : 0));
        J(0, 3) = -(hessianV(0, 1));
        J(1, 2) = -(hessianV(1, 0)); // symmetric, so same as above
        J(1, 3) = -(hessianV(1, 1) + (index1nonzero ? RT / rhovecV(1) : 0));
        // Pressure contributions in Jacobian
        J(2, 0) = 0;
        J(2, 1) = 0;
        J(2, 2) = dpdrhovecV(0) / p;
        J(2, 3) = dpdrhovecV(1) / p;
        // Other pressure specification in Jacobian
        J.row(3).array() = 0.0;
        J(3, 0) = dpdrhovecL(0) / p;
        J(3, 1) = dpdrhovecL(1) / p;
        return 0;
    }
};

/***
* \brief Do a vapor-liquid phase equilibrium problem for a binary mixture with temperature and pressure specified
* 
* The mole concentrations are solved for to give the right pressure
* 
* \param model The model to operate on
* \param T Temperature
* \param pgiven Given pressure
* \param rhovecL0 Initial values for liquid mole concentrations
* \param rhovecV0 Initial values for vapor mole concentrations
* \param flags Flags controlling the iteration and stopping conditions
*/
template<typename Model, typename Scalar, typename Vector>
auto mix_VLE_Tp(const Model& model, Scalar T, Scalar pgiven, const Vector& rhovecL0, const Vector& rhovecV0, const MixVLETpFlags& flags = {}) {

    const Eigen::Index N = rhovecL0.size();
    auto lengths = (Eigen::ArrayXi(2) << rhovecL0.size(), rhovecV0.size()).finished();
    if (lengths.minCoeff() != lengths.maxCoeff()) {
        throw InvalidArgument("lengths of rhovecs must be the same in mix_VLE_Tx");
    }
    Eigen::VectorXd x(2*N, 1);
    x.col(0).array().head(N) = rhovecL0;
    x.col(0).array().tail(N) = rhovecV0;
    
    VLE_return_code return_code = VLE_return_code::unset;
    std::string message = "";

    using FunctorType = hybrj_functor__mix_VLE_Tp<Model>;
    FunctorType functor(model, T, pgiven);
    Eigen::VectorXd initial_r(2 * N); initial_r.setZero();
    functor(x, initial_r);

    bool success = false;
    bool powell = false;
    
    /*Eigen::MatrixXd J(2*N, 2*N), J2(2*N, 2*N);
    Eigen::VectorXd dxvec = 1e-4*final_r.array();
    for (auto i = 0; i < 4; ++i){
        Eigen::VectorXd xplus = x, rplus(4);
        xplus[i] += 1e-4*x[i];
        functor(xplus, rplus);
        J2.col(i) = (rplus.array() - final_r.array()) / (1e-4*x[i]);
    }
    functor.df(x, J);
    Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-final_r);*/

    int niter = 0, nfev = 0;
    Eigen::MatrixXd J(2 * N, 2 * N);
    if (powell) {
        HybridNonLinearSolver<FunctorType> solver(functor);
        
        solver.diag.setConstant(2 * N, 1.);
        solver.useExternalScaling = true;
        auto info = solver.solve(x);
        
        using e = Eigen::HybridNonLinearSolverSpace::Status;
        success = (info == e::RelativeErrorTooSmall || info == e::TolTooSmall);
        switch (info) {
        case e::ImproperInputParameters:
            return_code = VLE_return_code::notfinite_step;
        case e::RelativeErrorTooSmall:
            return_code = VLE_return_code::functol_satisfied;
        case e::TooManyFunctionEvaluation:
            return_code = VLE_return_code::maxfev_met;
        case e::TolTooSmall:
            return_code = VLE_return_code::xtol_satisfied;
            //NotMakingProgressJacobian = 4,
            //NotMakingProgressIterations = 5,
        default:
            return_code = VLE_return_code::unset;
        }
        niter = solver.iter;
        nfev = solver.nfev;
    }
    else {
        for (auto iter = 0; iter < flags.maxiter; ++iter) {
            Eigen::VectorXd rv(2 * N); rv.setZero();
            functor(x, rv);
            functor.df(x, J);
            Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-rv);
            if ((x.array() + dx.array() < 0).any()) {
                // The step that would take all the concentrations to zero
                Eigen::ArrayXd dxmax = -x;
                // Most limiting variable is the smallest allowed
                // before going negative
                auto f = (dx/dxmax).minCoeff();
                dx *= f/2; // Only allow a step half the way to most constraining molar concentrations at most
            }
            x.array() += dx.array();
            niter = iter;
            nfev = iter;
        }
    }
    Eigen::VectorXd final_r(2 * N); final_r.setZero();
    functor(x, final_r);

    Eigen::Map<const Eigen::ArrayXd> rhovecL(&(x(0)), N);
    Eigen::Map<const Eigen::ArrayXd> rhovecV(&(x(0 + N)), N);

    MixVLEReturn r;
    r.return_code = return_code;
    r.num_iter = niter;
    r.num_fev = nfev;
    r.r = final_r;
    r.initial_r = initial_r;
    r.success = success;
    r.rhovecL = rhovecL;
    r.rhovecV = rhovecV;
    r.T = T;
    return r;
}

struct MixVLEpxFlags {
    double atol = 1e-10,
        reltol = 1e-10,
        axtol = 1e-10,
        relxtol = 1e-10;
    int maxiter = 10;
};

/***
* \brief Do vapor-liquid phase equilibrium problem at specified pressure and mole fractions in the bulk phase
* \param model The model to operate on
* \param p_spec Specified pressure
* \param xmolar_spec Specified mole fractions for all components in the bulk phase
* \param T0 Initial temperature
* \param rhovecL0 Initial values for liquid mole concentrations
* \param rhovecV0 Initial values for vapor mole concentrations

* \param flags Additional flags
*/
template<typename Model, typename Scalar, typename Vector>
auto mixture_VLE_px(const Model& model, Scalar p_spec, const Vector& xmolar_spec, Scalar T0, const Vector& rhovecL0, const Vector& rhovecV0, const MixVLEpxFlags& flags = {}) {

    const Eigen::Index N = rhovecL0.size();
    auto lengths = (Eigen::ArrayXi(3) << rhovecL0.size(), rhovecV0.size(), xmolar_spec.size()).finished();
    if (lengths.minCoeff() != lengths.maxCoeff()) {
        throw InvalidArgument("lengths of rhovecs and xspec must be the same in mixture_VLE_px");
    }
    if ((rhovecV0 == 0).any()) {
        throw InvalidArgument("Infinite dilution is not allowed for rhovecV0 in mixture_VLE_px");
    }
    if ((rhovecL0 == 0).any()) {
        throw InvalidArgument("Infinite dilution is not allowed for rhovecL0 in mixture_VLE_px");
    }
    Eigen::MatrixXd J(2*N+1, 2*N+1); J.setZero();
    Eigen::VectorXd r(2*N + 1), x(2*N + 1);
    x(0) = T0;
    x.segment(1, N) = rhovecL0;
    x.tail(N) = rhovecV0;
    using isochoric = IsochoricDerivatives<Model, Scalar>;

    Eigen::Map<Eigen::ArrayXd> rhovecL(&(x(1)), N);
    Eigen::Map<Eigen::ArrayXd> rhovecV(&(x(1 + N)), N);

    double T = T0;

    VLE_return_code return_code = VLE_return_code::unset;

    for (int iter = 0; iter < flags.maxiter; ++iter) {

        auto RL = model.R(xmolar_spec);
        auto RLT = RL * T;
        auto RVT = RLT; // Note: this should not be exactly the same if you use mole-fraction-weighted gas constants
        
        // calculations from the EOS in the isochoric thermodynamics formalism
        auto [PsirL, PsirgradL, hessianL] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecL);
        auto [PsirV, PsirgradV, hessianV] = isochoric::build_Psir_fgradHessian_autodiff(model, T, rhovecV);
        auto DELTAdmu_dT_res = (isochoric::build_d2PsirdTdrhoi_autodiff(model, T, rhovecL.eval()) 
                              - isochoric::build_d2PsirdTdrhoi_autodiff(model, T, rhovecV.eval())).eval();

        auto make_diag = [](const Eigen::ArrayXd& v) -> Eigen::ArrayXXd {
            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(v.size(), v.size());
            A.diagonal() = v;
            return A;
        };
        auto HtotL = (hessianL.array() + make_diag(RLT/rhovecL)).eval();
        auto HtotV = (hessianV.array() + make_diag(RVT/rhovecV)).eval();

        auto rhoL = rhovecL.sum();
        auto rhoV = rhovecV.sum();
        Scalar pL = rhoL * RLT - PsirL + (rhovecL.array() * PsirgradL.array()).sum(); // The (array*array).sum is a dot product
        Scalar pV = rhoV * RVT - PsirV + (rhovecV.array() * PsirgradV.array()).sum();
        auto dpdrhovecL = RLT + (hessianL * rhovecL.matrix()).array();
        auto dpdrhovecV = RVT + (hessianV * rhovecV.matrix()).array();
        
        auto DELTA_dchempot_dT = (DELTAdmu_dT_res + RL*log(rhovecL/rhovecV)).eval();

        // First N equations are equalities of chemical potentials in both phases
        r.head(N) = PsirgradL + RLT*log(rhovecL) - (PsirgradV + RVT*log(rhovecV));
        // Next two are pressures in each phase equaling the specification
        r(N) = pL/p_spec - 1;
        r(N+1) = pV/p_spec - 1;
        // Remainder are N-1 mole fraction equalities in the liquid phase
        r.tail(N-1) = (rhovecL/rhovecL.sum()).head(N-1) - xmolar_spec.head(N-1);
        // So in total we have N + 2 + (N-1) = 2*N+1 equations and 2*N+1 independent variables

        // Columns in Jacobian are: [T, rhovecL, rhovecV]
        // ...
        // N Chemical potential contributions in Jacobian (indices 0 to N-1)
        J.block(0, 0, N, 1) = DELTA_dchempot_dT; 
        J.block(0, 1, N, N) = HtotL; // These are the concentration derivatives
        J.block(0, N+1, N, N) = -HtotV; // These are the concentration derivatives
        // Pressure contributions in Jacobian
        J(N, 0) = isochoric::get_dpdT_constrhovec(model, T, rhovecL)/p_spec;
        J.block(N, 1, 1, N) = dpdrhovecL.transpose()/p_spec;
        // No vapor concentration derivatives
        J(N+1, 0) = isochoric::get_dpdT_constrhovec(model, T, rhovecV)/p_spec;
        // No liquid concentration derivatives
        J.block(N+1, N+1, 1, N) = dpdrhovecV.transpose()/p_spec;
        // Mole fraction contributions in Jacobian
        // dxi/drhoj = (rho*Kronecker(i,j)-rho_i)/rho^2 since x_i = rho_i/rho
        //
        Eigen::ArrayXXd AA = rhovecL.matrix().reshaped(N, 1).replicate(1, N).array();
        Eigen::MatrixXd M = ((rhoL * Eigen::MatrixXd::Identity(N, N).array() - AA) / (rhoL * rhoL));
        J.block(N+2, 1, N-1, N) = M.block(0,0,N-1,N);

        // Solve for the step
        Eigen::ArrayXd dx = J.colPivHouseholderQr().solve(-r);

        if ((!dx.isFinite()).all()) {
            return_code = VLE_return_code::notfinite_step;
            break;
        }

        T += dx(0);
        x.tail(2*N).array() += dx.tail(2*N);

        auto xtol_threshold = (flags.axtol + flags.relxtol * x.array().cwiseAbs()).eval();
        if ((dx.array().cwiseAbs() < xtol_threshold).all()) {
            return_code = VLE_return_code::xtol_satisfied;
            break;
        }

        auto error_threshold = (flags.atol + flags.reltol * r.array().cwiseAbs()).eval();
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
        if (iter == flags.maxiter - 1) {
            return_code = VLE_return_code::maxiter_met;
        }
    }
    Eigen::ArrayXd rhovecLfinal = rhovecL, rhovecVfinal = rhovecV;
    return std::make_tuple(return_code, T, rhovecLfinal, rhovecVfinal);
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

/**
 * Derivative of molar concentration vectors w.r.t. p along an isobar of the phase envelope for binary mixtures
*/
template<class Model, class Scalar, class VecType>
auto get_drhovecdT_psat(const Model& model, const Scalar &T, const VecType& rhovecL, const VecType& rhovecV) {
    
    using id = IsochoricDerivatives<Model, Scalar, VecType>;
    if (rhovecL.size() != 2) { throw std::invalid_argument("Binary mixtures only"); }
    assert(rhovecL.size() == rhovecV.size());

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hliq = id::build_Psi_Hessian_autodiff(model, T, rhovecL).eval();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Hvap = id::build_Psi_Hessian_autodiff(model, T, rhovecV).eval();

    auto N = rhovecL.size();
    Eigen::MatrixXd A = decltype(Hliq)::Zero(N, N);
    Eigen::MatrixXd b = decltype(Hliq)::Ones(N, 1);
    decltype(Hliq) drhovecdT_liq, drhovecdT_vap;
    assert(rhovecL.size() == rhovecV.size());

    if ((rhovecL != 0).all() && (rhovecV != 0).all()) {
        // Normal treatment for all concentrations not equal to zero
        A(0, 0) = Hliq.row(0).dot(rhovecV.matrix());
        A(0, 1) = Hliq.row(1).dot(rhovecV.matrix());
        A(1, 0) = Hliq.row(0).dot(rhovecL.matrix());
        A(1, 1) = Hliq.row(1).dot(rhovecL.matrix());

        auto DELTAdmu_dT = (id::get_dchempotdT_autodiff(model, T, rhovecV) - id::get_dchempotdT_autodiff(model, T, rhovecL)).eval();
        b(0) = DELTAdmu_dT.matrix().dot(rhovecV.matrix()) - id::get_dpdT_constrhovec(model, T, rhovecV);
        b(1) = -id::get_dpdT_constrhovec(model, T, rhovecL);
        // Calculate the derivatives of the liquid phase
        drhovecdT_liq = linsolve(A, b);
        // Calculate the derivatives of the vapor phase
        drhovecdT_vap = linsolve(Hvap, ((Hliq*drhovecdT_liq).array() - DELTAdmu_dT.array()).eval());
    }
    else{
        // Special treatment for infinite dilution
        auto murL = id::build_Psir_gradient_autodiff(model, T, rhovecL);
        auto murV = id::build_Psir_gradient_autodiff(model, T, rhovecV);
        auto RL = model.R(rhovecL / rhovecL.sum());
        auto RV = model.R(rhovecV / rhovecV.sum());

        // The dot product contains terms of the type:
        // rho'_i (R ln(rho"_i /rho'_i) + d mu ^ r"_i/d T - d mu^r'_i/d T)

        // Residual contribution to the difference in temperature derivative of chemical potential
        // It should be fine to evaluate with zero densities:
        auto DELTAdmu_dT_res = (id::build_d2PsirdTdrhoi_autodiff(model, T, rhovecV) - id::build_d2PsirdTdrhoi_autodiff(model, T, rhovecL)).eval();
        // Now the ideal-gas part causes trouble, so multiply by the rhovec, once with liquid, another with vapor
        // Start off with the assumption that the rhovec is all positive (fix elements later)
        auto DELTAdmu_dT_rhoV_ideal = (rhovecV*(RV*log(rhovecV/rhovecL))).eval();
        auto DELTAdmu_dT_ideal = (RV*log(rhovecV/rhovecL)).eval();
        // Zero out contributions where a concentration is zero
        for (auto i = 0; i < rhovecV.size(); ++i) {
            if (rhovecV[i] == 0) {
                DELTAdmu_dT_rhoV_ideal(i) = 0;
                DELTAdmu_dT_ideal(i) = 0;
            }
        }
        double DELTAdmu_dT_rhoV = rhovecV.matrix().dot(DELTAdmu_dT_res.matrix()) + DELTAdmu_dT_rhoV_ideal.sum();
        
        b(0) = DELTAdmu_dT_rhoV - id::get_dpdT_constrhovec(model, T, rhovecV);
        b(1) = -id::get_dpdT_constrhovec(model, T, rhovecL);

        // First, for the liquid part
        for (auto i = 0; i < N; ++i) {
            // A throwaway boolean for clarity
            bool is_liq = (i == 1);
            for (auto j = 0; j < N; ++j) {
                auto rhovec = (is_liq) ? rhovecL : rhovecV;
                // Initial values
                auto Aij = (Hliq.row(j).array().cwiseProduct(rhovec.array().transpose())).eval(); // coefficient - wise product
                // Only rows in H that have a divergent entry in a column need to be adjusted
                if (!(Hliq.row(j)).array().isFinite().all()) { 
                    // A correction is needed in the entry in Aij corresponding to entry for zero concentration
                    if (rhovec[j] == 0) {
                        // Apply correction to the j term (RT if liquid, RT*phi for vapor)
                        Aij[j] = (is_liq) ? RL * T : RL * T * exp(-(murV[j] - murL[j]) / (RL * T));
                    }
                }
                
                // Fill in entry
                A(i, j) = Aij.sum();
            }
        }
        drhovecdT_liq = linsolve(A, b);

        // Then, for the vapor part, also requiring special treatment
        // Left-multiplication of both sides of equation by diagonal matrix with 
        // liquid concentrations along diagonal, all others zero
        auto diagrhovecL = rhovecL.matrix().asDiagonal();
        auto Hvapstar = (diagrhovecL*Hvap).eval();
        auto Hliqstar = (diagrhovecL*Hliq).eval();
        for (auto j = 0; j < N; ++j) {
            if (rhovecL[j] == 0) {
                Hliqstar(j, j) = RL*T;
                Hvapstar(j, j) = RV*T/ exp((murL[j] - murV[j]) / (RV * T)); // Note not as given in Deiters
            }
        }
        auto diagrhovecL_dot_DELTAdmu_dT = (diagrhovecL*(DELTAdmu_dT_res+DELTAdmu_dT_ideal).matrix()).array();
        auto RHS = ((Hliqstar * drhovecdT_liq).array() - diagrhovecL_dot_DELTAdmu_dT).eval();
        drhovecdT_vap = linsolve(Hvapstar.eval(), RHS);
    }
    return std::make_tuple(drhovecdT_liq, drhovecdT_vap);
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
    bool terminate_unstable = false;
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

        //double dtold = dt;
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
            // Check if the solution has gone mechanically unstable
            if (opt.calc_criticality) {
                using ct = CriticalTracing<Model, Scalar, VecType>;
                auto condsL = ct::get_criticality_conditions(model, T, rhovecL);
                auto condsV = ct::get_criticality_conditions(model, T, rhovecV);
                if (condsL[0] < 1e-12 || condsV[0] < 1e-12){
                    return true;
                }
            }
            if ((x < 0).any() || (x > 1).any() || (y < 0).any() || (y > 1).any() || (!rhovecL.isFinite()).any() || (!rhovecV.isFinite()).any()) {
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


struct PVLEOptions {
    double init_dt = 1e-5, abs_err = 1e-8, rel_err = 1e-8, max_dt = 100000, init_c = 1.0;
    int max_steps = 1000, integration_order = 5;
    bool polish = true;
    bool calc_criticality = false;
    bool terminate_unstable = false;
};

/***
* \brief Trace an isobar with parametric tracing
*/
template<typename Model, typename Scalar, typename VecType>
auto trace_VLE_isobar_binary(const Model& model, Scalar p, Scalar T0, VecType rhovecL0, VecType rhovecV0, const std::optional<PVLEOptions>& options = std::nullopt)
{
    // Get the options, or the default values if not provided
    PVLEOptions opt = options.value_or(PVLEOptions{});
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
    state_type x0(2*N+1), last_drhodt(2*N+1), previous_drhodt(2*N+1);
    auto set_init_state = [&](state_type& X) {
        X[0] = T0; 
        auto rhovecL = Eigen::Map<Eigen::ArrayXd>(&(X[1]), N);
        auto rhovecV = Eigen::Map<Eigen::ArrayXd>(&(X[1]) + N, N);
        rhovecL = rhovecL0;
        rhovecV = rhovecV0;
    };
    set_init_state(x0);

    // The function to be integrated by odeint
    auto xprime = [&](const state_type& X, state_type& Xprime, double /*t*/) {
        // Memory maps into the state vector for inputs and their derivatives
        // These are views, not copies!
        const double& T = X[0];
        auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(X[1]), N);
        auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(X[1]) + N, N);
        auto& dTdt = Xprime[0];
        auto drhovecdtL = Eigen::Map<Eigen::ArrayXd>(&(Xprime[1]), N);
        auto drhovecdtV = Eigen::Map<Eigen::ArrayXd>(&(Xprime[1]) + N, N);
        // Get the derivatives with respect to temperature along the isobar of the phase envelope
        auto [drhovecdTL, drhovecdTV] = get_drhovecdT_psat(model, T, rhovecL, rhovecV);
        // Get the derivative of T w.r.t. parameter
        dTdt = 1.0 / sqrt(norm(drhovecdTL.array()) + norm(drhovecdTV.array()));
        // And finally the derivatives with respect to the tracing variable
        drhovecdtL = c * drhovecdTL * dTdt;
        drhovecdtV = c * drhovecdTV * dTdt;

        if (previous_drhodt.empty()) {
            return;
        }

        // Flip the step if it changes direction from the smooth continuation of previous steps
        auto get_const_view = [&](const auto& v, int N) {
            return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), N);
        };
        if (get_const_view(Xprime, N).matrix().dot(get_const_view(previous_drhodt, N).matrix()) < 0) {
            auto Xprimeview = Eigen::Map<Eigen::ArrayXd>(&(Xprime[0]), 2 * N);
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
            double T = x0[0];
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[1]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[1]) + N, N);
            auto rhototL = rhovecL.sum(), rhototV = rhovecV.sum();
            using id = IsochoricDerivatives<decltype(model), Scalar, VecType>;
            double pL = rhototL * model.R(rhovecL / rhovecL.sum()) * T + id::get_pr(model, T, rhovecL);
            double pV = rhototV * model.R(rhovecV / rhovecV.sum()) * T + id::get_pr(model, T, rhovecV);

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
                {"xL_0 / mole frac.", rhovecL[0] / rhovecL.sum()},
                {"xV_0 / mole frac.", rhovecV[0] / rhovecV.sum()},
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

        //double dtold = dt;
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
            auto N = (x0.size()-1) / 2;
            auto& T = x0[0];
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[1]), N);
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[1]) + N, N);
            auto x = rhovecL / rhovecL.sum();
            auto y = rhovecV / rhovecV.sum();
            // Check if the solution has gone mechanically unstable
            if (opt.calc_criticality) {
                using ct = CriticalTracing<Model, Scalar, VecType>;
                auto condsL = ct::get_criticality_conditions(model, T, rhovecL);
                auto condsV = ct::get_criticality_conditions(model, T, rhovecV);
                if (condsL[0] < 1e-12 || condsV[0] < 1e-12) {
                    return true;
                }
            }
            if ((x < 0).any() || (x > 1).any() || (y < 0).any() || (y > 1).any() || (!rhovecL.isFinite()).any() || (!rhovecV.isFinite()).any()) {
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
            double T = x0[0];
            auto rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(x0[1]), N).eval();
            auto rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(x0[1 + N]), N).eval();
            auto x = (Eigen::ArrayXd(2) << rhovecL(0) / rhovecL.sum(), rhovecL(1) / rhovecL.sum()).finished(); // Mole fractions in the liquid phase (to be kept constant)
            auto [return_code, Tnew, rhovecLnew, rhovecVnew] = mixture_VLE_px(model, p, x, T, rhovecL, rhovecV);

            // If the step is accepted, copy into x again ...
            x0[0] = Tnew;
            auto rhovecLview = Eigen::Map<Eigen::ArrayXd>(&(x0[1]), N);
            auto rhovecVview = Eigen::Map<Eigen::ArrayXd>(&(x0[1]) + N, N);
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