#pragma once

#include <optional>
#include <complex>
#include <tuple>
#include <map>

#include "MultiComplex/MultiComplex.hpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

template<typename T>
auto forceeval(T&& expr)
{
    using namespace autodiff::detail;
    if constexpr (isDual<T> || isExpr<T> || isNumber<T>) {
        return eval(expr);
    }
    else {
        return expr;
    }
}

/***
* \brief Given a function, use complex step derivatives to calculate the derivative with 
* respect to the first variable which here is temperature
*/
template <typename TType, typename ContainerType, typename FuncType>
typename ContainerType::value_type derivT(const FuncType& f, TType T, const ContainerType& rho) {
    double h = 1e-100;
    return f(std::complex<TType>(T, h), rho).imag() / h;
}

/***
* \brief Given a function, use multicomplex derivatives to calculate the derivative with
* respect to the first variable which here is temperature
*/
template <typename TType, typename ContainerType, typename FuncType>
typename ContainerType::value_type derivTmcx(const FuncType& f, TType T, const ContainerType& rho) {
    using fcn_t = std::function<mcx::MultiComplex<double>(const mcx::MultiComplex<double>&)>;
    fcn_t wrapper = [&rho, &f](const auto& T_) {return f(T_, rho); };
    auto ders = diff_mcx1(wrapper, T, 1);
    return ders[0];
}

/***
* \brief Given a function, use complex step derivatives to calculate the derivative with respect 
* to the given composition variable
*/
template <typename TType, typename ContainerType, typename FuncType, typename Integer>
typename ContainerType::value_type derivrhoi(const FuncType& f, TType T, const ContainerType& rho, Integer i) {
    double h = 1e-100;
    using comtype = std::complex<typename ContainerType::value_type>;
    std::valarray<comtype> rhocom(rho.size());
    for (auto j = 0; j < rho.size(); ++j) {
        rhocom[j] = comtype(rho[j], 0.0);
    }
    rhocom[i] = comtype(rho[i], h);
    return f(T, rhocom).imag() / h;
}





template <typename Model, typename TType, typename RhoType, typename ContainerType>
typename ContainerType::value_type get_Ar10(const Model& model, const TType T, const RhoType &rho, const ContainerType& molefrac) {
    double h = 1e-100;
    return -T*model.alphar(std::complex<TType>(T, h), rho, molefrac).imag()/h; // Complex step derivative
}

enum class ADBackends { autodiff, multicomplex, complex_step } ;

template <ADBackends be = ADBackends::autodiff, typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar01(const Model& model, const TType &T, const RhoType &rho, const MoleFracType& molefrac) {
    if constexpr(be == ADBackends::complex_step){
        double h = 1e-100;
        auto der = model.alphar(T, std::complex<double>(rho, h), molefrac).imag() / h;
        return der*rho;
    }
    else if constexpr(be == ADBackends::multicomplex){
        using fcn_t = std::function<mcx::MultiComplex<double>(const mcx::MultiComplex<double>&)>;
        bool and_val = true;
        fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
        auto ders = diff_mcx1(f, rho, 1, and_val);
        return ders[1] * rho;
    }
    else if constexpr(be == ADBackends::autodiff){
        autodiff::dual rhodual = rho;
        auto f = [&model, &T, &molefrac](const auto& rho_) { return eval(model.alphar(T, rho_, molefrac)); };
        auto der = derivative(f, wrt(rhodual), at(rhodual));
        return der * rho;
    }
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar02(const Model& model, const TType& T, const RhoType& rho, const MoleFracType& molefrac) {
    autodiff::dual2nd rhodual = rho;
    auto f = [&model, &T, &molefrac](const auto& rho_) { return eval(model.alphar(T, rho_, molefrac)); };
    auto ders = derivatives(f, wrt(rhodual), at(rhodual));
    return ders[2]*rho*rho;
}


template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct VirialDerivatives {

    static auto get_B2vir(const Model& model, const Scalar &T, const VectorType& molefrac) {
        double h = 1e-100;
        // B_2 = dalphar/drho|T,z at rho=0
        auto B2 = model.alphar(T, std::complex<double>(0.0, h), molefrac).imag()/h;
        return B2;
    }

    /**
    * \f$
    * B_n = \frac{1}{(n-2)!} lim_rho\to 0 d^{n-1}alphar/drho^{n-1}|T,z
    * \f$
    * \param model The model providing the alphar function
    * \param Nderiv The maximum virial coefficient to return; e.g. 5: B_2, B_3, ..., B_5
    * \param T Temperature
    * \param molefrac The mole fractions
    */

    template <int Nderiv, ADBackends be = ADBackends::autodiff>
    static auto get_Bnvir(const Model& model, const Scalar &T, const VectorType& molefrac) 
    {
        std::map<int, double> dnalphardrhon;
        if constexpr(be == ADBackends::multicomplex){
            using namespace mcx;
            using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
            fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
            auto derivs = diff_mcx1(f, 0.0, Nderiv+1, true /* and_val */);
            for (auto n = 1; n <= Nderiv; ++n){
                dnalphardrhon[n] = derivs[n];
            }
        }
        else if constexpr(be == ADBackends::autodiff){
            autodiff::HigherOrderDual<Nderiv+1, double> rhodual = 0.0;
            auto f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
            auto derivs = derivatives(f, wrt(rhodual), at(rhodual));
            for (auto n = 1; n <= Nderiv; ++n){
                 dnalphardrhon[n] = derivs[n];
            }
        }
        else{
            static_assert("algorithmic differentiation backend is invalid");
        }
        std::map<int, Scalar> o;
        for (int n = 2; n < Nderiv+1; ++n) {
            o[n] = dnalphardrhon[n-1];
            // 0!=1, 1!=1, so only n>3 terms need factorial correction
            if (n > 3) {
                auto factorial = [](int N) {return tgamma(N + 1); };
                o[n] /= factorial(n-2);
            }
        }
        return o;
    }

    static auto get_B12vir(const Model& model, const Scalar &T, const VectorType& molefrac) {
    
        auto B2 = get_B2vir(model, T, molefrac); // Overall B2 for mixture
        const auto xpure0 = (Eigen::ArrayXd(2) << 1,0).finished();
        const auto xpure1 = (Eigen::ArrayXd(2) << 0,1).finished();
        auto B20 = get_B2vir(model, T, xpure0); // Pure first component with index 0
        auto B21 = get_B2vir(model, T, xpure1); // Pure second component with index 1
        auto z0 = molefrac[0];
        auto B12 = (B2 - z0*z0*B20 - (1-z0)*(1-z0)*B21)/(2*z0*(1-z0));
        return B12;
    }
};


template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct IsochoricDerivatives{

    /***
    * \brief Calculate the residual entropy (s^+ = -sr/R) from derivatives of alphar
    */
    static auto get_splus(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = rhovec / rhotot;
        return model.alphar(T, rhotot, molefrac) - get_Ar10(model, T, rhovec);
    }

    /***
    * \brief Calculate the residual pressure from derivatives of alphar
    */
    static auto get_pr(const Model& model, const Scalar &T, const VectorType& rhovec)
    {
        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        return ::get_Ar01(model, T, rhotot_, rhovec / rhotot_) * rhotot_ * model.R * T;
    }

    static auto get_Ar00(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = rhovec / rhotot;
        return model.alphar(T, rhotot, molefrac);
    }

    static auto get_Ar10(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = rhovec / rhotot;
        return -T * derivT([&model, &rhotot, &molefrac](const auto& T, const auto& rhovec) { return model.alphar(T, rhotot, molefrac); }, T, rhovec);
    }

    static auto get_Ar01(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        decltype(rhovec[0] * T) Ar01 = 0.0;
        for (auto i = 0; i < rhovec.size(); ++i) {
            Ar01 += rhovec[i] * derivrhoi([&model](const auto& T, const auto& rhovec) { return model.alphar(T, rhovec); }, T, rhovec, i);
        }
        return Ar01;
    }

    /***
    * \brief Calculate Psir=ar*rho
    */
    static auto get_Psir(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        return model.alphar(T, rhotot_, rhovec / rhotot_) * model.R * T * rhotot_;
    }

    /***
    * \brief Calculate the Hessian of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Requires the use of autodiff derivatives to calculate second partial derivatives
    */
    static auto build_Psir_Hessian_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        // Double derivatives in each component's concentration
        // N^N matrix (symmetric)

        dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
        VectorXdual2nd g;
        VectorXdual2nd rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto hfunc = [&model, &T](const VectorXdual2nd& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R * T * rhotot_);
        };
        return autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g).eval(); // evaluate the function value u, its gradient, and its Hessian matrix H
    }

    /***
    * \brief Calculate the Hessian of Psi = a*rho w.r.t. the molar concentrations
    *
    * Uses autodiff derivatives to calculate second partial derivatives
    */
    static auto build_Psi_Hessian_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        auto H = build_Psir_Hessian_autodiff(model, T, rho);
        for (auto i = 0; i < 2; ++i) {
            H(i, i) += model.R * T / rho[i];
        }
        return H;
    }

    /***
    * \brief Calculate the Hessian of Psir = ar*rho w.r.t. the molar concentrations (residual contribution only)
    *
    * Requires the use of multicomplex derivatives to calculate second partial derivatives
    */
    static auto build_Psir_Hessian_mcx(const Model& model, const Scalar& T, const VectorType& rho) {
        // Double derivatives in each component's concentration
        // N^N matrix (symmetric)
        using namespace mcx;

        // Lambda function for getting Psir with multicomplex concentrations
        using fcn_t = std::function< MultiComplex<double>(const Eigen::ArrayX<MultiComplex<double>>&)>;
        fcn_t func = [&model, &T](const auto& rhovec) {
            auto rhotot_ = rhovec.sum();
            auto molefrac = (rhovec / rhotot_).eval();
            return model.alphar(T, rhotot_, molefrac) * model.R * T * rhotot_;
        };
        using mattype = Eigen::ArrayXXd;
        auto H = get_Hessian<mattype, fcn_t, VectorType, HessianMethods::Multiple>(func, rho);
        return H;
    }

    /***
    * \brief Gradient of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses autodiff to calculate second partial derivatives
    */
    static auto build_Psir_gradient_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        VectorXdual2nd rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto psirfunc = [&model, &T](const VectorXdual2nd& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R * T * rhotot_);
        };
        auto val = autodiff::gradient(psirfunc, wrt(rhovecc), at(rhovecc)).eval(); // evaluate the gradient
        return val;
    }

    /***
    * \brief Calculate the chemical potential of each component
    *
    * Uses autodiff derivatives to calculate second partial derivatives
    * See Eq. 9 of https://doi.org/10.1002/aic.16730
    * \note: Some contributions to the ideal gas part are missing (reference state and cp0), but are not relevant to phase equilibria
    */
    static auto get_chempot_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        typename VectorType::value_type rhotot = rho.sum();
        return (build_Psir_gradient_autodiff(model, T, rho).array() + model.R*T*(1.0 + log(rho / rhotot))).eval();
    }
};