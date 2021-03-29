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

template <typename TType, typename ContainerType, typename FuncType>
typename std::enable_if<is_container<ContainerType>::value, typename ContainerType::value_type>::type
caller(const FuncType& f, TType T, const ContainerType& rho) {
    return f(T, rho);
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
    using comtype = std::complex<ContainerType::value_type>;
    std::valarray<comtype> rhocom(rho.size());
    for (auto j = 0; j < rho.size(); ++j) {
        rhocom[j] = comtype(rho[j], 0.0);
    }
    rhocom[i] = comtype(rho[i], h);
    return f(T, rhocom).imag() / h;
}

template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_Ar10(const Model& model, const TType T, const ContainerType& rhovec){
    auto rhotot = rhovec.sum();
    auto molefrac = rhovec / rhotot;
    return -T*derivT([&model, &rhotot, &molefrac](const auto& T, const auto& rhovec) { return model.alphar(T, rhotot, molefrac); }, T, rhovec);
}

template <typename Model, typename TType, typename RhoType, typename ContainerType>
typename ContainerType::value_type get_Ar10(const Model& model, const TType T, const RhoType &rho, const ContainerType& molefrac) {
    return -T * derivT([&model, &rho, &molefrac](const auto& T, const auto& rhovec) { return model.alphar(T, rho, molefrac); }, T, rhovec);
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar01(const Model& model, const TType &T, const RhoType &rho, const MoleFracType& molefrac) {
    double h = 1e-100;
    auto der = model.alphar(T, std::complex<double>(rho, h), molefrac).imag() / h;
    return der*rho;
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar01mcx(const Model& model, const TType& T, const RhoType& rho, const MoleFracType& molefrac) {
    using fcn_t = std::function<mcx::MultiComplex<double>(const mcx::MultiComplex<double>&)>;
    bool and_val = true;
    fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
    auto ders = diff_mcx1(f, rho, 1, and_val);
    return ders[1] * rho;
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar01ad(const Model& model, const TType& T, const RhoType& rho, const MoleFracType& molefrac) {
    autodiff::dual rhodual = rho;
    auto f = [&model, &T, &molefrac](const auto& rho_) { return eval(model.alphar(T, rho_, molefrac)); };
    auto der = derivative(f, wrt(rhodual), at(rhodual));
    return der * rho;
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar02(const Model& model, const TType& T, const RhoType& rho, const MoleFracType& molefrac) {
    using fcn_t = std::function<mcx::MultiComplex<double>(const mcx::MultiComplex<double>&)>;
    bool and_val = true;
    fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
    auto ders = diff_mcx1(f, rho, 2, and_val);
    return ders[2]*rho*rho;
}

template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_Ar01(const Model& model, const TType T, const ContainerType& rhovec) {
    auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
    decltype(rhovec[0] * T) Ar01 = 0.0;
    for (auto i = 0; i < rhovec.size(); ++i) {
        Ar01 += rhovec[i] * derivrhoi([&model](const auto& T, const auto& rhovec) { return model.alphar(T, rhovec); }, T, rhovec, i);
    }
    return Ar01;
}

template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_B2vir(const Model& model, const TType T, const ContainerType& molefrac) {
    double h = 1e-100;
    // B_2 = dalphar/drho|T,z at rho=0
    auto B2 = model.alphar(T, std::complex<double>(0.0, h), molefrac).imag()/h;
    return B2;
}

/*
* \f$
* B_n = \frac{1}{(n-2)!} lim_rho\to 0 d^{n-1}alphar/drho^{n-1}|T,z
* \f$
* \param model The model providing the alphar function
* \param Nderiv The maximum virial coefficient to return; e.g. 5: B_2, B_3, ..., B_5
* \param T Temperature
* \param molefrac The mole fractions
*/

template <typename Model, typename TType, typename ContainerType>
auto get_Bnvir(const Model& model, int Nderiv, const TType T, const ContainerType& molefrac) {
    
    using namespace mcx;
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
    std::map<int, TType> o;
    auto dalphardrhon = diff_mcx1(f, 0.0, Nderiv+1, true /* and_val */);
    for (int n = 2; n < Nderiv+1; ++n) {
        o[n] = dalphardrhon[n-1];
        // 0!=1, 1!=1, so only n>3 terms need factorial correction
        if (n > 3) {
            auto factorial = [](int N) {return tgamma(N + 1); };
            o[n] /= factorial(n-2);
        }
    }
    return o;
}

/***
* \brief Calculate the residual entropy (s^+ = -sr/R) from derivatives of alphar
*/
template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_splus(const Model& model, const TType T, const ContainerType& rhovec){
    auto rhotot = rhovec.sum();
    auto molefrac = rhovec/rhotot;
    return model.alphar(T, rhotot, molefrac) - get_Ar10(model, T, rhovec);
}

/***
* \brief Calculate Psir=ar*rho
*/
template <typename TType, typename ContainerType, typename Model>
typename ContainerType::value_type get_Psir(const Model& model, const TType T, const ContainerType& rhovec) {
    auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
    return model.alphar(T, rhotot_, rhovec / rhotot_) * model.R * T * rhotot_;
}

/***
* \brief Calculate the residual pressure from derivatives of alphar
*/
template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_pr(const Model& model, const TType T, const ContainerType& rhovec)
{
    auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
    return get_Ar01(model, T, rhotot_, rhovec / rhotot_) * rhotot_ * model.R * T;
}



/***
* \brief Calculate the Hessian of Psir = ar*rho w.r.t. the molar concentrations
*
* Requires the use of autodiff derivatives to calculate second partial derivatives
*/
template<typename Model, typename TType, typename RhoType>
auto build_Psir_Hessian_autodiff(const Model& model, const TType &T, const RhoType& rho) {
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
* \brief Calculate the Hessian of Psir = ar*rho w.r.t. the molar concentrations
* 
* Requires the use of multicomplex derivatives to calculate second partial derivatives
*/
template<typename Model, typename TType, typename RhoType>
auto build_Psir_Hessian_mcx(const Model& model, const TType &T, const RhoType& rho) {
    // Double derivatives in each component's concentration
    // N^N matrix (symmetric)
    using namespace mcx;

    // Lambda function for getting Psir with multicomplex concentrations
    using fcn_t = std::function< MultiComplex<double>(const std::valarray<MultiComplex<double>>&)>;
    fcn_t func = [&model, &T](const auto& rhovec) {
        return get_Psir(model, T, rhovec);
    };
    using mattype = Eigen::ArrayXXd;
    auto H = get_Hessian<mattype, fcn_t, std::valarray<double>, HessianMethods::Multiple>(func, rho);
    return H;
}