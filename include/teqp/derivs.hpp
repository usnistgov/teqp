#pragma once

#include <optional>
#include <complex>
#include <tuple>

#include "MultiComplex/MultiComplex.hpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

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
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    fcn_t wrapper = [&rho, &f](const MultiComplex<TType>& T_) {return f(T_, rho); };
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
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    bool and_val = true;
    fcn_t f = [&model, &T, &molefrac](const MultiComplex<double>& rho_) -> MultiComplex<double> { return model.alphar(T, rho_, molefrac); };
    auto ders = diff_mcx1(f, rho, 1, and_val);
    return ders[1] * rho;
}

template <typename Model, typename TType, typename RhoType, typename MoleFracType>
auto get_Ar02(const Model& model, const TType& T, const RhoType& rho, const MoleFracType& molefrac) {
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    bool and_val = true;
    fcn_t f = [&model, &T, &molefrac](const MultiComplex<double>& rho_) -> MultiComplex<double> { return model.alphar(T, rho_, molefrac); };
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
    // B_2 = lim_rho\to 0 dalphar/drho|T,z
    auto B2 = model.alphar(T, std::complex<double>(0, h), molefrac).imag()/h;
    return B2;
}

template <typename Model, typename TType, typename ContainerType>
auto get_Bnvir(const Model& model, int Nderiv, const TType T, const ContainerType& molefrac) {
    // B_n = lim_rho\to 0 d^{n-1}alphar/drho^{n-1}|T,z
    using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
    fcn_t f = [&model, &T, &molefrac](const MultiComplex<double>& rho_) -> MultiComplex<double> { return model.alphar(T, rho_, molefrac); };
    return diff_mcx1(f, 0.0, Nderiv, true /* and_val */);
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

// Generic setting functions to handle Eigen types and STL types with the same interface
template<typename MatrixLike, typename Integer, typename ValType>
void setval(MatrixLike &m, Integer i, Integer j, const ValType val) {
    m(i,j) = val;
}

// Partial specialization for valarray "matrix"
template <> void setval<std::valarray<std::valarray<double>>, std::size_t, double>(std::valarray<std::valarray<double>>& m, std::size_t i, std::size_t j, const double val) {
    m[i][j] = val;
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
        auto molefrac = rho_ / rhotot_;
        return eval(model.alphar(T, rhotot_, molefrac) * model.R * T * rhotot_);
    };
    return autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g); // evaluate the function value u, its gradient, and its Hessian matrix H
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

    // Lambda function for getting Psir with multicomplex concentrations
    auto func = [&model, &T](const std::vector<MultiComplex<double>>& rhovec) {
        std::valarray<MultiComplex<double>> xs(&(rhovec[0]), rhovec.size());
        return get_Psir(model, T, xs);
    };
    // The set of values around which the pertubations will happen
    const std::size_t N = rho.size();
    std::vector<double> xs(std::begin(rho), std::end(rho));

    Eigen::MatrixXd H(N, N);
    
    for (std::size_t i = 0; i < rho.size(); ++i) {
        for (std::size_t j = i; j < rho.size(); ++j) {
            std::vector<int> order = { 0, 0 };
            order[i] += 1;
            order[j] += 1;
            auto val = diff_mcxN<double>(func, xs, order);
            setval(H,i,j,val);
            setval(H,j,i,val);
        }
    }
    return H;
}