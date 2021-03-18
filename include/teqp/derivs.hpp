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

/***
* \brief Calculate the Psir=ar*rho
*/
template <typename TType, typename ContainerType, typename Model>
typename ContainerType::value_type get_Psir(const Model& model, const TType T, const ContainerType& rhovec) {
    using container = decltype(rhovec);
    auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
    return model.alphar(T, rhovec)*model.R*T*rhotot_;
}

/***
* \brief Calculate the residual pressure from derivatives of alphar
*/
template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_pr(const Model& model, const TType T, const ContainerType& rhovec)
{
    auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
    decltype(rhovec[0]*T) pr = 0.0;
    for (auto i = 0; i < rhovec.size(); ++i) {
        pr += rhovec[i]*derivrhoi([&model](const auto& T, const auto& rhovec){ return model.alphar(T, rhovec); }, T, rhovec, i);
    }
    return pr*rhotot_*model.R*T;
}

template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_Ar10(const Model& model, const TType T, const ContainerType& rhovec){
    return -T*derivT([&model](const auto& T, const auto& rhovec) { return model.alphar(T, rhovec); }, T, rhovec);
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

/***
* \brief Calculate the residual entropy (s^+ = -sr/R) from derivatives of alphar
*/
template <typename Model, typename TType, typename ContainerType>
typename ContainerType::value_type get_splus(const Model& model, const TType T, const ContainerType& rhovec){
    return model.alphar(T, rhovec) - get_Ar10(model, T, rhovec);
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
auto build_Psir_Hessian_autodiff(const Model& model, const TType T, const RhoType& rho) {
    // Double derivatives in each component's concentration
    // N^N matrix (symmetric)

    dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
    VectorXdual2nd g;
    VectorXdual2nd rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
    auto hfunc = [&model, &T](const VectorXdual2nd& rho_) {
        auto rhotot_ = std::accumulate(std::begin(rho_), std::end(rho_), (decltype(rho_[0]))0.0);
        return eval(model.alphar(T, rho_) * model.R * T * rhotot_);
    };
    return autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g); // evaluate the function value u, its gradient, and its Hessian matrix H
}

/***
* \brief Calculate the Hessian of Psir = ar*rho w.r.t. the molar concentrations
* 
* Requires the use of multicomplex derivatives to calculate second partial derivatives
*/
template<typename Model, typename TType, typename RhoType>
auto build_Psir_Hessian_mcx(const Model& model, const TType T, const RhoType& rho) {
    // Double derivatives in each component's concentration
    // N^N matrix (symmetric)

    // Lambda function for getting Psir with multicomplex concentrations
    auto func = [&model, &T](const std::vector<MultiComplex<double>>& rhovec) {
        auto N = rhovec.size();
        std::valarray<MultiComplex<double>> xs(N); for (auto i = 0; i < N; ++i) { xs[i] = rhovec[i]; }
        return get_Psir(model, T, xs);
    };
    // The set of values around which the pertubations will happen
    const std::size_t N = rho.size();
    std::vector<double> xs(N); for(auto i = 0; i < N; ++i){ xs[i] = rho[i]; }

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