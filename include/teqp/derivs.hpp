#pragma once

#include <complex>
#include <map>
#include <tuple>

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

#if defined(TEQP_MULTICOMPLEX_ENABLED)
#include "MultiComplex/MultiComplex.hpp"
#endif

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;

namespace teqp {

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
    Eigen::ArrayX<comtype> rhocom(rho.size());
    for (auto j = 0; j < rho.size(); ++j) {
        rhocom[j] = comtype(rho[j], 0.0);
    }
    rhocom[i] = comtype(rho[i], h);
    return f(T, rhocom).imag() / h;
}

/// Helper function for build_duplicated_tuple
/// See example here for the general concept https://en.cppreference.com/w/cpp/utility/integer_sequence
template<typename T, size_t ... I>
auto build_duplicated_tuple_impl(const T& val, std::index_sequence<I...>) {
    return std::make_tuple((static_cast<void>(I), val)...);  // The comma operator (a,b) evaluates the first argument and discards it, and keeps the second argument 
}

/// A function to generate a tuple of N repeated copies of argument val at compile-time
template<int N, typename T>
auto build_duplicated_tuple(const T& val) {
    return build_duplicated_tuple_impl(val, std::make_index_sequence<N>());
}

/// A class to help with the forwarding of arguments to wrt of autodiff.  std::apply cannot be applied to 
/// wrt directly because the wrt uses perfect forwarding and the compiler cannot work out argument types
/// but by making the struct with operator(), the compiler can figure out the argument types
struct wrt_helper {
    template<typename... Args>
    auto operator()(Args&&... args) const
    {
        return Wrt<Args&&...>{ std::forward_as_tuple(std::forward<Args>(args)...) };
    }
};

enum class AlphaWrapperOption {residual, idealgas};
/**
* \brief This class is used to wrap a model that exposes the generic 
* functions alphar, alphaig, etc., and allow the desired member function to be
* called at runtime via perfect forwarding
* 
* This class is needed because conventional binding methods (e.g., std::bind)
* require the argument types to be known, and they are not known in this case
* so we give the hard work of managing the argument types to the compiler
*/
template<AlphaWrapperOption o, class Model>
struct AlphaCallWrapper {
    const Model& m_model;
    AlphaCallWrapper(const Model& model) : m_model(model) {};

    template <typename ... Args>
    auto alpha(const Args& ... args) const {
        if constexpr (o == AlphaWrapperOption::residual) {
            // The alphar method is REQUIRED to be implemented by all
            // models, so can just call it via perfect fowarding
            return m_model.alphar(std::forward<const Args>(args)...);
        }
        else {
            //throw teqp::InvalidArgument("Missing implementation for alphaig");
            return m_model.alphaig(std::forward<const Args>(args)...);
        }
    }
};

enum class ADBackends { autodiff
#if defined(TEQP_MULTICOMPLEX_ENABLED)
    ,multicomplex
#endif
    ,complex_step
};

template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct TDXDerivatives {

    static auto get_Ar00(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return model.alphar(T, rho, molefrac);
    }

    /**
    * Calculate the derivative \f$\Lambda_{xy}\f$, where 
    * \f[
    * \Lambda^{\rm r}_{ij} = (1/T)^i\rho^j\left(\frac{\partial^{i+j}(\alpha^*)}{\partial(1/T)^i\partial\rho^j}\right)
    * \f]
    * 
    * Note: none of the intermediate derivatives are returned, although they are calculated
    */
    template<int iT, int iD, ADBackends be = ADBackends::autodiff, class AlphaWrapper>
    static auto get_Agenxy(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {

        static_assert(iT > 0 || iD > 0);
        if constexpr (iT == 0 && iD > 0) {
            if constexpr (be == ADBackends::autodiff) {
                // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
                autodiff::Real<iD, Scalar> rho_ = rho;
                auto f = [&w, &T, &molefrac](const auto& rho__) { return w.alpha(T, rho__, molefrac); };
                return powi(rho, iD)*derivatives(f, along(1), at(rho_))[iD];
            }
            else if constexpr (iD == 1 && be == ADBackends::complex_step) {
                double h = 1e-100;
                auto rho_ = std::complex<Scalar>(rho, h);
                return powi(rho, iD) * w.alpha(T, rho_, molefrac).imag() / h;
            }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
                fcn_t f = [&](const auto& rhomcx) { return w.alpha(T, rhomcx, molefrac); };
                auto ders = diff_mcx1(f, rho, iD, true /* and_val */);
                return powi(rho, iD)*ders[iD];
            }
#endif
            else {
                throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Agenxy for iT == 0");
            }
        }
        else if constexpr (iT > 0 && iD == 0) {
            auto Trecip = 1.0 / T;
            if constexpr (be == ADBackends::autodiff) {
                // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
                autodiff::Real<iT, Scalar> Trecipad = Trecip;
                auto f = [&w, &rho, &molefrac](const auto& Trecip__) {return w.alpha(1.0/Trecip__, rho, molefrac); };
                return powi(Trecip, iT)*derivatives(f, along(1), at(Trecipad))[iT];
            }
            else if constexpr (iT == 1 && be == ADBackends::complex_step) {
                double h = 1e-100;
                auto Trecipcsd = std::complex<Scalar>(Trecip, h);
                return powi(Trecip, iT)* w.alpha(1/Trecipcsd, rho, molefrac).imag()/h;
            }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
                fcn_t f = [&](const auto& Trecipmcx) { return w.alpha(1.0/Trecipmcx, rho, molefrac); };
                auto ders = diff_mcx1(f, Trecip, iT, true /* and_val */);
                return powi(Trecip, iT)*ders[iT];
            }
#endif
            else {
                throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Agenxy for iD == 0");
            }
        }
        else { // iT > 0 and iD > 0
            if constexpr (be == ADBackends::autodiff) {
                using adtype = autodiff::HigherOrderDual<iT + iD, double>;
                adtype Trecipad = 1.0 / T, rhoad = rho;
                auto f = [&w, &molefrac](const adtype& Trecip, const adtype& rho_) { return eval(w.alpha(eval(1.0/Trecip), rho_, molefrac)); };
                auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(Trecipad)), build_duplicated_tuple<iD>(std::ref(rhoad)));
                auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(Trecipad, rhoad));
                return powi(1.0 / T, iT) * powi(rho, iD) * der[der.size() - 1];
            }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function< mcx::MultiComplex<double>(const std::valarray<mcx::MultiComplex<double>>&)>;
                const fcn_t func = [&w, &molefrac](const auto& zs) {
                    auto Trecip = zs[0], rhomolar = zs[1];
                    return w.alpha(1.0 / Trecip, rhomolar, molefrac);
                };
                std::vector<double> xs = { 1.0 / T, rho};
                std::vector<int> order = { iT, iD };
                auto der = mcx::diff_mcxN(func, xs, order);
                return powi(1.0 / T, iT)*powi(rho, iD)*der;
            }
#endif
            else {
                throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Agenxy for iD > 0 and iT > 0");
            }
        }
        return static_cast<Scalar>(-999999999*T); // This will never hit, only to make compiler happy because it doesn't know the return type
    }

    /**
    * Calculate the derivative \f$\Lambda^{\rm r}_{xy}\f$, where
    * \f[
    * \Lambda^{\rm r}_{ij} = (1/T)^i\rho^j\left(\frac{\partial^{i+j}(\alpha^r)}{\partial(1/T)^i\partial\rho^j}\right)
    * \f]
    *
    * Note: none of the intermediate derivatives are returned, although they are calculated
    */
    template<int iT, int iD, ADBackends be = ADBackends::autodiff>
    static auto get_Arxy(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        auto wrapper = AlphaCallWrapper<AlphaWrapperOption::residual, decltype(model)>(model);
        if constexpr (iT == 0 && iD == 0) {
            return wrapper.alpha(T, rho, molefrac);
        }
        else {
            return get_Agenxy<iT, iD, be>(wrapper, T, rho, molefrac);
        }
    }

    /**
    * Calculate the derivative \f$\Lambda^{\rm ig}_{xy}\f$, where
    * \f[
    * \Lambda^{\rm ig}_{ij} = (1/T)^i\rho^j\left(\frac{\partial^{i+j}(\alpha^{\rm ig})}{\partial(1/T)^i\partial\rho^j}\right)
    * \f]
    *
    * Note: none of the intermediate derivatives are returned, although they are calculated
    */
    template<int iT, int iD, ADBackends be = ADBackends::autodiff>
    static auto get_Aigxy(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        auto wrapper = AlphaCallWrapper<AlphaWrapperOption::idealgas, decltype(model)>(model);
        if constexpr (iT == 0 && iD == 0) {
            return wrapper.alpha(T, rho, molefrac);
        }
        else {
            return get_Agenxy<iT, iD, be>(wrapper, T, rho, molefrac);
        }
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar10(const Model& model, const Scalar &T, const Scalar &rho, const VectorType& molefrac) {
        return get_Arxy<1, 0, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static Scalar get_Ar01(const Model& model, const Scalar&T, const Scalar &rho, const VectorType& molefrac){
        return get_Arxy<0, 1, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar02(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<0, 2, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar20(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<2, 0, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar12(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<1, 2, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar11(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<1, 1, be>(model, T, rho, molefrac);
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_Aig11(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Aigxy<1, 1, be>(model, T, rho, molefrac);
    }

    template<int Nderiv, ADBackends be = ADBackends::autodiff, class AlphaWrapper>
    static auto get_Agen0n(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        std::valarray<Scalar> o(Nderiv+1);
        if constexpr (be == ADBackends::autodiff) {
            // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
            autodiff::Real<Nderiv, Scalar> rho_ = rho;
            auto f = [&w, &T, &molefrac](const auto& rho__) { return w.alpha(T, rho__, molefrac); };
            auto ders = derivatives(f, along(1), at(rho_));
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = forceeval(powi(rho, n) * ders[n]);
            }
            return o;
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else {
            using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
            bool and_val = true;
            fcn_t f = [&w, &T, &molefrac](const auto& rhomcx) { return w.alpha(T, rhomcx, molefrac); };
            auto ders = diff_mcx1(f, rho, Nderiv, and_val);
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = powi(rho, n) * ders[n];
            }
            return o;
        }
#endif
        throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Ar0n");
    }
    
    template<int Nderiv, ADBackends be = ADBackends::autodiff, class AlphaWrapper>
    static auto get_Agenn0(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        std::valarray<Scalar> o(Nderiv+1);
        auto Trecip = 1.0 / T;
        if constexpr (be == ADBackends::autodiff) {
            // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
            autodiff::Real<Nderiv, Scalar> Trecipad = Trecip;
            auto f = [&w, &rho, &molefrac](const auto& Trecip__) {return w.alpha(1.0/Trecip__, rho, molefrac); };
            auto ders = derivatives(f, along(1), at(Trecipad));
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = powi(Trecip, n) * ders[n];
            }
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else if constexpr (be == ADBackends::multicomplex) {
            using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
            fcn_t f = [&](const auto& Trecipmcx) { return w.alpha(1.0/Trecipmcx, rho, molefrac); };
            auto ders = diff_mcx1(f, Trecip, Nderiv+1, true /* and_val */);
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = powi(Trecip, n) * ders[n];
            }
        }
#endif
        else {
            throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Agenn0");
        }
        return o;
    }
    
    /**
    * Calculate the derivative \f$\Lambda^{\rm r}_{x0}\f$, where
    * \f[
    * \Lambda^{\rm r}_{ij} = (1/T)^i\\left(\frac{\partial^{j}(\alpha^r)}{\partial(1/T)^i}\right)
    * \f]
    *
    * Note:The intermediate derivatives are returned
    */
    template<int iT, ADBackends be = ADBackends::autodiff>
    static auto get_Arn0(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        auto wrapper = AlphaCallWrapper<AlphaWrapperOption::residual, decltype(model)>(model);
        return get_Agenn0<iT, be>(wrapper, T, rho, molefrac);
    }
    
    /**
    * Calculate the derivative \f$\Lambda^{\rm ig}_{0y}\f$, where
    * \f[
    * \Lambda^{\rm ig}_{ij} = \rho^j\left(\frac{\partial^{j}(\alpha^{\rm ig})}{\partial\rho^j}\right)
    * \f]
    *
    * Note:The intermediate derivatives are returned
    */
    template<int iD, ADBackends be = ADBackends::autodiff>
    static auto get_Ar0n(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        auto wrapper = AlphaCallWrapper<AlphaWrapperOption::residual, decltype(model)>(model);
        return get_Agen0n<iD, be>(wrapper, T, rho, molefrac);
    }
    

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar(const int itau, const int idelta, const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        if (itau == 0) {
            if (idelta == 0) {
                return get_Ar00(model, T, rho, molefrac);
            }
            else if (idelta == 1) {
                return get_Ar01(model, T, rho, molefrac);
            }
            else if (idelta == 2) {
                return get_Ar02(model, T, rho, molefrac);
            }
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else if (itau == 1){
            if (idelta == 0) {
                return get_Ar10(model, T, rho, molefrac);
            }
            //else if (idelta == 1) {
            //    return get_Ar11(model, T, rho, molefrac);
            //}
            /*else if (idelta == 2) {
                return get_Ar12(model, T, rho, molefrac);
            }*/
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else if (itau == 2) {
            if (idelta == 0) {
                return get_Ar20(model, T, rho, molefrac);
            }
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else {
            throw std::invalid_argument("Invalid value for itau");
        }
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_neff(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        auto Ar01 = get_Ar01<be>(model, T, rho, molefrac);
        auto Ar11 = get_Ar11<be>(model, T, rho, molefrac);
        auto Ar20 = get_Ar20<be>(model, T, rho, molefrac);
        return -3*(Ar01-Ar11)/Ar20;
    }
};

template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct VirialDerivatives {

    static auto get_B2vir(const Model& model, const Scalar &T, const VectorType& molefrac) {
        Scalar h = 1e-100;
        // B_2 = dalphar/drho|T,z at rho=0
        auto B2 = model.alphar(T, std::complex<Scalar>(0.0, h), molefrac).imag()/h;
        return B2;
    }

    /**
    * \f$
    * B_n = \frac{1}{(n-2)!} \lim_{\rho\to 0} \left(\frac{\partial ^{n-1}\alpha^{\rm r}}{\partial \rho^{n-1}}\right)_{T,z}
    * \f$
    * \tparam Nderiv The maximum virial coefficient to return; e.g. 5: B_2, B_3, ..., B_5
    * \param model The model providing the alphar function
    * \param T Temperature
    * \param molefrac The mole fractions
    */

    template <int Nderiv, ADBackends be = ADBackends::autodiff>
    static auto get_Bnvir(const Model& model, const Scalar &T, const VectorType& molefrac) 
    {
        std::map<int, double> dnalphardrhon;
        if constexpr(be == ADBackends::autodiff){
            autodiff::HigherOrderDual<Nderiv, double> rhodual = 0.0;
            auto f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
            auto derivs = derivatives(f, wrt(rhodual), at(rhodual));
            for (auto n = 1; n < Nderiv; ++n){
                 dnalphardrhon[n] = derivs[n];
            }
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else if constexpr(be == ADBackends::multicomplex){
            using namespace mcx;
            using fcn_t = std::function<MultiComplex<double>(const MultiComplex<double>&)>;
            fcn_t f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
            auto derivs = diff_mcx1(f, 0.0, Nderiv, true /* and_val */);
            for (auto n = 1; n < Nderiv; ++n){
                dnalphardrhon[n] = derivs[n];
            }
        }
#endif
        else{
            //static_assert(false, "algorithmic differentiation backend is invalid");
            throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Bnvir");
        }
        
        std::map<int, Scalar> o;
        for (int n = 2; n <= Nderiv; ++n) {
            o[n] = dnalphardrhon[n-1];
            // 0! = 1, 1! = 1, so only n>3 terms need factorial correction
            if (n > 3) {
                auto factorial = [](int N) {return tgamma(N + 1); };
                o[n] /= factorial(n-2);
            }
        }
        return o;
    }

    /// This version of the get_Bnvir takes the maximum number of derivatives as a runtime argument
    /// and then forwards all arguments to the templated function
    template <ADBackends be = ADBackends::autodiff>
    static auto get_Bnvir_runtime(const int Nderiv, const Model& model, const Scalar &T, const VectorType& molefrac) {
        switch(Nderiv){
            case 2: return get_Bnvir<2,be>(model, T, molefrac);
            case 3: return get_Bnvir<3,be>(model, T, molefrac);
            case 4: return get_Bnvir<4,be>(model, T, molefrac);
            case 5: return get_Bnvir<5,be>(model, T, molefrac);
            case 6: return get_Bnvir<6,be>(model, T, molefrac);
            case 7: return get_Bnvir<7,be>(model, T, molefrac);
            case 8: return get_Bnvir<8,be>(model, T, molefrac);
            case 9: return get_Bnvir<9,be>(model, T, molefrac);
            default: throw std::invalid_argument("Only Nderiv up to 9 is supported, get_Bnvir templated function allows more");
        }
    }

    /**
    * \brief Temperature derivatives of a virial coefficient
    * 
    * \f$
    * \left(\frac{\partial^m{B_n}}{\partial T^m}\right) = \frac{1}{(n-2)!} \lim_{\rho\to 0} \left(\frac{\partial ^{(n-1)+m}\alpha^{\rm r}}{\partial T^m \partial \rho^{n-1}}\right)_{T, z}
    * \f$
    * \tparam Nderiv The virial coefficient to return; e.g. 5: B_5
    * \tparam NTderiv The number of temperature derivatives to calculate
    * \param model The model providing the alphar function
    * \param T Temperature
    * \param molefrac The mole fractions
    */
    template <int Nderiv, int NTderiv, ADBackends be = ADBackends::autodiff>
    static auto get_dmBnvirdTm(const Model& model, const Scalar& T, const VectorType& molefrac)
    {
        std::map<int, Scalar> o;
        auto factorial = [](int N) {return tgamma(N + 1); };
        if constexpr (be == ADBackends::autodiff) {
            autodiff::HigherOrderDual<NTderiv + Nderiv-1, double> rhodual = 0.0, Tdual = T;
            auto f = [&model, &molefrac](const auto& T_, const auto& rho_) { return model.alphar(T_, rho_, molefrac); };
            auto wrts = std::tuple_cat(build_duplicated_tuple<NTderiv>(std::ref(Tdual)), build_duplicated_tuple<Nderiv-1>(std::ref(rhodual)));
            auto derivs = derivatives(f, std::apply(wrt_helper(), wrts), at(Tdual, rhodual));
            return derivs.back() / factorial(Nderiv - 2);
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else if constexpr (be == ADBackends::multicomplex) {
            using namespace mcx;
            using fcn_t = std::function<MultiComplex<double>(const std::valarray<MultiComplex<double>>&)>;
            fcn_t f = [&model, &molefrac](const auto& zs) {
                auto T_ = zs[0], rho_ = zs[1];
                return model.alphar(T_, rho_, molefrac);
            };
            std::valarray<double> at = { T, 0.0 };
            auto deriv = diff_mcxN(f, at, { NTderiv, Nderiv-1});
            return deriv / factorial(Nderiv - 2);
        }
#endif
        else {
            //static_assert(false, "algorithmic differentiation backend is invalid");
            throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Bnvir");
        }
    }

    /// This version of the get_dmBnvirdTm takes the maximum number of derivatives as runtime arguments
    /// and then forwards all arguments to the templated function
    template <ADBackends be = ADBackends::autodiff>
    static auto get_dmBnvirdTm_runtime(const int Nderiv, const int NTderiv, const Model& model, const Scalar& T, const VectorType& molefrac) {
        if (Nderiv == 2) { // B_2
            switch (NTderiv) {
            case 0: return get_Bnvir<2, be>(model, T, molefrac)[2];
            case 1: return get_dmBnvirdTm<2, 1, be>(model, T, molefrac);
            case 2: return get_dmBnvirdTm<2, 2, be>(model, T, molefrac);
            case 3: return get_dmBnvirdTm<2, 3, be>(model, T, molefrac);
            default: throw std::invalid_argument("NTderiv is invalid in get_dmBnvirdTm_runtime");
            }
        }
        else if (Nderiv == 3) { // B_3
            switch (NTderiv) {
            case 0: return get_Bnvir<3, be>(model, T, molefrac)[3];
            case 1: return get_dmBnvirdTm<3, 1, be>(model, T, molefrac);
            case 2: return get_dmBnvirdTm<3, 2, be>(model, T, molefrac);
            case 3: return get_dmBnvirdTm<3, 3, be>(model, T, molefrac);
            default: throw std::invalid_argument("NTderiv is invalid in get_dmBnvirdTm_runtime");
            }
        }
        else if (Nderiv == 4) { // B_4
            switch (NTderiv) {
            case 0: return get_Bnvir<4, be>(model, T, molefrac)[4];
            case 1: return get_dmBnvirdTm<4, 1, be>(model, T, molefrac);
            case 2: return get_dmBnvirdTm<4, 2, be>(model, T, molefrac);
            case 3: return get_dmBnvirdTm<4, 3, be>(model, T, molefrac);
            default: throw std::invalid_argument("NTderiv is invalid in get_dmBnvirdTm_runtime");
            }
        }
        else {
            throw std::invalid_argument("Nderiv is invalid in get_dmBnvirdTm_runtime");
        }
    }
    
    /**
     * \brief Calculate the cross-virial coefficient \f$B_{12}\f$
     * \param model The model to use
     * \param T temperature
     * \param molefrac mole fractions
     */
    static auto get_B12vir(const Model& model, const Scalar &T, const VectorType& molefrac) {
        if (molefrac.size() != 2) { throw std::invalid_argument("length of mole fraction vector must be 2 in get_B12vir"); }
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
        auto rhotot_ = rhovec.sum();
        auto molefrac = (rhovec / rhotot_).eval();
        auto h = 1e-100;
        auto Ar01 = model.alphar(T, std::complex<double>(rhotot_, h), molefrac).imag() / h * rhotot_;
        return Ar01*rhotot_*model.R(molefrac)*T;
    }

    static auto get_Ar00(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        return model.alphar(T, rhotot, molefrac);
    }

    static auto get_Ar10(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        return -T * derivT([&model, &rhotot, &molefrac](const auto& T, const auto& rhovec) { return model.alphar(T, rhotot, molefrac); }, T, rhovec);
    }

    static auto get_Ar01(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot_ = std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        decltype(rhovec[0] * T) Ar01 = 0.0;
        for (auto i = 0; i < rhovec.size(); ++i) {
            auto Ar00 = [&model](const auto &T, const auto&rhovec) {
                auto rhotot = rhovec.sum();
                auto molefrac = rhovec / rhotot;
                return model.alphar(T, rhotot, molefrac);
            };
            Ar01 += rhovec[i] * derivrhoi(Ar00, T, rhovec, i);
        }
        return Ar01;
    }

    /***
    * \brief Calculate Psir=ar*rho
    */
    static auto get_Psir(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot_ = rhovec.sum();
        auto molefrac = rhovec / rhotot_;
        return model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_;
    }

    /***
    * \brief Calculate derivative Psir=ar*rho w.r.t. T at constant molar concentrations
    */
    static auto get_dPsirdT_constrhovec(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot_ = rhovec.sum();
        auto molefrac = rhovec / rhotot_;
        autodiff::Real<1, Scalar> Tad = T;
        auto f = [&model, &rhotot_, &molefrac](const auto& T_) {return rhotot_*model.R(molefrac)*T_*model.alphar(T_, rhotot_, molefrac); };
        return derivatives(f, along(1), at(Tad))[1];
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
        ArrayXdual2nd g;
        ArrayXdual2nd rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto hfunc = [&model, &T](const ArrayXdual2nd& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        return autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g).eval(); // evaluate the function value u, its gradient, and its Hessian matrix H
    }

    /***
    * \brief Calculate the function value, gradient, and Hessian of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses autodiff to calculate the derivatives
    */
    static auto build_Psir_fgradHessian_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        // Double derivatives in each component's concentration
        // N^N matrix (symmetric)

        dual2nd u; // the output scalar u = f(x), evaluated together with Hessian below
        ArrayXdual g;
        ArrayXdual2nd rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto hfunc = [&model, &T](const ArrayXdual2nd& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        // Evaluate the function value u, its gradient, and its Hessian matrix H
        Eigen::MatrixXd H = autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g); 
        // Remove autodiff stuff from the numerical values
        auto f = getbaseval(u);
        auto gg = g.cast<double>().eval();
        return std::make_tuple(f, gg, H);
    }

    /***
    * \brief Calculate the Hessian of Psi = a*rho w.r.t. the molar concentrations
    *
    * Uses autodiff derivatives to calculate second partial derivatives
    */
    static auto build_Psi_Hessian_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        auto rhotot_ = rho.sum();
        auto molefrac = (rho / rhotot_).eval();
        auto H = build_Psir_Hessian_autodiff(model, T, rho).eval();
        for (auto i = 0; i < 2; ++i) {
            H(i, i) += model.R(molefrac) * T / rho[i];
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
            return model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_;
        };
        using mattype = Eigen::ArrayXXd;
        auto H = get_Hessian<mattype, fcn_t, VectorType, HessianMethods::Multiple>(func, rho);
        return H;
    }

    /***
    * \brief Gradient of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses autodiff to calculate derivatives
    */
    static auto build_Psir_gradient_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        ArrayXdual rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto psirfunc = [&model, &T](const ArrayXdual& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        auto val = autodiff::gradient(psirfunc, wrt(rhovecc), at(rhovecc)).eval(); // evaluate the gradient
        return val;
    }

    /***
    * \brief Gradient of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses multicomplex to calculate derivatives
    */
    static auto build_Psir_gradient_multicomplex(const Model& model, const Scalar& T, const VectorType& rho) {
        using rho_type = typename VectorType::value_type;
        Eigen::ArrayX<mcx::MultiComplex<rho_type>> rhovecc(rho.size()); //for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto psirfunc = [&model](const auto &T, const auto& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return eval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        VectorType out(rho.size());
        for (int i = 0; i < rho.size(); ++i) {
            out[i] = derivrhoi(psirfunc, T, rho, i);
        }
        return out;
    }
    /***
    * \brief Gradient of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses complex step to calculate derivatives
    */
    static auto build_Psir_gradient_complex_step(const Model& model, const Scalar& T, const VectorType& rho) {
        using rho_type = typename VectorType::value_type;
        Eigen::ArrayX<std::complex<rho_type>> rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto psirfunc = [&model](const auto& T, const auto& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return forceeval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        VectorType out(rho.size());
        for (int i = 0; i < rho.size(); ++i) {
            auto rhocopy = rhovecc;
            rho_type h = 1e-100;
            rhocopy[i] = rhocopy[i] + std::complex<rho_type>(0,h);
            auto calc = psirfunc(T, rhocopy);
            out[i] = calc.imag / static_cast<double>(h);
        }
        return out;
    }

    /* Convenience function to select the correct implementation at compile-time */
    template<ADBackends be = ADBackends::autodiff>
    static auto build_Psir_gradient(const Model& model, const Scalar& T, const VectorType& rho) {
        if constexpr (be == ADBackends::autodiff) {
            return build_Psir_gradient_autodiff(model, T, rho);
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else if constexpr (be == ADBackends::multicomplex) {
            return build_Psir_gradient_multicomplex(model, T, rho);
        }
#endif
        else if constexpr (be == ADBackends::complex_step) {
            return build_Psir_gradient_complex_step(model, T, rho);
        }
    }

    /***
    * \brief Calculate the chemical potential of each component
    *
    * Uses autodiff to calculate derivatives
    * See Eq. 5 of https://doi.org/10.1002/aic.16730, but the rho in the denominator should be a rhoref (taken to be 1)
    * \note: Some contributions to the ideal gas part are missing (reference state and cp0), but are not relevant to phase equilibria
    */
    static auto get_chempotVLE_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        typename VectorType::value_type rhotot = rho.sum();
        auto molefrac = (rho / rhotot).eval();
        auto rhorefideal = 1.0;
        return (build_Psir_gradient_autodiff(model, T, rho).array() + model.R(molefrac)*T*(rhorefideal + log(rho / rhorefideal))).eval();
    }

    /***
    * \brief Calculate the fugacity coefficient of each component
    *
    * Uses autodiff to calculate derivatives
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_fugacity_coefficients(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Z = 1.0 + tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto grad = build_Psir_gradient<be>(model, T, rhovec).eval();
        auto RT = R * T;
        auto lnphi = ((grad / RT).array() - log(Z)).eval();
        return exp(lnphi).eval();
    }

    static auto build_d2PsirdTdrhoi_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        Eigen::ArrayXd deriv(rho.size());
        // d^2psir/dTdrho_i
        for (auto i = 0; i < rho.size(); ++i) {
            auto psirfunc = [&model, &rho, i](const auto& T, const auto& rhoi) {
                ArrayXdual2nd rhovecc(rho.size()); for (auto j = 0; j < rho.size(); ++j) { rhovecc[j] = rho[j]; }
                rhovecc[i] = rhoi;
                auto rhotot_ = rhovecc.sum();
                auto molefrac = (rhovecc / rhotot_).eval();
                return eval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
            };
            dual2nd Tdual = T, rhoidual = rho[i];
            auto [u00, u10, u11] = derivatives(psirfunc, wrt(Tdual, rhoidual), at(Tdual, rhoidual));
            deriv[i] = u11;
        }
        return deriv;
    }

    /***
    * \brief Calculate the temperature derivative of the chemical potential of each component
    * \note: Some contributions to the ideal gas part are missing (reference state and cp0), but are not relevant to phase equilibria
    */
    static auto get_dchempotdT_autodiff(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto rhorefideal = 1.0;
        return (build_d2PsirdTdrhoi_autodiff(model, T, rhovec) + model.R(molefrac)*(rhorefideal + log(rhovec/rhorefideal))).eval();
    }

    /***
    * \brief Calculate the temperature derivative of the pressure at constant molar concentrations
    */
    static auto get_dpdT_constrhovec(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto dPsirdT = get_dPsirdT_constrhovec(model, T, rhovec);
        return rhotot*model.R(molefrac) - dPsirdT + rhovec.matrix().dot(build_d2PsirdTdrhoi_autodiff(model, T, rhovec).matrix());
    }

    /***
    * \brief Calculate the molar concentration derivatives of the pressure at constant temperature
    */
    static auto get_dpdrhovec_constT(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto RT = model.R(molefrac)*T;
        auto [func, grad, hessian] = build_Psir_fgradHessian_autodiff(model, T, rhovec); // The hessian matrix
        return (RT + (hessian*rhovec.matrix()).array()).eval(); // at constant temperature
    }

    /***
    * \brief Calculate the partial molar volumes of each component
    * 
    * \f[
    * \hat v_i = \left(\frac{\partial V}{\partial n_i}\right)_{T,V,n_{j \neq i}}
    * \f]
    */
    static auto get_partial_molar_volumes(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto RT = model.R(molefrac)*T;

        auto dpdrhovec = get_dpdrhovec_constT(model, T, rhovec);
        auto numerator = -rhotot*dpdrhovec;
        
        auto ders = tdx::template get_Ar0n<2>(model, T, rhotot, molefrac);
        auto denominator = -pow2(rhotot)*RT*(1 + 2*ders[1] + ders[2]);
        return (numerator/denominator).eval();
    }
};

template<int Nderivsmax, AlphaWrapperOption opt>
class DerivativeHolderSquare{
    
public:
    Eigen::Array<double, Nderivsmax+1, Nderivsmax+1> derivs;
    
    template<typename Model, typename Scalar, typename VecType>
    DerivativeHolderSquare(const Model& model, const Scalar& T, const Scalar& rho, const VecType& z) {
        using tdx = TDXDerivatives<decltype(model), Scalar, VecType>;
        static_assert(Nderivsmax == 2, "It's gotta be 2 for now");
        AlphaCallWrapper<opt, Model> wrapper(model);
        
        auto AX02 = tdx::template get_Agen0n<2>(wrapper, T, rho, z);
        derivs(0, 0) = AX02[0];
        derivs(0, 1) = AX02[1];
        derivs(0, 2) = AX02[2];
        
        auto AX20 = tdx::template get_Agenn0<2>(wrapper, T, rho, z);
        derivs(0, 0) = AX20[0];
        derivs(1, 0) = AX20[1];
        derivs(2, 0) = AX20[2];
        
        derivs(1, 1) = tdx::template get_Agenxy<1,1>(wrapper, T, rho, z);
    }
};



}; // namespace teqp
