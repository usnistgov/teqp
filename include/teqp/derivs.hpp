#pragma once

#if defined(TEQP_COMPLEXSTEP_ENABLED)
#include <complex>
#endif
#include <map>
#include <tuple>
#include <numeric>
#include <concepts>

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

/**
* \brief Given a function, use complex step derivatives to calculate the derivative with 
* respect to the first variable which here is temperature
*/
template <typename TType, typename ContainerType, typename FuncType>
typename ContainerType::value_type derivT(const FuncType& f, TType T, const ContainerType& rho) {
    double h = 1e-100;
    return f(std::complex<TType>(T, h), rho).imag() / h;
}

#if defined(TEQP_MULTICOMPLEX_ENABLED)
/**
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
#endif

/**
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

enum class ADBackends { autodiff
#if defined(TEQP_MULTICOMPLEX_ENABLED)
    ,multicomplex
#endif
#if defined(TEQP_COMPLEXSTEP_ENABLED)
    ,complex_step
#endif
};

template<typename T, typename U, typename V, typename W>
concept CallableAlpha = requires(T t, U u, V v, W w) {
    { t.alpha(u,v,w) };
};

template<typename T, typename U, typename V, typename W>
concept CallableAlphar = requires(T t, U u, V v, W w) {
    { t.alphar(u,v,w) };
};

template<typename T, typename U, typename V, typename W>
concept CallableAlpharTauDelta = requires(T t, U u, V v, W w) {
    { t.alphar_taudelta(u,v,w) };
};

template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct TDXDerivatives {
    
    static auto get_Ar00(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return model.alphar(T, rho, molefrac);
    }
    
    template<typename AlphaWrapper, typename S1, typename S2, typename Vec>
    static auto AlphaCaller(const AlphaWrapper& w, const S1& T, const S2& rho, const Vec& molefrac) requires CallableAlpha<AlphaWrapper, S1, S2, Vec>{
        return w.alpha(T, rho, molefrac);
    }
    template<typename AlphaWrapper, typename S1, typename S2, typename Vec>
    static auto AlphaCaller(const AlphaWrapper& w, const S1& T, const S2& rho, const Vec& molefrac) requires CallableAlphar<AlphaWrapper, S1, S2, Vec>{
        return w.alphar(T, rho, molefrac);
    }
    
    template<typename AlphaWrapper, typename S1, typename S2, typename Vec>
    static auto AlpharTauDeltaCaller(const AlphaWrapper& w, const S1& T, const S2& rho, const Vec& molefrac) requires CallableAlpharTauDelta<AlphaWrapper, S1, S2, Vec>{
        return w.alphar_taudelta(T, rho, molefrac);
    }
    template<typename AlphaWrapper, typename S1, typename S2, typename Vec>
    static auto AlpharTauDeltaCaller(const AlphaWrapper& , const S1&, const S2&, const Vec& molefrac){
        throw teqp::NotImplementedError("Cannot take derivatives of a class that doesn't define the alphar_taudelta method");
        return std::common_type_t<S1, S2, decltype(molefrac[0])>(1e99);
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
        
        if constexpr (iT == 0 && iD == 0){
            return AlphaCaller(w, T, rho, molefrac);
        }
        else if constexpr (iT == 0 && iD > 0) {
            if constexpr (be == ADBackends::autodiff) {
                // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
                autodiff::Real<iD, Scalar> rho_ = rho;
                auto f = [&w, &T, &molefrac](const auto& rho__) { return AlphaCaller(w, T, rho__, molefrac); };
                return powi(rho, iD)*derivatives(f, along(1), at(rho_))[iD];
            }
#if defined(TEQP_COMPLEXSTEP_ENABLED)
            else if constexpr (iD == 1 && be == ADBackends::complex_step) {
                double h = 1e-100;
                auto rho_ = std::complex<Scalar>(rho, h);
                return powi(rho, iD) * AlphaCaller(w, T, rho_, molefrac).imag() / h;
            }
#endif
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
                fcn_t f = [&](const auto& rhomcx) { return AlphaCaller(w, T, rhomcx, molefrac); };
                auto ders = diff_mcx1(f, rho, iD, true /* and_val */);
                return powi(rho, iD)*ders[iD];
            }
#endif
            else {
                throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Agenxy for iT == 0");
            }
        }
        else if constexpr (iT > 0 && iD == 0) {
            Scalar Trecip = 1.0 / T;
            if constexpr (be == ADBackends::autodiff) {
                // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
                autodiff::Real<iT, Scalar> Trecipad = Trecip;
                auto f = [&w, &rho, &molefrac](const auto& Trecip__) {return AlphaCaller(w, forceeval(1.0/Trecip__), rho, molefrac); };
                return powi(Trecip, iT)*derivatives(f, along(1), at(Trecipad))[iT];
            }
#if defined(TEQP_COMPLEXSTEP_ENABLED)
            else if constexpr (iT == 1 && be == ADBackends::complex_step) {
                double h = 1e-100;
                auto Trecipcsd = std::complex<Scalar>(Trecip, h);
                return powi(Trecip, iT)* w.alpha(1/Trecipcsd, rho, molefrac).imag()/h;
            }
#endif
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
                fcn_t f = [&](const auto& Trecipmcx) { return AlphaCaller(w, 1.0/Trecipmcx, rho, molefrac); };
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
                auto f = [&w, &molefrac](const adtype& Trecip, const adtype& rho_) {
                    adtype T_ = 1.0/Trecip;
                    return forceeval(AlphaCaller(w, T_, rho_, molefrac)); };
                auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(Trecipad)), build_duplicated_tuple<iD>(std::ref(rhoad)));
                auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(Trecipad, rhoad));
                return powi(forceeval(1.0 / T), iT) * powi(rho, iD) * der[der.size() - 1];
            }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
            else if constexpr (be == ADBackends::multicomplex) {
                using fcn_t = std::function< mcx::MultiComplex<double>(const std::valarray<mcx::MultiComplex<double>>&)>;
                const fcn_t func = [&w, &molefrac](const auto& zs) {
                    auto Trecip = zs[0], rhomolar = zs[1];
                    return AlphaCaller(w, 1.0 / Trecip, rhomolar, molefrac);
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
        //        return static_cast<Scalar>(-999999999*T); // This will never hit, only to make compiler happy because it doesn't know the return type
    }
    
    /**
     Calculate the derivative
     \f[
     \Lambda_{xyz_i} = (1/T)^x(\rho)^y\deriv{^{x+y+z_i}(\alpha^r)}{(1/T)^x\partial \rho^y \partial \mathbf{Z}_i^{z_i}}}{}
     \f]
     in which all the compositions are treated as being independent
     */
    template<int iT, int iD, int iXi, typename AlphaWrapper>
    static auto get_ATrhoXi(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac, int i){
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi, double>;
        adtype Trecipad = 1.0 / T, rhoad = rho, xi = molefrac[i];
        auto f = [&w, &molefrac, &i](const adtype& Trecip, const adtype& rho_, const adtype& xi_) {
            adtype T_ = 1.0/Trecip;
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            return forceeval(AlphaCaller(w, T_, rho_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(Trecipad)), build_duplicated_tuple<iD>(std::ref(rhoad)), build_duplicated_tuple<iXi>(std::ref(xi)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(Trecipad, rhoad, xi));
        return powi(forceeval(1.0 / T), iT) * powi(rho, iD) * der[der.size() - 1];
    }
    
    #define get_ATrhoXi_runtime_combinations \
        X(0,0,1) \
        X(0,0,2) \
        X(0,0,3) \
        X(1,0,0) \
        X(1,0,1) \
        X(1,0,2) \
        X(1,0,3) \
        X(0,1,0) \
        X(0,1,1) \
        X(0,1,2) \
        X(0,1,3) \
        X(2,0,0) \
        X(2,0,1) \
        X(2,0,2) \
        X(2,0,3) \
        X(1,1,0) \
        X(1,1,1) \
        X(1,1,2) \
        X(1,1,3) \
        X(0,2,0) \
        X(0,2,1) \
        X(0,2,2) \
        X(0,2,3)
    
    template<typename AlphaWrapper>
    static auto get_ATrhoXi_runtime(const AlphaWrapper& w, const Scalar& T, int iT, const Scalar& rho, int iD, const VectorType& molefrac, int i, int iXi){
        #define X(a,b,c) if (iT == a && iD == b && iXi == c) { return get_ATrhoXi<a,b,c>(w, T, rho, molefrac, i); }
        get_ATrhoXi_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }
    
    #define get_ATrhoXiXj_runtime_combinations \
        X(0,0,1,0) \
        X(0,0,2,0) \
        X(0,0,0,1) \
        X(0,0,0,2) \
        X(0,0,1,1) \
        X(1,0,1,0) \
        X(1,0,2,0) \
        X(1,0,0,1) \
        X(1,0,0,2) \
        X(1,0,1,1) \
        X(0,1,1,0) \
        X(0,1,2,0) \
        X(0,1,0,1) \
        X(0,1,0,2) \
        X(0,1,1,1)

    template<typename AlphaWrapper>
    static auto get_ATrhoXiXj_runtime(const AlphaWrapper& w, const Scalar& T, int iT, const Scalar& rho, int iD, const VectorType& molefrac, int i, int iXi, int j, int iXj){
        #define X(a,b,c,d) if (iT == a && iD == b && iXi == c && iXj == d) { return get_ATrhoXiXj<a,b,c,d>(w, T, rho, molefrac, i, j); }
        get_ATrhoXiXj_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }
    
    #define get_ATrhoXiXjXk_runtime_combinations \
        X(0,0,0,1,1) \
        X(0,0,1,0,1) \
        X(0,0,1,1,0) \
        X(0,0,1,1,1) \
        X(1,0,0,1,1) \
        X(1,0,1,0,1) \
        X(1,0,1,1,0) \
        X(1,0,1,1,1) \
        X(0,1,0,1,1) \
        X(0,1,1,0,1) \
        X(0,1,1,1,0) \
        X(0,1,1,1,1)

    template<typename AlphaWrapper>
    static auto get_ATrhoXiXjXk_runtime(const AlphaWrapper& w, const Scalar& T, int iT, const Scalar& rho, int iD, const VectorType& molefrac, int i, int iXi, int j, int iXj, int k, int iXk){
        #define X(a,b,c,d,e) if (iT == a && iD == b && iXi == c && iXj == d && iXk == e) { return get_ATrhoXiXjXk<a,b,c,d,e>(w, T, rho, molefrac, i, j, k); }
        get_ATrhoXiXjXk_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }
    
    template<int iT, int iD, int iXi, typename AlphaWrapper>
    static auto get_AtaudeltaXi(const AlphaWrapper& w, const Scalar& tau, const Scalar& delta, const VectorType& molefrac, const int i) {
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi, double>;
        adtype tauad = tau, deltaad = delta, xi = molefrac[i];
        auto f = [&w, &molefrac, &i](const adtype& tau_, const adtype& delta_, const adtype& xi_) {
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            return forceeval(AlpharTauDeltaCaller(w, tau_, delta_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(tauad)), build_duplicated_tuple<iD>(std::ref(deltaad)), build_duplicated_tuple<iXi>(std::ref(xi)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(tauad, deltaad, xi));
        return powi(tau, iT) * powi(delta, iD) * der[der.size() - 1];
    }
    
    template<int iT, int iD, int iXi, int iXj, typename AlphaWrapper>
    static auto get_AtaudeltaXiXj(const AlphaWrapper& w, const Scalar& tau, const Scalar& delta, const VectorType& molefrac, const int i, const int j) {
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi + iXj, double>;
        if (i == j){
            throw teqp::InvalidArgument("i cannot equal j");
        }
        adtype tauad = tau, deltaad = delta, xi = molefrac[i], xj = molefrac[j];
        auto f = [&w, &molefrac, i, j](const adtype& tau_, const adtype& delta_, const adtype& xi_, const adtype& xj_) {
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            molefracdual[j] = xj_;
            return forceeval(AlpharTauDeltaCaller(w, tau_, delta_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(tauad)), build_duplicated_tuple<iD>(std::ref(deltaad)), build_duplicated_tuple<iXi>(std::ref(xi)), build_duplicated_tuple<iXj>(std::ref(xj)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(tauad, deltaad, xi, xj));
        return powi(tau, iT) * powi(delta, iD) * der[der.size() - 1];
    }
    
    template<int iT, int iD, int iXi, int iXj, int iXk, typename AlphaWrapper>
    static auto get_AtaudeltaXiXjXk(const AlphaWrapper& w, const Scalar& tau, const Scalar& delta, const VectorType& molefrac, const int i, const int j, const int k) {
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi + iXj + iXk, double>;
        if (i == j || j == k || i == k){
            throw teqp::InvalidArgument("i, j, and k must all be unique");
        }
        adtype tauad = tau, deltaad = delta, xi = molefrac[i], xj = molefrac[j], xk = molefrac[k];
        auto f = [&w, &molefrac, i, j, k](const adtype& tau_, const adtype& delta_, const adtype& xi_, const adtype& xj_, const adtype& xk_) {
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            molefracdual[j] = xj_;
            molefracdual[k] = xk_;
            return forceeval(AlpharTauDeltaCaller(w, tau_, delta_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(tauad)), build_duplicated_tuple<iD>(std::ref(deltaad)), build_duplicated_tuple<iXi>(std::ref(xi)), build_duplicated_tuple<iXj>(std::ref(xj)), build_duplicated_tuple<iXk>(std::ref(xk)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(tauad, deltaad, xi, xj, xk));
        return powi(tau, iT) * powi(delta, iD) * der[der.size() - 1];
    }

    template<typename AlphaWrapper>
    static auto get_AtaudeltaXi_runtime(const AlphaWrapper& w, const Scalar& tau, const int iT, const Scalar& delta, const int iD, const VectorType& molefrac, const int i, const int iXi){
        if (iT == 0 && iD == 0 && iXi == 0){
            return AlpharTauDeltaCaller(w, tau, delta, molefrac);
        }
        #define X(a,b,c) if (iT == a && iD == b && iXi == c) { return get_AtaudeltaXi<a,b,c>(w, tau, delta, molefrac, i); }
        get_ATrhoXi_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }

    template<typename AlphaWrapper>
    static auto get_AtaudeltaXiXj_runtime(const AlphaWrapper& w, const Scalar& tau, const int iT, const Scalar& delta, const int iD, const VectorType& molefrac, const int i, const int iXi, const int j, const int iXj){
        #define X(a,b,c,d) if (iT == a && iD == b && iXi == c && iXj == d) { return get_AtaudeltaXiXj<a,b,c,d>(w, tau, delta, molefrac, i, j); }
        get_ATrhoXiXj_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }

    template<typename AlphaWrapper>
    static auto get_AtaudeltaXiXjXk_runtime(const AlphaWrapper& w, const Scalar& tau, const int iT, const Scalar& delta, const int iD, const VectorType& molefrac, const int i, int iXi, const int j, const int iXj, const int k, const int iXk){
        #define X(a,b,c,d,e) if (iT == a && iD == b && iXi == c && iXj == d && iXk == e) { return get_AtaudeltaXiXjXk<a,b,c,d,e>(w, tau, delta, molefrac, i, j, k); }
        get_ATrhoXiXjXk_runtime_combinations
        #undef X
        throw teqp::InvalidArgument("Can't match these derivative counts");
    }
    
    /**
     Calculate the derivative
     \f[
     \Lambda_{xyz_i z_j } = (1/T)^x(\rho)^y\left(\frac{\partial^{x+y+z_i+z_j}(\alpha^r)}{\partial (1/T)^x\partial \rho^y \partial \mathbf{Z}_i^{z_i} \partial \mathbf{Z}_j^{z_j}  \}\right)
     \f]
     in which all the compositions are treated as being independent
     */
    template<int iT, int iD, int iXi, int iXj, typename AlphaWrapper>
    static auto get_ATrhoXiXj(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac, int i, int j){
        if (i == j){
            throw teqp::InvalidArgument("i cannot equal j");
        }
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi + iXj, double>;
        adtype Trecipad = 1.0 / T, rhoad = rho, xi = molefrac[i], xj = molefrac[j];
        auto f = [&w, &molefrac, i, j](const adtype& Trecip, const adtype& rho_, const adtype& xi_, const adtype& xj_) {
            adtype T_ = 1.0/Trecip;
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            molefracdual[j] = xj_;
            return forceeval(AlphaCaller(w, T_, rho_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(Trecipad)), build_duplicated_tuple<iD>(std::ref(rhoad)), build_duplicated_tuple<iXi>(std::ref(xi)), build_duplicated_tuple<iXj>(std::ref(xj)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(Trecipad, rhoad, xi, xj));
        return powi(forceeval(1.0 / T), iT) * powi(rho, iD) * der[der.size() - 1];
    }
    
    /**
     Calculate the derivative
     \f[
     \Lambda_{xyz_i z_j z_k} = (1/T)^x(\rho)^y\left(\frac{\partial^{x+y+z_i+z_j+z_k}(\alpha^r)}{\partial (1/T)^x\partial \rho^y \partial \mathbf{Z}_i^{z_i} \partial \mathbf{Z}_j^{z_j} \partial \mathbf{Z}_k^{z_k}   \}\right
     \f]
     in which all the compositions are treated as being independent
     */
    template<int iT, int iD, int iXi, int iXj, int iXk, typename AlphaWrapper>
    static auto get_ATrhoXiXjXk(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac, int i, int j, int k){
        if (i == j || j == k || i == k){
            throw teqp::InvalidArgument("i, j, and k must all be unique");
        }
        using adtype = autodiff::HigherOrderDual<iT + iD + iXi + iXj + iXk, double>;
        adtype Trecipad = 1.0 / T, rhoad = rho, xi = molefrac[i], xj = molefrac[j], xk = molefrac[k];
        auto f = [&w, &molefrac, i, j, k](const adtype& Trecip, const adtype& rho_, const adtype& xi_, const adtype& xj_, const adtype& xk_) {
            adtype T_ = 1.0/Trecip;
            Eigen::ArrayX<adtype> molefracdual = molefrac.template cast<adtype>();
            molefracdual[i] = xi_;
            molefracdual[j] = xj_;
            molefracdual[k] = xk_;
            return forceeval(AlphaCaller(w, T_, rho_, molefracdual)); };
        auto wrts = std::tuple_cat(build_duplicated_tuple<iT>(std::ref(Trecipad)), build_duplicated_tuple<iD>(std::ref(rhoad)), build_duplicated_tuple<iXi>(std::ref(xi)), build_duplicated_tuple<iXj>(std::ref(xj)), build_duplicated_tuple<iXk>(std::ref(xk)));
        auto der = derivatives(f, std::apply(wrt_helper(), wrts), at(Trecipad, rhoad, xi, xj, xk));
        return powi(forceeval(1.0 / T), iT) * powi(rho, iD) * der[der.size() - 1];
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
        return get_Agenxy<iT, iD, be>(model, T, rho, molefrac);
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
        return get_Agenxy<iT, iD, be>(model, T, rho, molefrac);
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
    static auto get_Ar03(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<0, 3, be>(model, T, rho, molefrac);
    }

    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar20(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<2, 0, be>(model, T, rho, molefrac);
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar30(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<3, 0, be>(model, T, rho, molefrac);
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_Ar21(const Model& model, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        return get_Arxy<2, 1, be>(model, T, rho, molefrac);
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
            auto f = [&w, &T, &molefrac](const auto& rho__) { return AlphaCaller(w, T, rho__, molefrac); };
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
            fcn_t f = [&w, &T, &molefrac](const auto& rhomcx) { return AlphaCaller(w, T, rhomcx, molefrac); };
            auto ders = diff_mcx1(f, rho, Nderiv, and_val);
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = powi(rho, n) * ders[n];
            }
            return o;
        }
#endif
//        throw std::invalid_argument("algorithmic differentiation backend is invalid in get_Ar0n");
    }
    
    template<int Nderiv, ADBackends be = ADBackends::autodiff, class AlphaWrapper>
    static auto get_Agenn0(const AlphaWrapper& w, const Scalar& T, const Scalar& rho, const VectorType& molefrac) {
        std::valarray<Scalar> o(Nderiv+1);
        Scalar Trecip = 1.0 / T;
        if constexpr (be == ADBackends::autodiff) {
            // If a pure derivative, then we can use autodiff::Real for that variable and Scalar for other variable
            autodiff::Real<Nderiv, Scalar> Trecipad = Trecip;
            auto f = [&w, &rho, &molefrac](const auto& Trecip__) {return AlphaCaller(w, forceeval(1.0/Trecip__), rho, molefrac); };
            auto ders = derivatives(f, along(1), at(Trecipad));
            for (auto n = 0; n <= Nderiv; ++n) {
                o[n] = powi(Trecip, n) * ders[n];
            }
        }
#if defined(TEQP_MULTICOMPLEX_ENABLED)
        else if constexpr (be == ADBackends::multicomplex) {
            using fcn_t = std::function<mcx::MultiComplex<Scalar>(const mcx::MultiComplex<Scalar>&)>;
            fcn_t f = [&](const auto& Trecipmcx) { return AlphaCaller(w, 1.0/Trecipmcx, rho, molefrac); };
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
        return get_Agenn0<iT, be>(model, T, rho, molefrac);
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
        return get_Agen0n<iD, be>(model, T, rho, molefrac);
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
            else if (idelta == 3) {
                return get_Ar03(model, T, rho, molefrac);
            }
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else if (itau == 1){
            if (idelta == 0) {
                return get_Ar10(model, T, rho, molefrac);
            }
            else if (idelta == 1) {
                return get_Ar11(model, T, rho, molefrac);
            }
            else if (idelta == 2) {
                return get_Ar12(model, T, rho, molefrac);
            }
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else if (itau == 2) {
            if (idelta == 0) {
                return get_Ar20(model, T, rho, molefrac);
            }
            else if (idelta == 1) {
                return get_Ar21(model, T, rho, molefrac);
            }
            else {
                throw std::invalid_argument("Invalid value for idelta");
            }
        }
        else if (itau == 3) {
            if (idelta == 0) {
                return get_Ar30(model, T, rho, molefrac);
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
            auto f = [&model, &T, &molefrac](const auto& rho_) { return model.alphar(T, rho_, molefrac); };
            autodiff::Real<Nderiv, Scalar> rhoreal = 0.0;
            auto derivs = derivatives(f, along(1), at(rhoreal));
            
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
    /// and then forwards all arguments to the corresponding templated function
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

/**
 In the isochoric formalism, the fugacity coefficient array can be obtained by the gradient of the residual Helmholtz energy density (which is a scalar) and the compressibility factor \f$Z\f$  (which is also a scalar) in terms of the temperature \f$T\f$ and the molar concentration vector \f$\vec\rho\f$:
 \f[
     \ln\vec\phi = \frac{1}{RT}\frac{\partial \Psi^r}{d\vec\rho} - \ln(Z)
 \f]

 Easy: temperature derivative at constant molar concentrations (implying constant volume and molar composition)
 \f[
     \deriv{ \ln\vec\phi}{T}{\vec\rho} = \frac{1}{RT}\frac{\partial^2 \Psi^r}{\partial \vec\rho\partial T} + \frac{-1}{RT^2}\deriv{\Psi^r}{\vec\rho}{T} - \frac{1}{Z}\deriv{Z}{T}{\vec\rho}
 \f]

 Medium: molar density derivative at constant temperature and mole fractions
 \f[
     \deriv{ \ln\vec\phi}{\rho}{T,\vec x} = \frac{1}{RT}\frac{\partial^2 \Psi^r}{\partial \vec\rho\partial \rho}  - \frac{1}{Z}\deriv{Z}{\rho}{T,\vec x}
 \f]
 \f[
     Z = 1+\rho\deriv{\alpha^r}{\rho}{T}
 \f]
 \f[
 \deriv{Z}{\rho}{T,\vec x} = \rho\deriv{^2\alpha^r}{\rho^2}{T} + \deriv{\alpha^r}{\rho}{T}
 \f]

 Back to basics, for a quantity \f$\chi\f$ that is a function of \f$T\f$ and \f$\vec\rho\f$, and then the derivative taken w.r.t. density at constant temperature and mole fractions:
 \f[
     \deriv{\chi}{\rho}{T, \vec x} =     \deriv{\chi}{T}{\vec \rho}\cancelto{0}{\deriv{T}{\rho}{T}} + \sum_i\deriv{\chi}{\rho_i}{T, \rho_{j\neq i}}\deriv{\rho_i}{\rho}{T,\vec x}
 \f]
 with \f$\rho_i =x_i\rho\f$
 \f[
 \deriv{\rho_i}{\rho}{T, \vec x} = x_i
 \f]
 thus
 \f[
     \deriv{\chi}{\rho}{T, \vec x} =  \sum_i\deriv{\chi}{\rho_i}{T, \rho_{j\neq i}}x_i
 \f]

 and following the pattern yields
 \f[
 \frac{\partial^2 \Psi^r}{\partial \vec\rho\partial \rho} =     \sum_i\deriv{\frac{\partial \Psi^r}{d\vec\rho} }{\rho_i}{T, \rho_{j\neq i}}x_i
 \f]
 where the big thing is the Hessian of the residual Hessian matrix of the residual Helmholtz energy density.  This uses terms that are already developed.

 Medium+: Volume derivative, based on the density derivative

 \f[
 \deriv{ \ln\vec\phi}{v}{T,\vec x} = \deriv{ \ln\vec\phi}{\rho}{T,\vec x}\deriv{ \rho}{v}{}
 \f]
 \f[
 \deriv{\rho}{v}{} = -1/v^2 = -\rho^2
 \f]

 Hard: mole fraction derivatives (this results in a matrix rather than a vector)
 \f[
     \deriv{ \ln\vec\phi}{\vec x}{T,\rho} = ?
 \f]

 The first term is conceptually tricky. Again, considering a generic quantity \f$\chi\f$
 \f[
     \deriv{\chi}{x_i}{T, \rho,  x_{j\neq i}} = \deriv{\chi}{T}{\vec \rho}\cancelto{0}{\deriv{T}{x_i}{T,\rho,x_{j\neq i}}} + \sum_i\deriv{\chi}{\rho_i}{T, \rho_{j\neq i}}\deriv{\rho_i}{x_i}{T,\rho, x_{j\neq i}}
 \f]
 yields
 \f[
     \deriv{\chi}{x_i}{T, \rho,  x_{j\neq i}} = \rho \sum_i\deriv{\chi}{\rho_i}{T, \rho_{j\neq i}}
 \f]
 so the first part becomes
 \f[
     \deriv{\frac{\partial \Psi^r}{d\vec\rho}}{x_i}{T, \rho,  x_{j\neq i}} = \rho \sum_i\deriv{\frac{\partial \Psi^r}{d\vec\rho}}{\rho_i}{T, \rho_{j\neq i}}
 \f]
 or
 \f[
     \deriv{^2\partial \Psi^r}{\vec\rho \partial \vec x}{T, \rho} = \rho H(\Psi^r)
 \f]
 which is somewhat surprising because the order of derivatives with respect to composition and density doesn't matter, as the Hessian is symmetric

 The second part, from derivatives of \f$\ln Z\f$, with \f$Z\f$ given by
 \f[
     Z = 1+\rho\deriv{\alpha^r}{\rho}{T, \vec x}
 \f]
 yields
 \f[
 \deriv{\ln Z}{x_i}{T,\rho,x_{k \neq j}} = \frac{1}{Z}\deriv{Z}{x_i}{T,\rho,x_{k \neq i}}
 \f]
 which results in a vector because you have
 \f[
 \deriv{Z}{x_i}{T,\rho,x_{k \neq i}} = \rho \deriv{^2\alpha^r}{\rho\partial x_i}{T}
 \f]
 
 */
template<typename Model, typename Scalar = double, typename VectorType = Eigen::ArrayXd>
struct IsochoricDerivatives{

    /**
    * \brief Calculate the residual entropy (\f$s^+ = -s^r/R\f$) from derivatives of alphar
    */
    static auto get_splus(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = rhovec / rhotot;
        return model.alphar(T, rhotot, molefrac) - get_Ar10(model, T, rhovec);
    }

    /**
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
        return -T * derivT([&model, &rhotot, &molefrac](const auto& T, const auto& /*rhovec*/) { return model.alphar(T, rhotot, molefrac); }, T, rhovec);
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

    /**
    * \brief Calculate \f$\Psi^r=a^r \rho\f$
    */
    static auto get_Psir(const Model& model, const Scalar &T, const VectorType& rhovec) {
        auto rhotot_ = rhovec.sum();
        auto molefrac = rhovec / rhotot_;
        return model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_;
    }

    /**
    * \brief Calculate derivative \f$\Psi^r=a^r \rho\f$ w.r.t. T at constant molar concentrations
    */
    static auto get_dPsirdT_constrhovec(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot_ = rhovec.sum();
        auto molefrac = (rhovec / rhotot_).eval();
        autodiff::Real<1, Scalar> Tad = T;
        auto f = [&model, &rhotot_, &molefrac](const auto& T_) {return rhotot_*model.R(molefrac)*T_*model.alphar(T_, rhotot_, molefrac); };
        return derivatives(f, along(1), at(Tad))[1];
    }

    /**
    * \brief Calculate the Hessian of \f$\Psi^r = a^r \rho\f$ w.r.t. the molar concentrations
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
            return forceeval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        return autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g).eval(); // evaluate the function value u, its gradient, and its Hessian matrix H
    }

    /**
    * \brief Calculate the function value, gradient, and Hessian of \f$Psi^r = a^r\rho\f$ w.r.t. the molar concentrations
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
            return forceeval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        // Evaluate the function value u, its gradient, and its Hessian matrix H
        Eigen::MatrixXd H = autodiff::hessian(hfunc, wrt(rhovecc), at(rhovecc), u, g); 
        // Remove autodiff stuff from the numerical values
        auto f = getbaseval(u);
        auto gg = g.cast<double>().eval();
        return std::make_tuple(f, gg, H);
    }

    /**
    * \brief Calculate the Hessian of \f$\Psi = a \rho\f$ w.r.t. the molar concentrations
    *
    * Uses autodiff derivatives to calculate second partial derivatives
    */
    static auto build_Psi_Hessian_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        auto rhotot_ = rho.sum();
        auto molefrac = (rho / rhotot_).eval();
        auto H = build_Psir_Hessian_autodiff(model, T, rho).eval();
        for (auto i = 0; i < rho.size(); ++i) {
            H(i, i) += model.R(molefrac) * T / rho[i];
        }
        return H;
    }

#if defined(TEQP_MULTICOMPLEX_ENABLED)
    /**
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
#endif

    /**
    * \brief Gradient of Psir = ar*rho w.r.t. the molar concentrations
    *
    * Uses autodiff to calculate derivatives
    */
    static auto build_Psir_gradient_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        ArrayXdual rhovecc(rho.size()); for (auto i = 0; i < rho.size(); ++i) { rhovecc[i] = rho[i]; }
        auto psirfunc = [&model, &T](const ArrayXdual& rho_) {
            auto rhotot_ = rho_.sum();
            auto molefrac = (rho_ / rhotot_).eval();
            return forceeval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
        };
        auto val = autodiff::gradient(psirfunc, wrt(rhovecc), at(rhovecc)).eval(); // evaluate the gradient
        return val;
    }

#if defined(TEQP_MULTICOMPLEX_ENABLED)
    /**
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
#endif
    /**
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
#if defined(TEQP_COMPLEXSTEP_ENABLED)
        else if constexpr (be == ADBackends::complex_step) {
            return build_Psir_gradient_complex_step(model, T, rho);
        }
#endif
    }

    /**
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

    /**
    * \brief Calculate the fugacity coefficient of each component
    *
    * Uses autodiff to calculate derivatives
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_fugacity_coefficients(const Model& model, const Scalar& T, const VectorType& rhovec) {
        VectorType lnphi = get_ln_fugacity_coefficients<be>(model, T, rhovec);
        return exp(lnphi).eval();
    }
    
    /**
    * \brief Calculate the natural logarithm of fugacity coefficient of each component
    *
    * Uses autodiff to calculate derivatives by default
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_ln_fugacity_coefficients(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Z = 1.0 + tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto grad = build_Psir_gradient<be>(model, T, rhovec).eval();
        auto RT = R * T;
        auto lnphi = ((grad / RT).array() - log(Z)).eval();
        return forceeval(lnphi.eval());
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_ln_fugacity_coefficients_Trhomolefracs(const Model& model, const Scalar& T, const Scalar& rhotot, const VectorType& molefrac) {
        auto R = model.R(molefrac);
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Z = 1.0 + tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto rhovec = (rhotot*molefrac).eval();
        auto grad = build_Psir_gradient<be>(model, T, rhovec).eval();
        auto RT = R * T;
        auto lnphi = ((grad / RT).array() - log(Z)).eval();
        return forceeval(lnphi.eval());
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_ln_fugacity_coefficients1(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        auto grad = build_Psir_gradient<be>(model, T, rhovec).eval();
        auto RT = R * T;
        auto lnphi = ((grad / RT).array()).eval();
        return forceeval(lnphi);
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_ln_fugacity_coefficients2(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Z = 1.0 + tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        return forceeval(-log(Z));
    }
    
    static Eigen::ArrayXd build_d2PsirdTdrhoi_autodiff(const Model& model, const Scalar& T, const VectorType& rho) {
        Eigen::ArrayXd deriv(rho.size());
        // d^2psir/dTdrho_i
        for (auto i = 0; i < rho.size(); ++i) {
            auto psirfunc = [&model, &rho, i](const auto& T, const auto& rhoi) {
                ArrayXdual2nd rhovecc(rho.size()); for (auto j = 0; j < rho.size(); ++j) { rhovecc[j] = rho[j]; }
                rhovecc[i] = rhoi;
                auto rhotot_ = rhovecc.sum();
                auto molefrac = (rhovecc / rhotot_).eval();
                return forceeval(model.alphar(T, rhotot_, molefrac) * model.R(molefrac) * T * rhotot_);
            };
            dual2nd Tdual = T, rhoidual = rho[i];
            auto [u00, u10, u11] = derivatives(psirfunc, wrt(Tdual, rhoidual), at(Tdual, rhoidual));
            deriv[i] = u11;
        }
        return deriv;
    }
    
    static Eigen::ArrayXd build_d2alphardrhodxi_constT(const Model& model, const Scalar& T, const Scalar& rhomolar, const VectorType& molefrac) {
        Eigen::ArrayXd deriv(molefrac.size());
        // d^2alphar/drhodx_i|T
        for (auto i = 0; i < molefrac.size(); ++i) {
            auto alpharfunc = [&model, &T, &molefrac, i](const auto& rho, const auto& xi) {
                ArrayXdual2nd molefracdual = molefrac.template cast<autodiff::dual2nd>();
                molefracdual[i] = xi;
                return forceeval(model.alphar(T, rho, molefracdual));
            };
            dual2nd rhodual = rhomolar, xidual = molefrac[i];
            auto [u00, u10, u11] = derivatives(alpharfunc, wrt(rhodual, xidual), at(rhodual, xidual));
            deriv[i] = u11;
        }
        return deriv;
    }
    
    /**
    * \brief Calculate the derivative of the natural logarithm of fugacity coefficient of each component w.r.t. temperature at constant mole concentrations (implying constant mole fractions and density)
    *
    * Uses autodiff to calculate derivatives by default
    * \f[
    * \deriv{ \ln\vec\phi}{T}{\vec \rho}
    * \f]
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_d_ln_fugacity_coefficients_dT_constrhovec(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Z = 1.0 + tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto dZdT_Z = tdx::template get_Ar11<be>(model, T, rhotot, molefrac)/(-T)/Z; // Note: (1/T)dX/d(1/T) = -TdX/dT, the deriv in RHS is what we want, the left is what we get, so divide by -T
        VectorType grad = build_Psir_gradient<be>(model, T, rhovec).eval();
        VectorType Tgrad = build_d2PsirdTdrhoi_autodiff(model, T, rhovec);
        return forceeval((1/(R*T)*(Tgrad - 1.0/T*grad)-dZdT_Z).eval());
    }
    
    /**
    * \brief Calculate ln(Z), Z, and dZ/drho at constant temperature and mole fractions
    *
    * Uses autodiff to calculate derivatives by default
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_lnZ_Z_dZdrho(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Ar01 = tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto Ar02 = tdx::template get_Ar02<be>(model, T, rhotot, molefrac);
        auto Z = 1.0 + Ar01;
        auto dZdrho = (Ar01 + Ar02)/rhotot; // (dZ/rhotot)_{T,x}
        return std::make_tuple(log(Z), Z, dZdrho);
    }
    
    /**
    * \brief Calculate the derivative of the natural logarithm of fugacity coefficient of each component w.r.t. molar density at constant temperature and mole fractions
    *
    * \f[
    * \deriv{ \ln\vec\phi}{\rho}{T,\vec x} = \frac{1}{RT}H(\Psi_r)\vec x - \frac{1}{Z}\deriv{Z}{\rho}{T,\vec x}
    * \f]
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_d_ln_fugacity_coefficients_drho_constTmolefracs(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        auto [lnZ, Z, dZdrho] = get_lnZ_Z_dZdrho(model, T, rhovec);
        auto hessian = build_Psir_Hessian_autodiff(model, T, rhovec);
        return forceeval((1/(R*T)*(hessian*molefrac.matrix()).array() - dZdrho/Z).eval());
    }
    
    /**
    * \brief Calculate the derivative of the natural logarithm of fugacity coefficient of each component w.r.t. molar volume at constant temperature and mole fractions
    *
    * \f[
    * \deriv{ \ln\vec\phi}{v}{T,\vec x} = \deriv{ \ln\vec\phi}{\rho}{T,\vec x}\deriv{ \rho}{v}{}
    * \f]
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_d_ln_fugacity_coefficients_dv_constTmolefracs(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto drhodv = -rhotot*rhotot; //  rho = 1/v; drho/dv = -1/v^2 = -rho^2
        return get_d_ln_fugacity_coefficients_drho_constTmolefracs(model, T, rhovec)*drhodv;
    }
    
    /**
    * \brief Calculate the derivative of the natural logarithm of fugacity coefficient of each component w.r.t. mole fraction of each component, at constant temperature and molar density
    *
    * \f[
    * \deriv{ \ln\vec\phi}{\vec x}{T, \rho}
    * \f]
    */
    template<ADBackends be = ADBackends::autodiff>
    static auto get_d_ln_fugacity_coefficients_dmolefracs_constTrho(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        
        using tdx = TDXDerivatives<Model, Scalar, VectorType>;
        auto Ar01 = tdx::template get_Ar01<be>(model, T, rhotot, molefrac);
        auto Z = 1.0 + Ar01;
        VectorType dZdx = rhotot*build_d2alphardrhodxi_constT(model, T, rhotot, molefrac);
        Eigen::RowVector<decltype(rhotot), Eigen::Dynamic> dZdx_Z = dZdx/Z;
        
        // Starting matrix is from the first term
        auto hessian = build_Psir_Hessian_autodiff(model, T, rhovec);
        Eigen::ArrayXXd out = rhotot/(R*T)*hessian;
        
        // Then each row gets the second part
        out.rowwise() -= dZdx_Z.array();
        return out;
    }
    
    template<ADBackends be = ADBackends::autodiff>
    static auto get_d_ln_fugacity_coefficients_dmolefracs_constTrho1(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = forceeval(rhovec.sum());
        auto molefrac = (rhovec / rhotot).eval();
        auto R = model.R(molefrac);
        
        auto hessian = build_Psir_Hessian_autodiff(model, T, rhovec);
        // Starting matrix is from the first term
        Eigen::ArrayXXd out = 1/(R*T)*rhotot*hessian;
        return out;
    }

    /**
    * \brief Calculate the temperature derivative of the chemical potential of each component
    * \note: Some contributions to the ideal gas part are missing (reference state and cp0), but are not relevant to phase equilibria
    */
    static auto get_dchempotdT_autodiff(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto rhorefideal = 1.0;
        return (build_d2PsirdTdrhoi_autodiff(model, T, rhovec) + model.R(molefrac)*(rhorefideal + log(rhovec/rhorefideal))).eval();
    }

    /**
    * \brief Calculate the temperature derivative of the pressure at constant molar concentrations
    */
    static auto get_dpdT_constrhovec(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto dPsirdT = get_dPsirdT_constrhovec(model, T, rhovec);
        return rhotot*model.R(molefrac) - dPsirdT + rhovec.matrix().dot(build_d2PsirdTdrhoi_autodiff(model, T, rhovec).matrix());
    }

    /**
    * \brief Calculate the molar concentration derivatives of the pressure at constant temperature
    */
    static auto get_dpdrhovec_constT(const Model& model, const Scalar& T, const VectorType& rhovec) {
        auto rhotot = rhovec.sum();
        auto molefrac = (rhovec / rhotot).eval();
        auto RT = model.R(molefrac)*T;
        auto [func, grad, hessian] = build_Psir_fgradHessian_autodiff(model, T, rhovec); // The hessian matrix
        return (RT + (hessian*rhovec.matrix()).array()).eval(); // at constant temperature
    }

    /**
    \brief Calculate the partial molar volumes of each component
    
    \f[
    \hat v_i = \left(\frac{\partial V}{\partial n_i}\right)_{T,V,n_{j \neq i}}
    \f]
    
     Eq 7.32 from GERG-2004
     \f[
             \hat v_i = \deriv{V}{n_i}{T,p,n_j} = \displaystyle\frac{-n\deriv{p}{n_i}{T,V,n_j}}{n\deriv{p}{V}{T,\vec{n}}}
     \f]

     Total differential of a variable \f$Y\f$ that is a function of \f$T\f$, \f$\vec{\rho}\f$:
     \f[
     \mathrm{d} Y = \deriv{Y}{T}{\vec{\rho}} \mathrm{d} T + \sum_k \deriv{Y}{\rho_k}{T,\rho_{j\neq i}} \mathrm{d} \rho_k
     \f]
     so
     \f[
     \deriv{Y}{n_i}{T,V,n_{j\neq i}} = \deriv{Y}{T}{\vec{\rho}} \cancelto{0}{\deriv{T}{n_i}{T,V,n_j}} + \sum_k \deriv{Y}{\rho_k}{T,\rho_{j\neq i}}  \deriv{\rho_k}{n_i}{T,V,n_j}
     \f]
     \f[
     \deriv{\rho_k}{n_i}{T,V,n_j} = \frac{\delta_{ik}}{V}
     \f]
     because \f$\rho_k = n_k/V\f$.

     Thus in the isochoric framework, the partial molar volume is given by
     \f[
     \hat v_i = \displaystyle\frac{-\frac{n}{V}\deriv{p}{\rho_i}{T,\rho_j}}{n\deriv{p}{V}{T,\vec{n}}}
     \f]
     The final formulation includes this term in the numerator (see the supporting info from isochoric paper):
     \f[
     \deriv{p}{\rho_i}{T,\rho_j} = RT + \sum_k\rho_k\deriv{^2\Psi^{\rm r}}{\rho_k\partial \rho_i}{T,\rho_{j\neq k}}
     \f]
     and the denominator is given by (Eq. 7.62 of GERG-2004)
     \f[
     n\deriv{p}{V}{T,\vec{n}} = -\rho^2 RT\left(1+2\delta\deriv{\alpha^{\rm r}}{\delta}{\tau} + \delta^2\deriv{^2\alpha^{\rm r}}{\delta^2}{\tau} \right)
     \f]
     and finally
     \f[
     \hat v_i = \displaystyle\frac{-\rho\deriv{p}{\rho_i}{T,\rho_j}}{n\deriv{p}{V}{T,\vec{n}}}
     \f]
     
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
    
    static VectorType get_Psir_sigma_derivs(const Model& model, const Scalar& T, const VectorType& rhovec, const VectorType& v) {
        autodiff::Real<4, double> sigma = 0.0;
        auto rhovecad = rhovec.template cast<decltype(sigma)>(), vad = v.template cast<decltype(sigma)>();
        auto wrapper = [&rhovecad, &vad, &T, &model](const auto& sigma_1) {
            auto rhovecused = (rhovecad + sigma_1 * vad).eval();
            auto rhotot = rhovecused.sum();
            auto molefrac = (rhovecused / rhotot).eval();
            return forceeval(model.alphar(T, rhotot, molefrac) * model.R(molefrac) * T * rhotot);
        };
        auto der = derivatives(wrapper, along(1), at(sigma));
        VectorType ret(der.size());
        for (auto i = 0; i < ret.size(); ++i){ ret[i] = der[i];}
        return ret;
    }
};

template<int Nderivsmax>
class DerivativeHolderSquare{
    
public:
    Eigen::Array<double, Nderivsmax+1, Nderivsmax+1> derivs;
    
    template<typename Model, typename Scalar, typename VecType>
    DerivativeHolderSquare(const Model& model, const Scalar& T, const Scalar& rho, const VecType& z) {
        using tdx = TDXDerivatives<decltype(model), Scalar, VecType>;
        static_assert(Nderivsmax == 2, "It's gotta be 2 for now");
        
        
        auto AX02 = tdx::template get_Agen0n<2>(model, T, rho, z);
        derivs(0, 0) = AX02[0];
        derivs(0, 1) = AX02[1];
        derivs(0, 2) = AX02[2];
        
        auto AX20 = tdx::template get_Agenn0<2>(model, T, rho, z);
        derivs(0, 0) = AX20[0];
        derivs(1, 0) = AX20[1];
        derivs(2, 0) = AX20[2];
        
        derivs(1, 1) = tdx::template get_Agenxy<1,1>(model, T, rho, z);
    }
};



}; // namespace teqp
