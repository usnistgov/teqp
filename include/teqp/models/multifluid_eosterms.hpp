#pragma once

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/models/cubics/simple_cubics.hpp"
#include "teqp/models/saft/pcsaftpure.hpp"

namespace teqp {

/**
\f$ \alpha^{\rm r}=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i}\f$
*/
class JustPowerEOSTerm {
public:
    Eigen::ArrayXd n, t, d;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0;
        TauType lntau = log(tau);
        double base_delta = getbaseval(delta);
        if (base_delta == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau)*powi(delta, static_cast<int>(d[i]));
            }
        }
        else {
            DeltaType lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r += n[i] * exp(t[i] * lntau + d[i] * lndelta);
            }
        }
        return forceeval(r);
    }
};

/**
\f$ \alpha^{\rm r}=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i} \exp(-c_i\delta^{l_i})\f$
*/
class PowerEOSTerm {
public:
    struct PowerEOSTermCoeffs {
        Eigen::ArrayXd n, t, d, c, l;
        Eigen::ArrayXi l_i;
    };
    const PowerEOSTermCoeffs coeffs;
    
    PowerEOSTerm(const PowerEOSTermCoeffs& coef) : coeffs(coef){}

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0;
        TauType lntau = log(tau);
        if (coeffs.l_i.size() == 0 && coeffs.n.size() > 0) {
            throw std::invalid_argument("l_i cannot be zero length if some terms are provided");
        }
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < coeffs.n.size(); ++i) {
                r += coeffs.n[i] * exp(coeffs.t[i] * lntau - coeffs.c[i] * powi(delta, coeffs.l_i[i])) * powi(delta, static_cast<int>(coeffs.d[i]));
            }
        }
        else {
            DeltaType lndelta = log(delta);
            result arg;
            DeltaType dpart;
            for (auto i = 0; i < coeffs.n.size(); ++i) {
                dpart = coeffs.d[i] * lndelta - coeffs.c[i] * powi(delta, coeffs.l_i[i]);
                arg = (coeffs.t[i] * lntau) + dpart;
                r += coeffs.n[i] * exp(arg);
            }
        }
        return r;
    }
};

/**
\f$ \alpha^{\rm r}=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i} \exp(-\gamma_i\delta^{l_i})\f$
*/
class ExponentialEOSTerm {
public:
    Eigen::ArrayXd n, t, d, g, l;
    Eigen::ArrayXi l_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau  - g[i] * powi(delta, l_i[i]))*powi(delta,static_cast<int>(d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r += n[i] * exp(t[i] * lntau + d[i] * lndelta - g[i] * powi(delta, l_i[i]));
            }
        }
        return forceeval(r);
    }
};

/**
\f$ \alpha^{\rm r}=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i} \exp(-\gamma_{d,i}\delta^{l_{d,i}}-\gamma_{t,i}\tau^{l_{t,i}})\f$
*/
class DoubleExponentialEOSTerm {
public:
    Eigen::ArrayXd n, t, d, gd, ld, gt, lt;
    Eigen::ArrayXi ld_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        if (ld_i.size() == 0 && n.size() > 0) {
            throw std::invalid_argument("ld_i cannot be zero length if some terms are provided");
        }
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * powi(delta, static_cast<int>(d[i])) * exp(t[i] * lntau - gd[i]*powi(delta, ld_i[i]) - gt[i]*pow(tau, lt[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau + d[i] * lndelta - gd[i]*powi(delta, ld_i[i]) - gt[i]*pow(tau, lt[i]));
            }
        }
        return forceeval(r);
    }
};

/**
\f$ \alpha^{\rm r} = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 -\beta_i(\tau-\gamma_i)^2 )\f$
*/
class GaussianEOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        auto square = [](auto x) { return x * x; };
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau - eta[i] * square(delta - epsilon[i]) - beta[i] * square(tau - gamma[i]))*powi(delta, static_cast<int>(d[i]));
            }
        }
        else {
            DeltaType lndelta = log(delta);
            DeltaType d1, d2;
            TauType t1, t2;
            result arg;
            for (auto i = 0; i < n.size(); ++i) {
                d1 = delta - epsilon[i]; d2 = d1*d1;
                t1 = tau - gamma[i]; t2 = t1*t1;
                arg = t[i] * lntau + d[i] * lndelta - eta[i]*d2 - beta[i]*t2;
                r = r + n[i] * exp(arg);
            }
        }
        return forceeval(r);
    }
};

/**
\f$ \alpha^{\rm r} = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 -\beta_i(\delta-\gamma_i) )\f$
*/
class GERG2004EOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        auto square = [](auto x) { return x * x; };
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau - eta[i] * square(delta - epsilon[i]) - beta[i] * (delta - gamma[i]))*powi(delta, static_cast<int>(d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau + d[i] * lndelta - eta[i] * square(delta - epsilon[i]) - beta[i] * (delta - gamma[i]));
            }
        }
        return forceeval(r);
    }
};


/**
\f$ \alpha^{\rm r} = \displaystyle\sum_i n_i \delta^ { d_i } \tau^ { t_i } \exp(-\delta^ { l_i } - \tau^ { m_i })\f$
*/
class Lemmon2005EOSTerm {
public:
    Eigen::ArrayXd n, t, d, l, m;
    Eigen::ArrayXi l_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau - powi(delta, l_i[i]) - pow(tau, m[i]))*powi(delta, static_cast<int>(d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau + d[i] * lndelta - powi(delta, l_i[i]) - pow(tau, m[i]));
            }
        }
        return forceeval(r);
    }
};

/**
\f$ \alpha^{\rm r} = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp\left(-\eta_i(\delta-\epsilon_i)^2 + \frac{1}{\beta_i(\tau-\gamma_i)^2+b_i}\right)\f$
*/
class GaoBEOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon, b;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {

        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        auto square = [](auto x) { return x * x; };
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau - eta[i] * square(delta - epsilon[i]) + 1.0 / (beta[i] * square(tau - gamma[i]) + b[i]))*powi(delta, static_cast<int>(d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < n.size(); ++i) {
                r = r + n[i] * exp(t[i] * lntau + d[i] * lndelta - eta[i] * square(delta - epsilon[i]) + 1.0 / (beta[i] * square(tau - gamma[i]) + b[i]));
            }
        }
        return forceeval(r);
    }
};

/**
The contribution is a Chebyshev expansion in two dimensions
*/
class Chebyshev2DEOSTerm {
public:
    Eigen::ArrayXXd a;
    double taumin = -1, taumax = -1, deltamin = -1, deltamax = -1;

    /// Clenshaw evaluation of a Chebyshev expansion in 1D
    template<typename vectype, typename XType>
    static auto Clenshaw1D(const vectype &c, const XType &ind){
        int N = static_cast<int>(c.size()) - 1;
        std::common_type_t<typename vectype::Scalar, XType> u_k = 0, u_kp1 = 0, u_kp2 = 0;
        for (int k = N; k >= 0; --k){
            // Do the recurrent calculation
            u_k = 2.0*ind*u_kp1 - u_kp2 + c[k];
            if (k > 0){
                // Update the values
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return (u_k - u_kp2)/2.0;
    }

    /// Clenshaw evaluation of one dimensional flattening of the Chebyshev expansion
    template<typename MatType, typename XType>
    static auto Clenshaw1DByRow(const MatType& c, const XType &ind) {
        int N = static_cast<int>(c.rows()) - 1;
        constexpr int Cols = MatType::ColsAtCompileTime;
        using NumType = std::common_type_t<typename MatType::Scalar, XType>;
        static Eigen::Array<NumType, 1, Cols> u_k, u_kp1, u_kp2;
        // Not statically sized, need to resize
        if constexpr (Cols == Eigen::Dynamic) {
            int M = static_cast<int>(c.rows());
            u_k.resize(M); 
            u_kp1.resize(M);
            u_kp2.resize(M);
        }
        u_k.setZero(); u_kp1.setZero(); u_kp2.setZero();
        
        for (int k = N; k >= 0; --k) {
            // Do the recurrent calculation
            u_k = 2.0 * ind * u_kp1 - u_kp2 + c.row(k).template cast<XType>();
            if (k > 0) {
                // Update the values
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return (u_k - u_kp2) / 2.0;
    }

    /** Clenshaw evaluation of the complete expansion
     * \param a Matrix
     * \param x The first argument, in [-1,1]
     * \param y The second argument, in [-1,1]
     */
    template<typename MatType, typename XType, typename YType>
    static auto Clenshaw2DEigen(const MatType& a, const XType &x, const YType &y) {
        auto b = Clenshaw1DByRow(a, y);
        return Clenshaw1D(b.matrix(), x);
    }

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        TauType x = (2.0*tau - (taumax + taumin)) / (taumax - taumin);
        DeltaType y = (2.0*delta - (deltamax + deltamin)) / (deltamax - deltamin);
        return forceeval(Clenshaw2DEigen(a, x, y));
    }
};

/**
\f$ \alpha^r = 0\f$
*/
class NullEOSTerm {
public:
    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& /*tau*/, const DeltaType& /*delta*/) const {
        return static_cast<std::common_type_t<TauType, DeltaType>>(0.0);
    }
};

class NonAnalyticEOSTerm {
public:
    Eigen::ArrayXd A, B, C, D, a, b, beta, n;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        // The non-analytic term
        auto square = [](auto x) { return x * x; };
        auto delta_min1_sq = square(delta - 1.0);

        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0;
        for (auto i = 0; i < n.size(); ++i) {
            auto Psi = exp(-C[i]*delta_min1_sq - D[i]*square(tau - 1.0));
            auto k = 1.0 / (2.0 * beta[i]);
            auto theta = (1.0 - tau) + A[i] * pow(delta_min1_sq, k);
            auto Delta = square(theta) + B[i]*pow(delta_min1_sq, a[i]);
            r = r + n[i]*pow(Delta, b[i])*delta*Psi;
        }
        result outval = forceeval(r);

        // If we are really, really close to the critical point (tau=delta=1), then the term will become undefined, so let's just return 0 in that case
        double dbl = static_cast<double>(getbaseval(outval));
        if (std::isfinite(dbl)) {
            return outval;
        }
        else {
            return static_cast<decltype(outval)>(0.0);
        }
    }
};

/**
 This implementation is for generic cubic EOS, in teh 
 */
class GenericCubicTerm {
public:
    const double Tcrit_K, pcrit_Pa, R_gas, Delta1, Delta2, Tred_K, rhored_molm3, a0_cubic, b_cubic;
    const std::vector<AlphaFunctionOptions> alphas_cubic;
    
    GenericCubicTerm(const nlohmann::json& spec) :
        Tcrit_K(spec.at("Tcrit / K")),
        pcrit_Pa(spec.at("pcrit / Pa")),
        R_gas(spec.at("R / J/mol/K")),
        Delta1(spec.at("Delta1")),
        Delta2(spec.at("Delta2")),
        Tred_K(spec.at("Tred / K")),
        rhored_molm3(spec.at("rhored / mol/m^3")),
        a0_cubic(spec.at("OmegaA").get<double>() * pow2(R_gas * Tcrit_K) / pcrit_Pa),
        b_cubic(spec.at("OmegaB").get<double>() * R_gas * Tcrit_K / pcrit_Pa),
        alphas_cubic(build_alpha_functions(std::vector<double>(1, Tcrit_K), spec.at("alpha")))
    {
        if (alphas_cubic.size() != 1){
            throw teqp::InvalidArgument("alpha should be of size 1");
        }
    }
    
    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        auto T = Tred_K/tau;
        auto rhomolar = delta*rhored_molm3;
        auto alpha = forceeval(std::visit([&](auto& t) { return t(T); }, alphas_cubic[0]));
        auto a_cubic = a0_cubic*alpha;
        auto Psiminus = -log(1.0 - b_cubic * rhomolar);
        auto Psiplus = log((Delta1 * b_cubic * rhomolar + 1.0) / (Delta2 * b_cubic * rhomolar + 1.0)) / (b_cubic * (Delta1 - Delta2));
        auto val = Psiminus - a_cubic / (R_gas * T) * Psiplus;
        return forceeval(val);
    }
};

/**
 This implementation is for PC-SAFT for a pure fluid as taken from Gross & Sadowski, I&ECR, 2001
 */
class PCSAFTGrossSadowski2001Term {
public:
    const double Tred_K, rhored_molm3;
    const saft::PCSAFT::PCSAFTPureGrossSadowski2001 pcsaft;
    
    PCSAFTGrossSadowski2001Term(const nlohmann::json& spec) :
        Tred_K(spec.at("Tred / K")),
        rhored_molm3(spec.at("rhored / mol/m^3")),
        pcsaft(spec) // The remaining arguments will be consumed by the constructor
    {}
    
    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        auto T = forceeval(Tred_K/tau);
        auto rhomolar = forceeval(delta*rhored_molm3);
        return forceeval(pcsaft.alphar(T, rhomolar, Eigen::Array<double,1,1>{}));
    }
};

template<typename... Args>
class EOSTermContainer {  
private:
    using varEOSTerms = std::variant<Args...>;
    std::vector<varEOSTerms> coll;
public:

    auto size() const { return coll.size(); }

    template<typename Instance>
    auto add_term(Instance&& instance) {
        coll.emplace_back(instance);
    }

    template <class Tau, class Delta>
    auto alphar(const Tau& tau, const Delta& delta) const {
        std::common_type_t <Tau, Delta> ar = 0.0;
        for (const auto& term : coll) {
//            // This approach is recommended to speed up visitor, but doesn't seem to make a difference in Xcode
//            if (const auto t = std::get_if<JustPowerEOSTerm>(&term)){
//                ar += t->alphar(tau, delta); continue;
//            }
//            if (const auto t = std::get_if<GaussianEOSTerm>(&term)){
//                ar += t->alphar(tau, delta); continue;
//            }
//            if (const auto t = std::get_if<PowerEOSTerm>(&term)){
//                ar += t->alphar(tau, delta); continue;
//            }
            auto contrib = std::visit([&](auto& t) { return t.alphar(tau, delta); }, term);
            ar += contrib;
        }
        return ar;
    }
};

using EOSTerms = EOSTermContainer<JustPowerEOSTerm, PowerEOSTerm, GaussianEOSTerm, NonAnalyticEOSTerm, Lemmon2005EOSTerm, GaoBEOSTerm, ExponentialEOSTerm, DoubleExponentialEOSTerm, GenericCubicTerm, PCSAFTGrossSadowski2001Term>;

using DepartureTerms = EOSTermContainer<JustPowerEOSTerm, PowerEOSTerm, GaussianEOSTerm, GERG2004EOSTerm, NullEOSTerm, DoubleExponentialEOSTerm,Chebyshev2DEOSTerm>;

}; // namespace teqp
