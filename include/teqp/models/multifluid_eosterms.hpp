#pragma once

class PowerEOSTerm {
public:
    Eigen::ArrayXd n, t, d, c, l;
    Eigen::ArrayXi l_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - c * powIVi(delta, l_i))).sum());
    }
};

/**
\f$ \alpha^r=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i} \exp(-\gamma_i\delta^{l_i})\f$
*/
class ExponentialEOSTerm {
public:
    Eigen::ArrayXd n, t, d, g, l;
    Eigen::ArrayXi l_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - g * powIVi(delta, l_i))).sum());
    }
};

/**
\f$ \alpha^r = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 -\beta_i(\tau-\gamma_i)^2 }\f$
*/
class GaussianEOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - eta * (delta - epsilon).square() - beta * (tau - gamma).square())).sum());
    }
};

/**
\f$ \alpha^r = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 -\beta_i(\delta-\gamma_i) }\f$
*/
class GERG2004EOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - eta * (delta - epsilon).square() - beta * (tau - gamma).square())).sum());
    }
};


/**
\f$ \alpha^r = \displaystyle\sum_i n_i \delta^ { d_i } \tau^ { t_i } \exp(-\delta^ { l_i } - \tau^ { m_i })\f$
*/
class Lemmon2005EOSTerm {
public:
    Eigen::ArrayXd n, t, d, l, m;
    Eigen::ArrayXi l_i;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - powIVi(delta, l_i) - pow(tau, m))).sum());
    }
};

/**
\f$ \alpha^r = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 + \frac{1}{\beta_i(\tau-\gamma_i)^2+b_i}\f$
*/
class GaoBEOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon, b;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        auto terms = n * exp(t * log(tau) + d * log(delta) - eta * (delta - epsilon).square() + 1.0 / (beta * (tau - gamma).square() + b));
        return forceeval(terms.sum());
    }
};

/**
\f$ \alpha^r = 0\f$
*/
class NullEOSTerm {
public:
    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
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
        auto Psi = (exp(-C * delta_min1_sq - D * square(tau - 1.0))).eval();
        const Eigen::ArrayXd k = 1.0 / (2.0 * beta);
        auto theta = ((1.0 - tau) + A * pow(delta_min1_sq, k)).eval();
        auto Delta = (theta.square() + B * pow(delta_min1_sq, a)).eval();

        auto outval = forceeval((n * pow(Delta, b) * delta * Psi).eval().sum());

        // If we are really, really close to the critical point (tau=delta=1), then the term will become undefined, so let's just return 0 in that case
        double dbl = getbaseval(outval);
        if (std::isfinite(dbl)) {
            return outval;
        }
        else {
            return static_cast<decltype(outval)>(0.0);
        }
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
        coll.emplace_back(std::move(instance));
    }

    template <class Tau, class Delta>
    auto alphar(const Tau& tau, const Delta& delta) const {
        std::common_type_t <Tau, Delta> ar = 0.0;
        for (const auto& term : coll) {
            auto contrib = std::visit([&](auto& t) { return t.alphar(tau, delta); }, term);
            ar = ar + contrib;
        }
        return ar;
    }
};

using EOSTerms = EOSTermContainer<PowerEOSTerm, GaussianEOSTerm, NonAnalyticEOSTerm, Lemmon2005EOSTerm, GaoBEOSTerm, ExponentialEOSTerm>;

using DepartureTerms = EOSTermContainer<PowerEOSTerm, GaussianEOSTerm, GERG2004EOSTerm, NullEOSTerm>;