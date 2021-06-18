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
\f$ \alpha^ r = \displaystyle\sum_i n_i \delta^ { d_i } \tau^ { t_i } \exp(-\delta^ { l_i } - \tau^ { m_i })\f$
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
\f$ \alpha^ r = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 + \frac{1}{\beta_i(\tau-\gamma_i)^2+b_i}\f$
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

        return forceeval((n * pow(Delta, b) * delta * Psi).eval().sum());
    }
};



template<typename... Args>
class EOSTermContainer {
public:
    using varEOSTerms = std::variant<Args...>;
private:
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
            if (std::holds_alternative<NonAnalyticEOSTerm>(term)) {
                double dbl = getbaseval(abs(contrib));
                if (std::isfinite(dbl)) {
                    ar = ar + contrib;
                }
            }
            else {
                ar = ar + contrib;
            }
        }
        return ar;
    }
};

using EOSTerms = EOSTermContainer<PowerEOSTerm, GaussianEOSTerm, NonAnalyticEOSTerm, Lemmon2005EOSTerm, GaoBEOSTerm, ExponentialEOSTerm>;