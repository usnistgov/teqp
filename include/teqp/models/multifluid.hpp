#pragma once

#include "nlohmann/json.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <cmath>
#include <optional>
#include <variant>

#include "teqp/types.hpp"
#include "MultiComplex/MultiComplex.hpp"

// See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
namespace Eigen {
    template<typename TN> struct NumTraits<mcx::MultiComplex<TN>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };
}

template<typename EOSCollection>
class CorrespondingStatesContribution {

private:
    const EOSCollection EOSs;
public:
    CorrespondingStatesContribution(EOSCollection&& EOSs) : EOSs(EOSs) {};

    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = std::common_type_t<decltype(tau), decltype(molefracs[0]), decltype(delta)>; // Type promotion, without the const-ness
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        for (auto i = 0; i < N; ++i) {
            alphar = alphar + molefracs[i] * EOSs[i].alphar(tau, delta);
        }
        return alphar;
    }
};

template<typename FCollection, typename DepartureFunctionCollection>
class DepartureContribution {

private:
    const FCollection F;
    const DepartureFunctionCollection funcs;
public:
    DepartureContribution(FCollection&& F, DepartureFunctionCollection&& funcs) : F(F), funcs(funcs) {};

    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = std::common_type_t<decltype(tau), decltype(molefracs[0]), decltype(delta)>; // Type promotion, without the const-ness
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i+1; j < N; ++j) {
                alphar = alphar + molefracs[i] * molefracs[j] * F(i, j) * funcs[i][j].alphar(tau, delta);
            }
        }
        return alphar;
    }
};

template<typename ReducingFunction, typename CorrespondingTerm, typename DepartureTerm>
class MultiFluid {  

public:
    const ReducingFunction redfunc;
    const CorrespondingTerm corr;
    const DepartureTerm dep;

    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    MultiFluid(ReducingFunction&& redfunc, CorrespondingTerm&& corr, DepartureTerm&& dep) : redfunc(redfunc), corr(corr), dep(dep) {};

    template<typename TType, typename RhoType>
    auto alphar(TType T,
        const RhoType& rhovec,
        const std::optional<typename RhoType::value_type> rhotot = std::nullopt) const
    {
        typename RhoType::value_type rhotot_ = (rhotot.has_value()) ? rhotot.value() : std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        auto molefrac = rhovec / rhotot_;
        return alphar(T, rhotot_, molefrac);
    }

    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType &T,
        const RhoType &rho,
        const MoleFracType& molefrac) const
    {
        auto Tred = forceeval(redfunc.get_Tr(molefrac));
        auto rhored = forceeval(redfunc.get_rhor(molefrac));
        auto delta = forceeval(rho / rhored);
        auto tau = forceeval(Tred / T);
        auto val = corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
        return forceeval(val);
    }
};


class MultiFluidReducingFunction {
private:
    Eigen::MatrixXd YT, Yv;

    template <typename Num>
    auto cube(Num x) const {
        return x*x*x;
    }
    template <typename Num>
    auto square(Num x) const {
        return x*x;
    }

public:
    const Eigen::MatrixXd betaT, gammaT, betaV, gammaV;
    const Eigen::ArrayXd Tc, vc;

    template<typename ArrayLike>
    MultiFluidReducingFunction(
        const Eigen::MatrixXd& betaT, const Eigen::MatrixXd& gammaT,
        const Eigen::MatrixXd& betaV, const Eigen::MatrixXd& gammaV,
        const ArrayLike& Tc, const ArrayLike& vc)
        : betaT(betaT), gammaT(gammaT), betaV(betaV), gammaV(gammaV), Tc(Tc), vc(vc) {

        auto N = Tc.size();

        YT.resize(N, N); YT.setZero();
        Yv.resize(N, N); Yv.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i + 1; j < N; ++j) {
                YT(i, j) = betaT(i, j) * gammaT(i, j) * sqrt(Tc[i] * Tc[j]);
                YT(j, i) = betaT(j, i) * gammaT(j, i) * sqrt(Tc[i] * Tc[j]);
                Yv(i, j) = 1.0 / 8.0 * betaV(i, j) * gammaV(i, j) * cube(cbrt(vc[i]) + cbrt(vc[j]));
                Yv(j, i) = 1.0 / 8.0 * betaV(j, i) * gammaV(j, i) * cube(cbrt(vc[i]) + cbrt(vc[j]));
            }
        }
    }

    template <typename MoleFractions>
    auto Y(const MoleFractions& z, const Eigen::ArrayXd& Yc, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& Yij) const {

        auto N = z.size();
        typename MoleFractions::value_type sum1 = 0.0;
        for (auto i = 0; i < N; ++i) {
            sum1 = sum1 + square(z[i]) * Yc[i];
        }
        
        typename MoleFractions::value_type sum2 = 0.0;
        for (auto i = 0; i < N-1; ++i){
            for (auto j = i+1; j < N; ++j) {
                sum2 = sum2 + 2.0*z[i]*z[j]*(z[i] + z[j])/(square(beta(i, j))*z[i] + z[j])*Yij(i, j);
            }
        }

        return sum1 + sum2;
    }

    static auto get_BIPdep(const nlohmann::json& collection, const std::vector<std::string>& components, const nlohmann::json& flags) {

        if (flags.contains("estimate")) {
            return nlohmann::json({
                {"betaT", 1.0}, {"gammaT", 1.0}, {"betaV", 1.0}, {"gammaV", 1.0}, {"F", 0.0} 
            });
        }

        // convert string to upper case
        auto toupper = [](const std::string s){ auto data = s; std::for_each(data.begin(), data.end(), [](char& c) { c = ::toupper(c); }); return data;};

        std::string comp0 = toupper(components[0]);
        std::string comp1 = toupper(components[1]);
        for (auto& el : collection) {
            std::string name1 = toupper(el["Name1"]);
            std::string name2 = toupper(el["Name2"]);
            if (comp0 == name1 && comp1 == name2) {
                return el;
            }
            if (comp0 == name2 && comp1 == name1) {
                return el;
            }
        }
        throw std::invalid_argument("Can't match this binary pair");
    }
    static auto get_binary_interaction_double(const nlohmann::json& collection, const std::vector<std::string>& components, const nlohmann::json& flags) {
        auto el = get_BIPdep(collection, components, flags);

        double betaT = el["betaT"], gammaT = el["gammaT"], betaV = el["betaV"], gammaV = el["gammaV"];
        // Backwards order of components, flip beta values
        if (components[0] == el["Name2"] && components[1] == el["Name1"]) {
            betaT = 1.0 / betaT;
            betaV = 1.0 / betaV;
        }
        return std::make_tuple(betaT, gammaT, betaV, gammaV);
    }
    static auto get_BIP_matrices(const nlohmann::json& collection, const std::vector<std::string>& components, const nlohmann::json& flags) {
        Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;
        auto N = components.size();
        betaT.resize(N, N); betaT.setZero();
        gammaT.resize(N, N); gammaT.setZero();
        betaV.resize(N, N); betaV.setZero();
        gammaV.resize(N, N); gammaV.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i + 1; j < N; ++j) {
                auto [betaT_, gammaT_, betaV_, gammaV_] = get_binary_interaction_double(collection, { components[i], components[j] }, flags);
                betaT(i, j) = betaT_;         betaT(j, i) = 1.0 / betaT(i, j);
                gammaT(i, j) = gammaT_;       gammaT(j, i) = gammaT(i, j);
                betaV(i, j) = betaV_;         betaV(j, i) = 1.0 / betaV(i, j);
                gammaV(i, j) = gammaV_;       gammaV(j, i) = gammaV(i, j);
            }
        }
        return std::make_tuple(betaT, gammaT, betaV, gammaV);
    }
    static auto get_Tcvc(const std::string& coolprop_root, const std::vector<std::string>& components) {
        Eigen::ArrayXd Tc(components.size()), vc(components.size());
        using namespace nlohmann;
        auto i = 0;
        for (auto& c : components) {
            std::string path = coolprop_root + "/dev/fluids/" + c + ".json";
            std::ifstream ifs(path);
            if (!ifs) {
                throw std::invalid_argument("Load path is invalid: " + path);
            }
            auto j = json::parse(ifs);
            auto red = j["EOS"][0]["STATES"]["reducing"];
            double Tc_ = red["T"];
            double rhoc_ = red["rhomolar"];
            Tc[i] = Tc_;
            vc[i] = 1.0 / rhoc_;
            i++;
        }
        return std::make_tuple(Tc, vc);
    }
    static auto get_F_matrix(const nlohmann::json& collection, const std::vector<std::string>& components, const nlohmann::json& flags) {
        Eigen::MatrixXd F(components.size(), components.size());
        auto N = components.size();
        for (auto i = 0; i < N; ++i) {
            F(i, i) = 0.0;
            for (auto j = i + 1; j < N; ++j) {
                auto el = get_BIPdep(collection, { components[i], components[j] }, flags);
                if (el.empty()) {
                    F(i, j) = 0.0;
                    F(j, i) = 0.0;
                }
                else{
                    F(i, j) = el["F"];
                    F(j, i) = el["F"];
                }   
            }
        }
        return F;
    }
    template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return Y(molefracs, Tc, betaT, YT); }
    template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return 1.0 / Y(molefracs, vc, betaV, Yv); }
};

class MultiFluidDepartureFunction {
public:
    enum class types { NOTSETTYPE, GERG2004, GaussianExponential, NoDeparture };
private:
    types type = types::NOTSETTYPE;
public:
    Eigen::ArrayXd n, t, d, c, l, eta, beta, gamma, epsilon;

    void set_type(const std::string& kind) {
        if (kind == "GERG-2004" || kind == "GERG-2008") {
            type = types::GERG2004;
        }
        else if (kind == "Gaussian+Exponential") {
            type = types::GaussianExponential;
        }
        else if (kind == "none") {
            type = types::NoDeparture;
        }
        else {
            throw std::invalid_argument("Bad type:" + kind);
        }
    }

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        switch (type) {
        case (types::GaussianExponential):
            return forceeval((n * exp(t*log(tau) + d*log(delta)-c*pow(delta, l)-eta * (delta - epsilon).square() - beta * (tau - gamma).square())).sum());
        case (types::GERG2004):
            return forceeval((n * exp(t*log(tau) + d*log(delta) -eta * (delta - epsilon).square() - beta * (delta - gamma))).sum()); 
        case (types::NoDeparture):
            return forceeval(0.0*(tau*delta));
        default:
            throw - 1;
        }
    }
};

inline auto get_departure_function_matrix(const std::string& coolprop_root, const nlohmann::json& BIPcollection, const std::vector<std::string>& components, const nlohmann::json& flags) {

    // Allocate the matrix with default models
    std::vector<std::vector<MultiFluidDepartureFunction>> funcs(components.size()); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }

    auto depcollection = nlohmann::json::parse(std::ifstream(coolprop_root + "/dev/mixtures/mixture_departure_functions.json"));

    auto get_departure_function = [&depcollection](const std::string& Name) {
        for (auto& el : depcollection) {
            if (el["Name"] == Name) { return el; }
        }
        throw std::invalid_argument("Bad argument");
    };

    for (auto i = 0; i < funcs.size(); ++i) {
        for (auto j = i + 1; j < funcs.size(); ++j) {
            auto BIP = MultiFluidReducingFunction::get_BIPdep(BIPcollection, { components[i], components[j] }, flags);
            auto function = BIP["function"];
            if (!function.empty()) {

                auto info = get_departure_function(function);
                auto N = info["n"].size();

                auto toeig = [](const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); };
                auto eigorempty = [&info, &toeig, &N](const std::string& name) -> Eigen::ArrayXd {
                    if (!info[name].empty()) {
                        return toeig(info[name]);
                    }
                    else {
                        return Eigen::ArrayXd::Zero(N);
                    }
                };

                MultiFluidDepartureFunction f;
                f.set_type(info["type"]);
                f.n = toeig(info["n"]);
                f.t = toeig(info["t"]);
                f.d = toeig(info["d"]);

                f.eta = eigorempty("eta");
                f.beta = eigorempty("beta");
                f.gamma = eigorempty("gamma");
                f.epsilon = eigorempty("epsilon");

                Eigen::ArrayXd c(f.n.size()), l(f.n.size()); c.setZero();
                if (info["l"].empty()) {
                    // exponential part not included
                    l.setZero();
                }
                else {
                    l = toeig(info["l"]);
                    // l is included, use it to build c; c_i = 1 if l_i > 0, zero otherwise
                    for (auto i = 0; i < c.size(); ++i) {
                        if (l[i] > 0) {
                            c[i] = 1.0;
                        }
                    }
                }

                f.l = l;
                f.c = c;
                funcs[i][j] = f;
                funcs[j][i] = f;
                int rr = 0;
            }
            else {
                MultiFluidDepartureFunction f;
                f.set_type("none");
                funcs[i][j] = f;
                funcs[j][i] = f;
            }
        }
    }
    return funcs;
}

/// From Ulrich Deiters
template <typename T>                             // arbitrary integer power
T powi(const T& x, int n) {
    if (n == 0)
        return static_cast<T>(1.0);                       // x^0 = 1 even for x == 0
    else if (n < 0){
        using namespace autodiff::detail;
        if constexpr (isDual<T> || isExpr<T>) {
            return eval(powi(eval(1.0/x), -n));
        }
        else {
            return powi(static_cast<T>(1.0) / x, -n);
        }
    }
    else {
        T y(x), xpwr(x);
        n--;
        while (n > 0) {
            if (n % 2 == 1) {
                y = y*xpwr;
                n--;
            }
            xpwr = xpwr*xpwr;
            n /= 2;
        }
        return y;
    }
}

template<typename T>
inline auto powIVi(const T& x, const Eigen::ArrayXi& e) {
    //return e.binaryExpr(e.cast<T>(), [&x](const auto&& a_, const auto& e_) {return static_cast<T>(powi(x, a_)); });
    static Eigen::Array<T, Eigen::Dynamic, 1> o;
    o.resize(e.size());
    for (auto i = 0; i < e.size(); ++i) {
        o[i] = powi(x, e[i]);
    }
    return o;
    //return e.cast<T>().unaryExpr([&x](const auto& e_) {return powi(x, e_); }).eval();
}

//template<typename T>
//auto powIV(const T& x, const Eigen::ArrayXd& e) {
//    Eigen::Array<T, Eigen::Dynamic, 1> o = e.cast<T>();
//    return o.unaryExpr([&x](const auto& e_) {return powi(x, e_); } ).eval();
//}

template<typename T>
auto pow(const std::complex<T> &x, const Eigen::ArrayXd& e) {
    Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> o(e.size());
    for (auto i = 0; i < e.size(); ++i) {
        o[i] = pow(x, e[i]);
    }
    return o;
}

template<typename T>
auto pow(const mcx::MultiComplex<T> &x, const Eigen::ArrayXd& e) {
    Eigen::Array<mcx::MultiComplex<T>, Eigen::Dynamic, 1> o(e.size());
    for (auto i = 0; i < e.size(); ++i) {
        o[i] = pow(x, e[i]);
    }
    return o;
}

template<class T>
struct PowIUnaryFunctor {
    const T m_base;
    PowIUnaryFunctor(T base) : m_base(base) {};
    typedef T result_type;
    result_type operator()(const int& e) const{
        switch (e) {
        case 0:
            return 1.0;
        case 1:
            return m_base;
        case 2:
            return m_base * m_base;
        default:
            return powi(m_base, e);
        }
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
        for (auto& term : coll) {
            std::visit([&](auto& term) { ar = ar + term.alphar(tau, delta); }, term);
        }
        return ar;
    }
};

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
\f$ \alpha^ r = \displaystyle\sum_i n_i \tau^{t_i}\delta^ {d_i} \exp(-\eta_i(\delta-\epsilon_i)^2 -\beta_i(\tau-\gamma_i)^2 }\f$
*/
class GaussianEOSTerm {
public:
    Eigen::ArrayXd n, t, d, eta, beta, gamma, epsilon;

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        return forceeval((n * exp(t * log(tau) + d * log(delta) - eta*(delta-epsilon).square() - beta*(tau-gamma).square())).sum());
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
        auto terms = n* exp(t * log(tau) + d * log(delta) - eta * (delta - epsilon).square() + 1.0 / (beta * (tau - gamma).square()+b) );
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
        const Eigen::ArrayXd k = 1.0 / (2.0*beta);
        auto theta = ((1.0 - tau) + A * pow(delta_min1_sq, k)).eval();
        auto Delta = (theta.square() + B * pow(delta_min1_sq, a)).eval();

        return forceeval((n*pow(Delta, b) * delta * Psi).eval().sum());
    }
};

using EOSTerms = EOSTermContainer<PowerEOSTerm, GaussianEOSTerm, NonAnalyticEOSTerm, Lemmon2005EOSTerm, GaoBEOSTerm>;

inline auto get_EOS_terms(const std::string& coolprop_root, const std::string& name)
{
    using namespace nlohmann;
    auto j = json::parse(std::ifstream(coolprop_root + "/dev/fluids/" + name + ".json"));
    auto alphar = j["EOS"][0]["alphar"];

    const std::vector<std::string> allowed_types = { "ResidualHelmholtzPower", "ResidualHelmholtzGaussian", "ResidualHelmholtzNonAnalytic","ResidualHelmholtzGaoB", "ResidualHelmholtzLemmon2005" };

    auto isallowed = [&](const auto& conventional_types, const std::string& name) {
        for (auto& a : conventional_types) { if (name == a) { return true; }; } return false;
    };

    for (auto& term : alphar) {
        std::string type = term["type"];
        if (!isallowed(allowed_types, type)) {
            std::string a = allowed_types[0]; for (auto i = 1; i < allowed_types.size(); ++i) { a += "," + allowed_types[i]; }
            throw std::invalid_argument("Bad type:" + type + "; allowed types are: {" + a + "}");
        }
    }

    auto toeig = [](const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); };    

    EOSTerms container;

    auto build_power = [&](auto term) {
        std::size_t N = term["n"].size();

        PowerEOSTerm eos;

        auto eigorzero = [&term, &toeig, &N](const std::string& name) -> Eigen::ArrayXd {
            if (!term[name].empty()) {
                return toeig(term[name]);
            }
            else {
                return Eigen::ArrayXd::Zero(N);
            }
        };


        eos.n = eigorzero("n");
        eos.t = eigorzero("t");
        eos.d = eigorzero("d");

        Eigen::ArrayXd c(N), l(N); c.setZero();
        if (term["l"].empty()) {
            // exponential part not included
            l.setZero();
        }
        else {
            l = toeig(term["l"]);
            // l is included, use it to build c; c_i = 1 if l_i > 0, zero otherwise
            for (auto i = 0; i < c.size(); ++i) {
                if (l[i] > 0) {
                    c[i] = 1.0;
                }
            }
        }
        eos.c = c;
        eos.l = l;

        eos.l_i = eos.l.cast<int>();

        if (((eos.l_i.cast<double>() - eos.l).cwiseAbs() > 0.0).any()) {
            throw std::invalid_argument("Non-integer entry in l found");
        }
        
        return eos;
    };

    auto build_Lemmon2005 = [&](auto term) {
        std::size_t N = term["n"].size();

        Lemmon2005EOSTerm eos;

        auto eigorzero = [&term, &toeig, &N](const std::string& name) -> Eigen::ArrayXd {
            if (!term[name].empty()) {
                return toeig(term[name]);
            }
            else {
                return Eigen::ArrayXd::Zero(N);
            }
        };
        eos.n = eigorzero("n");
        eos.t = eigorzero("t");
        eos.d = eigorzero("d");
        eos.m = eigorzero("m");
        eos.l = eigorzero("l");
        eos.l_i = eos.l.cast<int>();
        if (((eos.l_i.cast<double>() - eos.l).cwiseAbs() > 0.0).any()) {
            throw std::invalid_argument("Non-integer entry in l found");
        }

        return eos;
    };

    auto build_gaussian = [&](auto term) {
        std::size_t N = term["n"].size();

        GaussianEOSTerm eos;

        auto eigorzero = [&term, &toeig, &N](const std::string& name) -> Eigen::ArrayXd {
            if (!term[name].empty()) {
                return toeig(term[name]);
            }
            else {
                return Eigen::ArrayXd::Zero(N);
            }
        };

        eos.n = eigorzero("n");
        eos.t = eigorzero("t");
        eos.d = eigorzero("d");
        eos.eta = eigorzero("eta");
        eos.beta = eigorzero("beta");
        eos.gamma = eigorzero("gamma");
        eos.epsilon = eigorzero("epsilon");
        return eos;
    };

    auto build_GaoB = [&](auto term) {
        GaoBEOSTerm eos;
        eos.n = toeig(term["n"]);
        eos.t = toeig(term["t"]);
        eos.d = toeig(term["d"]);
        eos.eta = -toeig(term["eta"]); // Watch out for this sign flip!!
        eos.beta = toeig(term["beta"]);
        eos.gamma = toeig(term["gamma"]);
        eos.epsilon = toeig(term["epsilon"]);
        eos.b = toeig(term["b"]);
        return eos;
    };

    /// lambda function for adding non-analytic terms
    auto build_na = [&toeig](auto& term) {
        auto eigorzero = [&term, &toeig](const std::string& name) -> Eigen::ArrayXd {
            return toeig(term[name]);
        };
        NonAnalyticEOSTerm eos;
        eos.n = eigorzero("n");
        eos.A = eigorzero("A");
        eos.B = eigorzero("B");
        eos.C = eigorzero("C");
        eos.D = eigorzero("D");
        eos.a = eigorzero("a");
        eos.b = eigorzero("b");
        eos.beta = eigorzero("beta");
        return eos;
    };
    
    for (auto& term : alphar) {
        auto type = term["type"];
        if (type == "ResidualHelmholtzPower") {
            container.add_term(build_power(term));
        }
        else if (type == "ResidualHelmholtzGaussian") {
            container.add_term(build_gaussian(term));
        }
        else if (type == "ResidualHelmholtzNonAnalytic") {
            container.add_term(build_na(term));
        }
        else if (type == "ResidualHelmholtzLemmon2005") {
            container.add_term(build_Lemmon2005(term));
        }
        else if (type == "ResidualHelmholtzGaoB") {
            container.add_term(build_GaoB(term));
        }
        else {
            throw std::invalid_argument("Bad term type, should not get here");
        }
    }
    return container;
}

inline auto get_EOSs(const std::string& coolprop_root, const std::vector<std::string>& names) {
    std::vector<EOSTerms> EOSs;
    for (auto& name : names) {
        auto term = get_EOS_terms(coolprop_root, name);
        EOSs.emplace_back(term);
    }
    return EOSs;
}

inline auto build_multifluid_model(const std::vector<std::string>& components, const std::string& coolprop_root, const std::string& BIPcollectionpath, const nlohmann::json& flags = {}) {

    const auto BIPcollection = nlohmann::json::parse(std::ifstream(BIPcollectionpath));

    // Pure fluids
    auto [Tc, vc] = MultiFluidReducingFunction::get_Tcvc(coolprop_root, components);
    auto EOSs = get_EOSs(coolprop_root, components); 
    
    // Things related to the mixture
    auto F = MultiFluidReducingFunction::get_F_matrix(BIPcollection, components, flags);
    auto funcs = get_departure_function_matrix(coolprop_root, BIPcollection, components, flags);
    auto [betaT, gammaT, betaV, gammaV] = MultiFluidReducingFunction::get_BIP_matrices(BIPcollection, components, flags);

    auto redfunc = MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc);

    return MultiFluid(
        std::move(redfunc),
        std::move(CorrespondingStatesContribution(std::move(EOSs))),
        std::move(DepartureContribution(std::move(F), std::move(funcs)))
    );
}

/**
This class holds a lightweight reference to the core parts of the model, allowing for the reducing function to be modified
by the user, perhaps for model optimization purposes

The reducing function is moved into this class, while the donor class is used for the remaining bits and pieces 
*/
template<typename ReducingFunction, typename BaseClass>
class MultiFluidReducingFunctionAdapter {

public:
    const BaseClass& base; 
    const ReducingFunction redfunc;

    template<class VecType>
    auto R(const VecType& molefrac) const { return base.R(molefrac); }

    MultiFluidReducingFunctionAdapter(const BaseClass& base, ReducingFunction&& redfunc) : base(base), redfunc(redfunc) {};

    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T,
        const RhoType& rho,
        const MoleFracType& molefrac) const
    {
        auto Tred = forceeval(redfunc.get_Tr(molefrac));
        auto rhored = forceeval(redfunc.get_rhor(molefrac));
        auto delta = forceeval(rho / rhored);
        auto tau = forceeval(Tred / T);
        auto val = base.corr.alphar(tau, delta, molefrac) + base.dep.alphar(tau, delta, molefrac);
        return forceeval(val);
    }
};

template<class Model>
auto build_BIPmodified(Model& model, const nlohmann::json& j) {
    auto red = model.redfunc;
    auto betaT = red.betaT;
    betaT(0, 1) = j["betaT"];
    betaT(1, 0) = 1/betaT(0, 1);
    auto betaV = red.betaV;
    betaV(0, 1) = j["betaV"];
    betaV(1, 0) = 1/betaV(0, 1);
    auto gammaT = red.gammaT, gammaV = red.gammaV;
    gammaT(0, 1) = j["gammaT"]; gammaT(1, 0) = gammaT(0, 1);
    gammaV(0, 1) = j["gammaV"]; gammaV(1, 0) = gammaV(0, 1);
    auto Tc = red.Tc, vc = red.vc;
    auto newred = MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc);
    return MultiFluidReducingFunctionAdapter(model, std::move(newred));
}

/**
This class holds a lightweight reference to the core parts of the model

The reducing and departure functions are moved into this class, while the donor class is used for the corresponding states portion
*/
template<typename ReducingFunction, typename DepartureFunction, typename BaseClass>
class MultiFluidAdapter {

public:
    const BaseClass& base;
    const ReducingFunction redfunc;
    const DepartureFunction depfunc;

    template<class VecType>
    auto R(const VecType& molefrac) const { return base.R(molefrac); }

    MultiFluidAdapter(const BaseClass& base, ReducingFunction&& redfunc, DepartureFunction &&depfunc) : base(base), redfunc(redfunc), depfunc(depfunc) {};

    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T,
        const RhoType& rho,
        const MoleFracType& molefrac) const
    {
        auto Tred = forceeval(redfunc.get_Tr(molefrac));
        auto rhored = forceeval(redfunc.get_rhor(molefrac));
        auto delta = forceeval(rho / rhored);
        auto tau = forceeval(Tred / T);
        auto val = base.corr.alphar(tau, delta, molefrac) + depfunc.alphar(tau, delta, molefrac);
        return forceeval(val);
    }
};

template<class Model>
auto build_multifluid_mutant(Model& model, const nlohmann::json& j) {

    auto red = model.redfunc;
    auto betaT = red.betaT;
    betaT(0, 1) = j["betaT"];
    betaT(1, 0) = 1 / betaT(0, 1);
    auto betaV = red.betaV;
    betaV(0, 1) = j["betaV"];
    betaV(1, 0) = 1 / betaV(0, 1);
    auto gammaT = red.gammaT, gammaV = red.gammaV;
    gammaT(0, 1) = j["gammaT"]; gammaT(1, 0) = gammaT(0, 1);
    gammaV(0, 1) = j["gammaV"]; gammaV(1, 0) = gammaV(0, 1);
    auto Tc = red.Tc, vc = red.vc;
    auto newred = MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc);

    if (j.contains("Fij") && j["Fij"] != 0.0) {
        throw std::invalid_argument("Don't support Fij != 0 for now");
    }
    auto N = 2;
    // Allocate the matrix with default models
    Eigen::MatrixXd F(2, 2); F.setZero();
    std::vector<std::vector<MultiFluidDepartureFunction>> funcs(N); 
    for (auto i = 0; i < funcs.size(); ++i) {
        funcs[i].resize(funcs.size());
        for (auto j = 0; j < N; ++j) {
            funcs[i][j].set_type("none");
        }
    }
    auto newdep = DepartureContribution(std::move(F), std::move(funcs));

    return MultiFluidAdapter(model, std::move(newred), std::move(newdep));
}


class DummyEOS {
public:
    template<typename TType, typename RhoType> auto alphar(TType tau, const RhoType& delta) const { return tau * delta; }
};
class DummyReducingFunction {
public:
    template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return molefracs[0]; }
    template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return molefracs[0]; }
};
inline auto build_dummy_multifluid_model(const std::vector<std::string>& components) {
    std::vector<DummyEOS> EOSs(2);
    std::vector<std::vector<DummyEOS>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }
    std::vector<std::vector<double>> F(2); for (auto i = 0; i < F.size(); ++i) { F[i].resize(F.size()); }

    struct Fwrapper {
    private: 
        const std::vector<std::vector<double>> F_;
    public:
        Fwrapper(const std::vector<std::vector<double>> &F) : F_(F){};
        auto operator ()(std::size_t i, std::size_t j) const{ return F_[i][j]; }
    };
    auto ff = Fwrapper(F);
    auto redfunc = DummyReducingFunction();
    return MultiFluid(std::move(redfunc), std::move(CorrespondingStatesContribution(std::move(EOSs))), std::move(DepartureContribution(std::move(ff), std::move(funcs))));
}
inline void test_dummy() {
    auto model = build_dummy_multifluid_model({ "A", "B" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    auto alphar = model.alphar(300.0, rhovec);
}