#pragma once

#include "json.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <cmath>
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
        using resulttype = std::remove_const<decltype(forceeval(tau* delta* molefracs[0]))>::type; // Type promotion, without the const-ness
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
        using resulttype = std::remove_const<decltype(forceeval(tau* delta* molefracs[0]))>::type; // Type promotion, without the const-ness
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

    const double R = get_R_gas<double>();

    MultiFluid(ReducingFunction&& redfunc, CorrespondingTerm&& corr, DepartureTerm&& dep) : redfunc(redfunc), corr(corr), dep(dep) {};

    template<typename TType, typename RhoType>
    auto alphar(TType T,
        const RhoType& rhovec,
        const std::optional<typename RhoType::value_type> rhotot = std::nullopt) const
    {
        RhoType::value_type rhotot_ = (rhotot.has_value()) ? rhotot.value() : std::accumulate(std::begin(rhovec), std::end(rhovec), (decltype(rhovec[0]))0.0);
        auto molefrac = rhovec / rhotot_;
        return alphar(T, rhotot_, molefrac);
    }

    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType &T,
        const RhoType &rho,
        const MoleFracType& molefrac) const
    {
        auto Tred = redfunc.get_Tr(molefrac);
        auto rhored = redfunc.get_rhor(molefrac);
        auto delta = forceeval(rho / rhored);
        auto tau = forceeval(Tred / T);
        auto val = corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
        return val;
    }
};


class MultiFluidReducingFunction {
private:
    Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;
    

    template <typename Num>
    auto cube(Num x) const {
        return x*x*x;
    }
    template <typename Num>
    auto square(Num x) const {
        return x*x;
    }

public:
    Eigen::ArrayXd Tc, vc;

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
        MoleFractions::value_type sum1 = 0.0;
        for (auto i = 0; i < N; ++i) {
            sum1 = sum1 + square(z[i]) * Yc[i];
        }
        
        MoleFractions::value_type sum2 = 0.0;
        for (auto i = 0; i < N-1; ++i){
            for (auto j = i+1; j < N; ++j) {
                sum2 = sum2 + 2.0*z[i]*z[j]*(z[i] + z[j])/(square(beta(i, j))*z[i] + z[j])*Yij(i, j);
            }
        }

        return sum1 + sum2;
    }

    static auto get_BIPdep(const nlohmann::json& collection, const std::vector<std::string>& components) {

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
    static auto get_binary_interaction_double(const nlohmann::json& collection, const std::vector<std::string>& components) {
        auto el = get_BIPdep(collection, components);

        double betaT = el["betaT"], gammaT = el["gammaT"], betaV = el["betaV"], gammaV = el["gammaV"];
        // Backwards order of components, flip beta values
        if (components[0] == el["Name2"] && components[1] == el["Name1"]) {
            betaT = 1.0 / betaT;
            betaV = 1.0 / betaV;
        }
        return std::make_tuple(betaT, gammaT, betaV, gammaV);
    }
    static auto get_BIP_matrices(const nlohmann::json& collection, const std::vector<std::string>& components) {
        Eigen::MatrixXd betaT, gammaT, betaV, gammaV, YT, Yv;
        auto N = components.size();
        betaT.resize(N, N); betaT.setZero();
        gammaT.resize(N, N); gammaT.setZero();
        betaV.resize(N, N); betaV.setZero();
        gammaV.resize(N, N); gammaV.setZero();
        for (auto i = 0; i < N; ++i) {
            for (auto j = i + 1; j < N; ++j) {
                auto [betaT_, gammaT_, betaV_, gammaV_] = get_binary_interaction_double(collection, { components[i], components[j] });
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
            auto j = json::parse(std::ifstream(coolprop_root + "/dev/fluids/" + c + ".json"));
            auto red = j["EOS"][0]["STATES"]["reducing"];
            double Tc_ = red["T"];
            double rhoc_ = red["rhomolar"];
            Tc[i] = Tc_;
            vc[i] = 1.0 / rhoc_;
            i++;
        }
        return std::make_tuple(Tc, vc);
    }
    static auto get_F_matrix(const nlohmann::json& collection, const std::vector<std::string>& components) {
        Eigen::MatrixXd F(components.size(), components.size());
        auto N = components.size();
        for (auto i = 0; i < N; ++i) {
            F(i, i) = 0.0;
            for (auto j = i + 1; j < N; ++j) {
                auto el = get_BIPdep(collection, { components[i], components[j] });
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

auto get_departure_function_matrix(const std::string& coolprop_root, const nlohmann::json& BIPcollection, const std::vector<std::string>& components) {

    // Allocate the matrix with default models
    std::vector<std::vector<MultiFluidDepartureFunction>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }

    auto depcollection = nlohmann::json::parse(std::ifstream(coolprop_root + "/dev/mixtures/mixture_departure_functions.json"));

    auto get_departure_function = [&depcollection](const std::string& Name) {
        for (auto& el : depcollection) {
            if (el["Name"] == Name) { return el; }
        }
        throw std::invalid_argument("Bad argument");
    };

    for (auto i = 0; i < funcs.size(); ++i) {
        for (auto j = i + 1; j < funcs.size(); ++j) {
            auto BIP = MultiFluidReducingFunction::get_BIPdep(BIPcollection, { components[i], components[j] });
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
    if (n < 0){
        using namespace autodiff::detail;
        if constexpr (isDual<T> || isExpr<T> || isNumber<T>) {
            return eval(powi(1.0/x, -n));
        }
        else {
            return powi(static_cast<T>(1.0) / x, -n);
        }
    }
    else if (n == 0)
        return static_cast<T>(1.0);                       // x^0 = 1 even for x == 0
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
auto powIV(const T& x, const Eigen::ArrayXd& e) {
    Eigen::Array<T, Eigen::Dynamic, 1> o(e.size());
    for (auto i = 0; i < e.size(); ++i) {
        auto ei = e[i];
        if constexpr (autodiff::detail::isDual<T>) {
            o[i] = pow(x, ei);
        }
        else {
            if (ei == static_cast<int>(ei)) {
                o[i] = powi(x, ei);
            }
            else {
                o[i] = pow(x, ei);
            }
        }
    }
    return o;
}

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

class MultiFluidEOS {
public:
    enum class types { NOTSETTYPE, GERG2004, GaussianExponential, GaussianExponentialNonAnalytic };
private:
    types type = types::NOTSETTYPE;
public:
    Eigen::ArrayXd n, t, d, c, l, eta, beta, gamma, epsilon;

    Eigen::ArrayXd na_A, na_B, na_C, na_D, na_a, na_b, na_beta, na_n;

    void allocate(std::size_t N) {
        auto go = [&N](Eigen::ArrayXd &v){ v.resize(N); v.setZero(); };
        go(n); go(t); go(d); go(l); go(c); go(eta); go(beta); go(gamma); go(epsilon);
    }

    void allocate_na(std::size_t N) {
        auto go = [&N](Eigen::ArrayXd& v) { v.resize(N); v.setZero(); };
        go(na_A); go(na_B); go(na_C); go(na_D); go(na_a); go(na_b); go(na_beta); go(na_n);
    }

    void set_type(const std::string& kind) {
        if (kind == "GaussianExponential") {
            type = types::GaussianExponential;
        }
        else if (kind == "GaussianExponentialNonAnalytic") {
            type = types::GaussianExponentialNonAnalytic;
        }
        else {
            throw std::invalid_argument("Bad type to set_type:" + kind);
        }
    }

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        switch (type) {
            case types::GaussianExponential:{
                return forceeval((n*exp(t*log(tau) + d*log(delta) - c*powIV(delta, l) - eta*(delta - epsilon).square() - beta * (tau - gamma).square())).sum());
                break;}
            case types::GaussianExponentialNonAnalytic:
                {
                // All the "normal" terms
                auto o1 = (n * exp(t * log(tau) + d * log(delta) - c * powIV(delta, l) - eta * (delta - epsilon).square() - beta * (tau - gamma).square())).sum();
                
                // The non-analytic terms
                auto square = [](auto x) { return x * x; };
                auto delta_min1_sq = square(delta-1.0);
                auto Psi = (exp(-na_C*delta_min1_sq -na_D*square(tau-1.0))).eval();
                const Eigen::ArrayXd k = 1.0/(2.0*na_beta);
                auto theta = ((1.0-tau) + na_A*pow(delta_min1_sq, k)).eval();
                auto Delta = (theta.square() + na_B*pow(delta_min1_sq, na_a)).eval();

                auto o2 = (na_n*pow(Delta, na_b)*delta*Psi).eval().sum();
                
                return forceeval(o1 + o2);
                break;
                }
            default:
                throw -1;
        }
    }
};

auto get_EOS(const std::string& coolprop_root, const std::string& name) 
{
    using namespace nlohmann;
    auto j = json::parse(std::ifstream(coolprop_root + "/dev/fluids/" + name + ".json"));
    auto alphar = j["EOS"][0]["alphar"];

    std::size_t ncoeff_conventional = 0;

    const std::vector<std::string> conventional_types = {"ResidualHelmholtzPower", "ResidualHelmholtzGaussian"};
    const std::vector<std::string> weird_types = { "ResidualHelmholtzNonAnalytic" };

    auto isallowed = [&](const auto &conventional_types, const std::string &name){ 
        for (auto &a : conventional_types){ if (name == a){return true;};} return false;
    };

    for (auto& term : alphar) {
        std::string type = term["type"];
        if (!isallowed(conventional_types, type) & !isallowed(weird_types, type)){
            throw std::invalid_argument("Bad type:" + type);
        }
        else{
            if (isallowed(conventional_types, type)){
                ncoeff_conventional += term["n"].size();
            }
        }
    }
    
    MultiFluidEOS eos; 
    eos.allocate(ncoeff_conventional); // Allocate arrays to the right size for conventional terms, fill with zero
    eos.set_type("GaussianExponential"); // The default, generic formulation
    
    auto toeig = [](const std::vector<double>& v) -> Eigen::ArrayXd { return Eigen::Map<const Eigen::ArrayXd>(&(v[0]), v.size()); };

    /// lambda function for adding non-analytic terms
    auto add_na = [&eos, &toeig](auto &term){
        auto eigorzero = [&term, &toeig](const std::string& name) -> Eigen::ArrayXd {
            return toeig(term[name]);
        };
        eos.na_n = eigorzero("n");
        eos.na_A = eigorzero("A");
        eos.na_B = eigorzero("B");
        eos.na_C = eigorzero("C");
        eos.na_D = eigorzero("D");
        eos.na_a = eigorzero("a");
        eos.na_b = eigorzero("b");
        eos.na_beta = eigorzero("beta");
        eos.set_type("GaussianExponentialNonAnalytic");
    };
    
    std::size_t offset = 0;
    for (auto &term: alphar){
        if (term["type"] == "ResidualHelmholtzNonAnalytic") {
            add_na(term); continue;
        }
        std::size_t N = term["n"].size(); 
        auto eigorzero = [&term, &toeig, &N](const std::string& name) -> Eigen::ArrayXd {
            if (!term[name].empty()) {
                return toeig(term[name]);
            }
            else {
                return Eigen::ArrayXd::Zero(N);
            }
        };

        eos.n.segment(offset, N) = eigorzero("n");
        eos.t.segment(offset, N) = eigorzero("t");
        eos.d.segment(offset, N) = eigorzero("d");
        eos.eta.segment(offset, N) = eigorzero("eta");
        eos.beta.segment(offset, N) = eigorzero("beta");
        eos.gamma.segment(offset, N) = eigorzero("gamma");
        eos.epsilon.segment(offset, N) = eigorzero("epsilon");

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
        eos.c.segment(offset, N) = c;
        eos.l.segment(offset, N) = l;

        offset += N;
    }
    return eos;
}

auto get_EOSs(const std::string& coolprop_root, const std::vector<std::string>& names) {
    std::vector<MultiFluidEOS> EOSs;
    for (auto& name : names) {
        EOSs.emplace_back(get_EOS(coolprop_root, name));
    }
    return EOSs;
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
auto build_dummy_multifluid_model(const std::vector<std::string>& components) {
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
void test_dummy() {
    auto model = build_dummy_multifluid_model({ "A", "B" });
    std::valarray<double> rhovec = { 1.0, 2.0 };
    auto alphar = model.alphar(300.0, rhovec);
}