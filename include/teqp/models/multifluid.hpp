#pragma once

#include "nlohmann/json.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <cmath>
#include <optional>
#include <variant>
#include <filesystem>

#include "teqp/derivs.hpp"
#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/filesystem.hpp"
#include "teqp/json_tools.hpp"
#include "teqp/exceptions.hpp"

#include "MultiComplex/MultiComplex.hpp"

#include "multifluid_eosterms.hpp"
#include "multifluid_reducing.hpp"

#include <boost/algorithm/string/join.hpp>

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

namespace teqp{

/// Load a JSON file from a specified file
inline nlohmann::json load_a_JSON_file(const std::string& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument("Path to be loaded does not exist: " + path);
    }
    auto stream = std::ifstream(path);
    if (!stream) {
        throw std::invalid_argument("File stream cannot be opened from: " + path);
    }
    try {
        return nlohmann::json::parse(stream);
    }
    catch (...) {
        throw std::invalid_argument("File at " + path + " is not valid JSON");
    }
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
        return forceeval(alphar);
    }

    template<typename TauType, typename DeltaType>
    auto alphari(const TauType& tau, const DeltaType& delta, std::size_t i) const {
        return EOSs[i].alphar(tau, delta);
    }

    auto get_EOS(std::size_t i) const{
        return EOSs[i];
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
        return forceeval(alphar);
    }

    /// Call a single departure term at i,j 
    template<typename TauType, typename DeltaType>
    auto get_alpharij(const int i, const int j,     const TauType& tau, const DeltaType& delta) const {
        int N = funcs.size();
        if (i < 0 || j < 0){
            throw teqp::InvalidArgument("i or j is negative");
        }
        if (i >= N || j >= N){
            throw teqp::InvalidArgument("i or j is invalid; size is " + std::to_string(N));
        }
        return forceeval(funcs[i][j].alphar(tau, delta));
    }
};

template<typename CorrespondingTerm, typename DepartureTerm>
class MultiFluid {  

private:
    std::string meta = ""; ///< A string that can be used to store arbitrary metadata as needed
public:
    const ReducingFunctions redfunc;
    const CorrespondingTerm corr;
    const DepartureTerm dep;

    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    /// Store some sort of metadata in string form (perhaps a JSON representation of the model?)
    void set_meta(const std::string& m) { meta = m; }
    /// Get the metadata stored in string form
    auto get_meta() const { return meta; }

    MultiFluid(ReducingFunctions&& redfunc, CorrespondingTerm&& corr, DepartureTerm&& dep) : redfunc(redfunc), corr(corr), dep(dep) {};

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


/***
* \brief Get the JSON data structure for a given departure function
* \param name The name (or alias) of the departure function to be looked up
* \parm path The root path to the fluid data, or alternatively, the path to the json file directly
*/
inline auto get_departure_json(const std::string& name, const std::string& path) {
    std::string filepath = std::filesystem::is_regular_file(path) ? path : path + "/dev/mixtures/mixture_departure_functions.json";
    nlohmann::json j = load_a_JSON_file(filepath);
    std::string js = j.dump(2);
    // First pass, direct name lookup
    for (auto& el : j) {
        if (el.at("Name") == name) {
            return el;
        }
    }
    // Second pass, iterate over aliases
    for (auto& el : j) {
        for (auto &alias : el.at("aliases")) {
            if (alias == name) {
                return el;
            }
        }
    }
    throw std::invalid_argument("Could not match the name: " + name + "when looking up departure function");
}
    
inline auto build_departure_function(const nlohmann::json& j) {
    auto build_power = [&](auto term, auto& dep) {
        std::size_t N = term["n"].size();

        // Don't add a departure function if there are no coefficients provided
        if (N == 0) {
            return;
        }

        PowerEOSTerm eos;

        auto eigorzero = [&term, &N](const std::string& name) -> Eigen::ArrayXd {
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

        int Nlzero = 0, Nlnonzero = 0;
        bool contiguous_lzero;

        if (term["l"].empty()) {
            // exponential part not included
            l.setZero();
            if (!all_same_length(term, { "n","t","d" })) {
                throw std::invalid_argument("Lengths are not all identical in polynomial-like term");
            }
        }
        else {
            if (!all_same_length(term, { "n","t","d","l"})) {
                throw std::invalid_argument("Lengths are not all identical in exponential term");
            }
            l = toeig(term["l"]);
            // l is included, use it to build c; c_i = 1 if l_i > 0, zero otherwise
            for (auto i = 0; i < c.size(); ++i) {
                if (l[i] > 0) {
                    c[i] = 1.0;
                }
            }

            // See how many of the first entries have zero values for l_i
            contiguous_lzero = (l[0] == 0);
            for (auto i = 0; i < c.size(); ++i) {
                if (l[i] == 0) {
                    Nlzero++;
                }
            }
        }
        Nlnonzero = l.size() - Nlzero;

        if ((l[0] != 0) && (l[l.size() - 1] == 0)) {
            throw std::invalid_argument("If l_i has zero and non-zero values, the zero values need to come first");
        }

        eos.c = c;
        eos.l = l;

        eos.l_i = eos.l.cast<int>();

        if (Nlzero + Nlnonzero != l.size()) {
            throw std::invalid_argument("Somehow the l lengths don't add up");
        }


        if (((eos.l_i.cast<double>() - eos.l).cwiseAbs() > 0.0).any()) {
            throw std::invalid_argument("Non-integer entry in l found");
        }

        // If a contiguous portion of the terms have values of l_i that are zero
        // it is computationally advantageous to break up the evaluation into 
        // part that has just the n_i*tau^t_i*delta^d_i and the part with the
        // exponential term exp(-delta^l_i)
        if (l.sum() == 0) {
            // No l term at all, just polynomial
            JustPowerEOSTerm poly;
            poly.n = eos.n;
            poly.t = eos.t;
            poly.d = eos.d;
            dep.add_term(poly);
        }
        else if (l.sum() > 0 && contiguous_lzero){
            JustPowerEOSTerm poly; 
            poly.n = eos.n.head(Nlzero);
            poly.t = eos.t.head(Nlzero);
            poly.d = eos.d.head(Nlzero);
            dep.add_term(poly);

            PowerEOSTerm e;
            e.n = eos.n.tail(Nlnonzero);
            e.t = eos.t.tail(Nlnonzero);
            e.d = eos.d.tail(Nlnonzero);
            e.c = eos.c.tail(Nlnonzero);
            e.l = eos.l.tail(Nlnonzero);
            e.l_i = eos.l_i.tail(Nlnonzero);
            dep.add_term(e);
        }
        else {
            // Don't try to get too clever, just add the departure term
            dep.add_term(eos);
        }
    };

    auto build_doubleexponential = [&](auto& term, auto& dep) {
        if (!all_same_length(term, { "n","t","d","ld","gd","lt","gt" })) {
            throw std::invalid_argument("Lengths are not all identical in double exponential term");
        }
        DoubleExponentialEOSTerm eos;
        eos.n = toeig(term.at("n"));
        eos.t = toeig(term.at("t"));
        eos.d = toeig(term.at("d"));
        eos.ld = toeig(term.at("ld"));
        eos.gd = toeig(term.at("gd"));
        eos.lt = toeig(term.at("lt"));
        eos.gt = toeig(term.at("gt"));
        eos.ld_i = eos.ld.cast<int>();
        dep.add_term(eos);
    }; 
    auto build_Chebyshev2D = [&](auto& term, auto& dep) {
        Chebyshev2DEOSTerm eos;
        int Ntau = term.at("Ntau"); // Degree in tau (there will be Ntau+1 coefficients in the tau direction)
        int Ndelta = term.at("Ndelta"); // Degree in delta (there will be Ndelta+1 coefficients in the delta direction)
        Eigen::ArrayXd c = toeig(term.at("a"));
        if ((Ntau + 1)*(Ndelta + 1) != c.size()){
            throw std::invalid_argument("Provided length [" + std::to_string(c.size()) + "] is not equal to (Ntau+1)*(Ndelta+1)");
        }
        eos.a = c.reshaped(Ntau+1, Ndelta+1).eval(); // All in one long array, then reshaped
        eos.taumin = term.at("taumin");
        eos.taumax = term.at("taumax");
        eos.deltamin = term.at("deltamin");
        eos.deltamax = term.at("deltamax");
        dep.add_term(eos);
    };
    //auto build_gaussian = [&](auto& term) {
    //    GaussianEOSTerm eos;
    //    eos.n = toeig(term["n"]);
    //    eos.t = toeig(term["t"]);
    //    eos.d = toeig(term["d"]);
    //    eos.eta = toeig(term["eta"]);
    //    eos.beta = toeig(term["beta"]);
    //    eos.gamma = toeig(term["gamma"]);
    //    eos.epsilon = toeig(term["epsilon"]);
    //    if (!all_same_length(term, { "n","t","d","eta","beta","gamma","epsilon" })) {
    //        throw std::invalid_argument("Lengths are not all identical in Gaussian term");
    //    }
    //    return eos;
    //};
    auto build_GERG2004 = [&](const auto& term, auto& dep) {
        if (!all_same_length(term, { "n","t","d","eta","beta","gamma","epsilon" })) {
            throw std::invalid_argument("Lengths are not all identical in GERG term");
        }
        int Npower = term["Npower"];
        auto NGERG = static_cast<int>(term["n"].size()) - Npower;

        PowerEOSTerm eos;
        eos.n = toeig(term["n"]).head(Npower);
        eos.t = toeig(term["t"]).head(Npower);
        eos.d = toeig(term["d"]).head(Npower);
        if (term.contains("l")) {
            eos.l = toeig(term["l"]).head(Npower);
        }
        else {
            eos.l = 0.0 * eos.n;
        }
        eos.c = (eos.l > 0).cast<int>().cast<double>();
        eos.l_i = eos.l.cast<int>();
        dep.add_term(eos);

        GERG2004EOSTerm e;
        e.n = toeig(term["n"]).tail(NGERG);
        e.t = toeig(term["t"]).tail(NGERG);
        e.d = toeig(term["d"]).tail(NGERG);
        e.eta = toeig(term["eta"]).tail(NGERG);
        e.beta = toeig(term["beta"]).tail(NGERG);
        e.gamma = toeig(term["gamma"]).tail(NGERG);
        e.epsilon = toeig(term["epsilon"]).tail(NGERG);
        dep.add_term(e);
    };
    auto build_GaussianExponential = [&](const auto& term, auto& dep) {
        if (!all_same_length(term, { "n","t","d","eta","beta","gamma","epsilon" })) {
            throw std::invalid_argument("Lengths are not all identical in Gaussian+Exponential term");
        }
        int Npower = term["Npower"];
        auto NGauss = static_cast<int>(term["n"].size()) - Npower;

        PowerEOSTerm eos;
        eos.n = toeig(term["n"]).head(Npower);
        eos.t = toeig(term["t"]).head(Npower);
        eos.d = toeig(term["d"]).head(Npower);
        if (term.contains("l")) {
            eos.l = toeig(term["l"]).head(Npower);
        }
        else {
            eos.l = 0.0 * eos.n;
        }
        eos.c = (eos.l > 0).cast<int>().cast<double>();
        eos.l_i = eos.l.cast<int>();
        dep.add_term(eos);

        GaussianEOSTerm e;
        e.n = toeig(term["n"]).tail(NGauss);
        e.t = toeig(term["t"]).tail(NGauss);
        e.d = toeig(term["d"]).tail(NGauss);
        e.eta = toeig(term["eta"]).tail(NGauss);
        e.beta = toeig(term["beta"]).tail(NGauss);
        e.gamma = toeig(term["gamma"]).tail(NGauss);
        e.epsilon = toeig(term["epsilon"]).tail(NGauss);
        dep.add_term(e);
    };

    std::string type = j.at("type");
    DepartureTerms dep;
    if (type == "Exponential") {
        build_power(j, dep);
    }
    else if (type == "DoubleExponential") {
        build_doubleexponential(j, dep);
    }
    else if (type == "GERG-2004" || type == "GERG-2008") {
        build_GERG2004(j, dep);
    }
    else if (type == "Gaussian+Exponential") {
        build_GaussianExponential(j, dep);
    }
    else if (type == "Chebyshev2D") {
        build_Chebyshev2D(j, dep);
    }
    else if (type == "none") {
        dep.add_term(NullEOSTerm());
    }
    else {
        
        std::vector<std::string> options = { "Exponential","GERG-2004","GERG-2008","Gaussian+Exponential", "none", "DoubleExponential","Chebyshev2D"};
        throw std::invalid_argument("Bad departure term type: " + type + ". Options are {" + boost::algorithm::join(options, ",") + "}");
    }
    return dep;
}

inline auto get_departure_function_matrix(const nlohmann::json& depcollection, const nlohmann::json& BIPcollection, const std::vector<std::string>& components, const nlohmann::json& flags) {

    // Allocate the matrix with default models
    std::vector<std::vector<DepartureTerms>> funcs(components.size()); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }

    // Load the collection of data on departure functions

    auto get_departure_json = [&depcollection](const std::string& Name) {
        for (auto& el : depcollection) {
            if (el["Name"] == Name) { return el; }
        }
        throw std::invalid_argument("Bad departure function name: "+Name);
    };

    auto funcsmeta = nlohmann::json::object();

    for (auto i = 0; i < funcs.size(); ++i) {
        std::string istr = std::to_string(i);
        if (funcsmeta.contains(istr)) { funcsmeta[istr] = {}; }
        for (auto j = i + 1; j < funcs.size(); ++j) {
            std::string jstr = std::to_string(j);
            auto [BIP, swap_needed] = reducing::get_BIPdep(BIPcollection, { components[i], components[j] }, flags);
            std::string funcname = BIP.contains("function") ? BIP["function"] : "";
            if (!funcname.empty()) {
                auto jj = get_departure_json(funcname);
                funcsmeta[istr][jstr] = { {"departure", jj}, {"BIP", BIP} };
                funcsmeta[istr][jstr]["BIP"]["swap_needed"] = swap_needed;
                funcs[i][j] = build_departure_function(jj);
                funcs[j][i] = build_departure_function(jj);
            }
            else {
                funcs[i][j].add_term(NullEOSTerm());
                funcs[j][i].add_term(NullEOSTerm());
            }
        }
    }
    return std::make_tuple(funcs, funcsmeta);
}

inline auto get_EOS_terms(const nlohmann::json& j)
{
    auto alphar = j["EOS"][0]["alphar"];

    const std::vector<std::string> allowed_types = { "ResidualHelmholtzPower", "ResidualHelmholtzGaussian", "ResidualHelmholtzNonAnalytic","ResidualHelmholtzGaoB", "ResidualHelmholtzLemmon2005", "ResidualHelmholtzExponential" };

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

    EOSTerms container;

    auto build_power = [&](auto term, auto & container) {
        std::size_t N = term["n"].size();

        PowerEOSTerm eos;

        auto eigorzero = [&term, &N](const std::string& name) -> Eigen::ArrayXd {
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
        int Nlzero = 0, Nlnonzero = 0;
        bool contiguous_lzero;
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

            // See how many of the first entries have zero values for l_i
            contiguous_lzero = (l[0] == 0);
            for (auto i = 0; i < c.size(); ++i) {
                if (l[i] == 0) {
                    Nlzero++;
                }
            }
        }
        Nlnonzero = l.size() - Nlzero;
        
        eos.c = c;
        eos.l = l;

        eos.l_i = eos.l.cast<int>();

        if (Nlzero + Nlnonzero != l.size()) {
            throw std::invalid_argument("Somehow the l lengths don't add up");
        }

        if (((eos.l_i.cast<double>() - eos.l).cwiseAbs() > 0.0).any()) {
            throw std::invalid_argument("Non-integer entry in l found");
        }
        
        // If a contiguous portion of the terms have values of l_i that are zero
        // it is computationally advantageous to break up the evaluation into 
        // part that has just the n_i*tau^t_i*delta^d_i and the part with the
        // exponential term exp(-delta^l_i)
        if (l.sum() == 0) {
            // No l term at all, just polynomial
            JustPowerEOSTerm poly;
            poly.n = eos.n;
            poly.t = eos.t;
            poly.d = eos.d;
            container.add_term(poly);
        }
        else if (l.sum() > 0 && contiguous_lzero) {
            JustPowerEOSTerm poly;
            poly.n = eos.n.head(Nlzero);
            poly.t = eos.t.head(Nlzero);
            poly.d = eos.d.head(Nlzero);
            container.add_term(poly);

            PowerEOSTerm e;
            e.n = eos.n.tail(Nlnonzero);
            e.t = eos.t.tail(Nlnonzero);
            e.d = eos.d.tail(Nlnonzero);
            e.c = eos.c.tail(Nlnonzero);
            e.l = eos.l.tail(Nlnonzero);
            e.l_i = eos.l_i.tail(Nlnonzero);
            container.add_term(e);
        }
        else {
            // Don't try to get too clever, just add the term
            container.add_term(eos);
        }
    };

    auto build_Lemmon2005 = [&](auto term) {
        Lemmon2005EOSTerm eos;
        eos.n = toeig(term["n"]);
        eos.t = toeig(term["t"]);
        eos.d = toeig(term["d"]);
        eos.m = toeig(term["m"]);
        eos.l = toeig(term["l"]);
        eos.l_i = eos.l.cast<int>();
        if (!all_same_length(term, { "n","t","d","m","l" })) {
            throw std::invalid_argument("Lengths are not all identical in Lemmon2005 term");
        }
        if (((eos.l_i.cast<double>() - eos.l).cwiseAbs() > 0.0).any()) {
            throw std::invalid_argument("Non-integer entry in l found");
        }
        return eos;
    };

    auto build_gaussian = [&](auto term) {
        GaussianEOSTerm eos;
        eos.n = toeig(term["n"]);
        eos.t = toeig(term["t"]);
        eos.d = toeig(term["d"]);
        eos.eta = toeig(term["eta"]);
        eos.beta = toeig(term["beta"]);
        eos.gamma = toeig(term["gamma"]);
        eos.epsilon = toeig(term["epsilon"]);
        if (!all_same_length(term, { "n","t","d","eta","beta","gamma","epsilon" })) {
            throw std::invalid_argument("Lengths are not all identical in Gaussian term");
        }
        return eos;
    };

    auto build_exponential = [&](auto term) {
        ExponentialEOSTerm eos;
        eos.n = toeig(term["n"]);
        eos.t = toeig(term["t"]);
        eos.d = toeig(term["d"]);
        eos.g = toeig(term["g"]);
        eos.l = toeig(term["l"]);
        eos.l_i = eos.l.cast<int>();
        if (!all_same_length(term, { "n","t","d","g","l" })) {
            throw std::invalid_argument("Lengths are not all identical in exponential term");
        }
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
        if (!all_same_length(term, { "n","t","d","eta","beta","gamma","epsilon","b" })) {
            throw std::invalid_argument("Lengths are not all identical in GaoB term");
        }
        return eos;
    };

    /// lambda function for adding non-analytic terms
    auto build_na = [&](auto& term) {
        NonAnalyticEOSTerm eos;
        eos.n = toeig(term["n"]);
        eos.A = toeig(term["A"]);
        eos.B = toeig(term["B"]);
        eos.C = toeig(term["C"]);
        eos.D = toeig(term["D"]);
        eos.a = toeig(term["a"]);
        eos.b = toeig(term["b"]);
        eos.beta = toeig(term["beta"]);
        if (!all_same_length(term, { "n","A","B","C","D","a","b","beta" })) {
            throw std::invalid_argument("Lengths are not all identical in nonanalytic term");
        }
        return eos;
    };
    
    for (auto& term : alphar) {
        std::string type = term["type"];
        if (type == "ResidualHelmholtzPower") {
            build_power(term, container);
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
        else if (type == "ResidualHelmholtzExponential") {
            container.add_term(build_exponential(term));
        }
        else {
            throw std::invalid_argument("Bad term type: "+type);
        }
    }
    return container;
}

inline auto get_EOSs(const std::vector<nlohmann::json>& pureJSON) {
    std::vector<EOSTerms> EOSs;
    for (auto& j : pureJSON) {
        auto term = get_EOS_terms(j);
        EOSs.emplace_back(term);
    }
    return EOSs;
}

inline auto collect_component_json(const std::vector<std::string>& components, const std::string& root) 
{
    std::vector<nlohmann::json> out;
    for (auto c : components) {
        // First we try to lookup the name as a path, which can be on the filesystem, or relative to the root for default name lookup
        std::vector<std::filesystem::path> candidates = { c, root + "/dev/fluids/" + c + ".json" };
        std::filesystem::path selected_path = "";
        for (auto candidate : candidates) {
            if (std::filesystem::is_regular_file(candidate)) {
                selected_path = candidate;
                break;
            }
        }
        if (selected_path != "") {
            out.push_back(load_a_JSON_file(selected_path.string()));
        }
        else {
            throw std::invalid_argument("Could not load any of the candidates:" + c);
        }
    }
    return out;
}

inline auto collect_identifiers(const std::vector<nlohmann::json>& pureJSON)
{
    std::vector<std::string> CAS, Name, REFPROP;
    for (auto j : pureJSON) {
        Name.push_back(j.at("INFO").at("NAME"));
        CAS.push_back(j.at("INFO").at("CAS"));
        REFPROP.push_back(j.at("INFO").at("REFPROP_NAME"));
    }
    return std::map<std::string, std::vector<std::string>>{
        {"CAS", CAS},
        {"Name", Name},
        {"REFPROP", REFPROP}
    };
}

/// Iterate over the possible options for identifiers to determine which one will satisfy all the binary pairs
template<typename mapvecstring>
inline auto select_identifier(const nlohmann::json& BIPcollection, const mapvecstring& identifierset, const nlohmann::json& flags){
    for (const auto &ident: identifierset){
        std::string key; std::vector<std::string> identifiers;
        std::tie(key, identifiers) = ident;
        try{
            for (auto i = 0; i < identifiers.size(); ++i){
                for (auto j = i+1; j < identifiers.size(); ++j){
                    const std::vector<std::string> pair = {identifiers[i], identifiers[j]};
                    reducing::get_BIPdep(BIPcollection, pair, flags);
                }
            }
            return key;
        }
        catch(...){
            
        }
    }
    throw std::invalid_argument("Unable to match any of the identifier options");
}

/// Build a reverse-lookup map for finding a fluid JSON structure given a backup identifier
inline auto build_alias_map(const std::string& root) {
    std::map<std::string, std::string> aliasmap;
    for (auto path : get_files_in_folder(root + "/dev/fluids", ".json")) {
        auto j = load_a_JSON_file(path.string());
        std::string REFPROP_name = j.at("INFO").at("REFPROP_NAME"); 
        std::string name = j.at("INFO").at("NAME");
        for (std::string k : {"NAME", "CAS", "REFPROP_NAME"}) {
            std::string val = j.at("INFO").at(k);
            // Skip REFPROP names that match the fluid itself
            if (k == "REFPROP_NAME" && val == name) {
                continue;
            }
            // Skip invalid REFPROP names
            if (k == "REFPROP_NAME" && val == "N/A") {
                continue;
            }
            if (aliasmap.count(val) > 0) {
                throw std::invalid_argument("Duplicated reverse lookup identifier ["+k+"] found in file:" + path.string());
            }
            else {
                aliasmap[val] = std::filesystem::absolute(path).string();
            }
        }
        std::vector<std::string> aliases = j.at("INFO").at("ALIASES");
        
        for (std::string alias : aliases) {
            if (alias != REFPROP_name && alias != name) { // Don't add REFPROP name or base name, were already above to list of aliases
                if (aliasmap.count(alias) > 0) {
                    throw std::invalid_argument("Duplicated alias [" + alias + "] found in file:" + path.string());
                }
                else {
                    aliasmap[alias] = std::filesystem::absolute(path).string();
                }
            }
        }
    }
    return aliasmap;
}

/// Internal method for actually constructing the model with the provided JSON data structures
inline auto _build_multifluid_model(const std::vector<nlohmann::json> &pureJSON, const nlohmann::json& BIPcollection, const nlohmann::json& depcollection, const nlohmann::json& flags = {}) {

    auto [Tc, vc] = reducing::get_Tcvc(pureJSON);
    auto EOSs = get_EOSs(pureJSON);

    // Extract the set of possible identifiers to be used to match parameters
    auto identifierset = collect_identifiers(pureJSON);
    // Decide which identifier is to be used (Name, CAS, REFPROP name)
    auto identifiers = identifierset[select_identifier(BIPcollection, identifierset, flags)];

    // Things related to the mixture
    auto F = reducing::get_F_matrix(BIPcollection, identifiers, flags);
    auto [funcs, funcsmeta] = get_departure_function_matrix(depcollection, BIPcollection, identifiers, flags);
    auto [betaT, gammaT, betaV, gammaV] = reducing::get_BIP_matrices(BIPcollection, identifiers, flags, Tc, vc);

    nlohmann::json meta = {
        {"pures", pureJSON},
        {"mix", funcsmeta},
    };

    auto redfunc = ReducingFunctions(std::move(MultiFluidReducingFunction(betaT, gammaT, betaV, gammaV, Tc, vc)));

    auto model = MultiFluid(
        std::move(redfunc),
        CorrespondingStatesContribution(std::move(EOSs)),
        DepartureContribution(std::move(F), std::move(funcs))
    );
    model.set_meta(meta.dump(1));
    return model;
}

/// A builder function where the JSON-formatted strings are provided explicitly rather than file paths
inline auto build_multifluid_JSONstr(const std::vector<std::string>& componentJSON, const std::string& BIPJSON, const std::string& departureJSON, const nlohmann::json& flags = {}) {

    // Mixture things
    const auto BIPcollection = nlohmann::json::parse(BIPJSON);
    const auto depcollection = nlohmann::json::parse(departureJSON);

    // Pure fluids
    std::vector<nlohmann::json> pureJSON;
    for (auto& c : componentJSON) {
        pureJSON.emplace_back(nlohmann::json::parse(c));
    }
    return _build_multifluid_model(pureJSON, BIPcollection, depcollection, flags);
}

inline auto build_multifluid_model(const std::vector<std::string>& components, const std::string& coolprop_root, const std::string& BIPcollectionpath = {}, const nlohmann::json& flags = {}, const std::string& departurepath = {}) {

    std::string BIPpath = (BIPcollectionpath.empty()) ? coolprop_root + "/dev/mixtures/mixture_binary_pairs.json" : BIPcollectionpath;
    const auto BIPcollection = load_a_JSON_file(BIPpath);

    std::string deppath = (departurepath.empty()) ? coolprop_root + "/dev/mixtures/mixture_departure_functions.json" : departurepath;
    const auto depcollection = load_a_JSON_file(deppath);

    // Pure fluids
    std::vector<nlohmann::json> pureJSON;
    try {
        // Try the normal lookup, matching component name to a file in dev/fluids (case sensitive match on linux!)
        pureJSON = collect_component_json(components, coolprop_root);
    }
    catch(...){
        // Lookup the absolute paths for each component
        auto aliasmap = build_alias_map(coolprop_root);
        std::vector<std::string> abspaths;
        for (auto c : components) {
            // Allow matching of absolute paths first
            if (std::filesystem::is_regular_file(c)) {
                abspaths.push_back(c);
            }
            else {
                abspaths.push_back(aliasmap[c]);
            }
        }
        // Backup lookup with absolute paths resolved for each component
        pureJSON = collect_component_json(abspaths, coolprop_root);
    }
    return _build_multifluid_model(pureJSON, BIPcollection, depcollection, flags);
}

/**
* \brief Load a model from a JSON data structure
* 
* Required fields are: components, BIP, departure
* 
* BIP and departure can be either the data in JSON format, or a path to file with those contents
* components is an array, which either contains the paths to the JSON data, or the file path
*/
inline auto multifluidfactory(const nlohmann::json& spec) {
    
    auto JSON_from_path_or_contents = [](const nlohmann::json &path_or_contents) -> nlohmann::json {
        if (path_or_contents.is_string()) {
            return load_a_JSON_file(path_or_contents.get<std::string>());
        }
        else {
            return path_or_contents;
        }
    };

    auto components = spec.at("components"); 
    auto depcollection = JSON_from_path_or_contents(spec.at("departure"));
    auto BIPcollection = JSON_from_path_or_contents(spec.at("BIP"));
    nlohmann::json flags = (spec.contains("flags")) ? spec.at("flags") : nlohmann::json();

    // Pure components
    std::vector<nlohmann::json> pureJSON;
    for (auto c : components) {
        pureJSON.push_back(JSON_from_path_or_contents(c));
    }

    return _build_multifluid_model(pureJSON, BIPcollection, depcollection, flags);
}
/// An overload of multifluidfactory that takes in a string
inline auto multifluidfactory(const std::string& specstring) {
    return multifluidfactory(nlohmann::json::parse(specstring));
}



//class DummyEOS {
//public:
//    template<typename TType, typename RhoType> auto alphar(TType tau, const RhoType& delta) const { return tau * delta; }
//};
//class DummyReducingFunction {
//public:
//    template<typename MoleFractions> auto get_Tr(const MoleFractions& molefracs) const { return molefracs[0]; }
//    template<typename MoleFractions> auto get_rhor(const MoleFractions& molefracs) const { return molefracs[0]; }
//};
//inline auto build_dummy_multifluid_model(const std::vector<std::string>& components) {
//    std::vector<DummyEOS> EOSs(2);
//    std::vector<std::vector<DummyEOS>> funcs(2); for (auto i = 0; i < funcs.size(); ++i) { funcs[i].resize(funcs.size()); }
//    std::vector<std::vector<double>> F(2); for (auto i = 0; i < F.size(); ++i) { F[i].resize(F.size()); }
//
//    struct Fwrapper {
//    private: 
//        const std::vector<std::vector<double>> F_;
//    public:
//        Fwrapper(const std::vector<std::vector<double>> &F) : F_(F){};
//        auto operator ()(std::size_t i, std::size_t j) const{ return F_[i][j]; }
//    };
//    auto ff = Fwrapper(F);
//    auto redfunc = DummyReducingFunction();
//    return MultiFluid(std::move(redfunc), std::move(CorrespondingStatesContribution(std::move(EOSs))), std::move(DepartureContribution(std::move(ff), std::move(funcs))));
//}
//inline void test_dummy() {
//    auto model = build_dummy_multifluid_model({ "A", "B" });
//    std::valarray<double> rhovec = { 1.0, 2.0 };
//    auto alphar = model.alphar(300.0, rhovec);
//}

}; // namespace teqp
