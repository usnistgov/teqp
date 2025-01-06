#pragma once

/*
Implementations of the canonical cubic equations of state
*/

#include <vector>
#include <variant>
#include <valarray>
#include <optional>

#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/exceptions.hpp"
#include "cubicsuperancillary.hpp"
#include "teqp/json_tools.hpp"
#include "teqp/math/pow_templates.hpp"

#include "nlohmann/json.hpp"

#include <Eigen/Dense>

namespace teqp {

/**
 * \brief The standard alpha function used by Peng-Robinson and SRK
 */
template<typename NumType>
class BasicAlphaFunction {
private:
    NumType Tci, ///< The critical temperature
    mi;  ///< The "m" parameter
public:
    BasicAlphaFunction(NumType Tci, NumType mi) : Tci(Tci), mi(mi) {};
    
    template<typename TType>
    auto operator () (const TType& T) const {
        return forceeval(pow2(forceeval(1.0 + mi * (1.0 - sqrt(T / Tci)))));
    }
};

/**
 * \brief The Twu alpha function used by Peng-Robinson and SRK
 * \f[
 * \alpha_i = \left(\frac{T}{T_{ci}\right)^{c_2(c_1-1)}\exp\left[c_0(1-\left(\frac{T}{T_{ci}\right)^{c_1c_2})\right]
 * \f]
 */
template<typename NumType>
class TwuAlphaFunction {
private:
    NumType Tci; ///< The critical temperature
    Eigen::Array3d c;
public:
    TwuAlphaFunction(NumType Tci, const Eigen::Array3d &c) : Tci(Tci), c(c) {
        if (c.size()!= 3){
            throw teqp::InvalidArgument("coefficients c for Twu alpha function must have length 3");
        }
    };
    template<typename TType>
    auto operator () (const TType& T) const {
        return forceeval(pow(T/Tci,c[2]*(c[1]-1))*exp(c[0]*(1.0-pow(T/Tci, c[1]*c[2]))));
    }
};

/**
 * \brief The Mathias-Copeman alpha function used by Peng-Robinson and SRK
 *
 ** \f[
 * \alpha_i = (1+c_0x + c_1x^2 + c_2x^3)^2
 * \f]
 * with
 * \f[
 * x = 1-\sqrt{\frac{T}{T_{ci}}}
 * \f]
 */
template<typename NumType>
class MathiasCopemanAlphaFunction {
private:
    NumType Tci; ///< The critical temperature
    Eigen::Array3d c;
public:
    MathiasCopemanAlphaFunction(NumType Tci, const Eigen::Array3d &c) : Tci(Tci), c(c) {
        if (c.size()!= 3){
            throw teqp::InvalidArgument("coefficients c for Mathias-Copeman alpha function must have length 3");
        }
    };
    template<typename TType>
    auto operator () (const TType& T) const {
        auto x = 1.0 - sqrt(T/Tci);
        auto paren = 1.0 + c[0]*x + c[1]*x*x + c[2]*x*x*x;
        return forceeval(paren*paren);
    }
};

using AlphaFunctionOptions = std::variant<BasicAlphaFunction<double>, TwuAlphaFunction<double>, MathiasCopemanAlphaFunction<double>>;

template<typename TC>
auto build_alpha_functions(const TC& Tc_K, const nlohmann::json& jalphas){
    std::vector<AlphaFunctionOptions> alphas;
    std::size_t i = 0;
    if (jalphas.size() != Tc_K.size()){
        throw teqp::InvalidArgument("alpha [size "+std::to_string(jalphas.size())+"] must be the same size as components [size "+std::to_string(Tc_K.size())+"]");
    }
    for (auto alpha : jalphas){
        std::string type = alpha.at("type");
        if (type == "Twu"){
            std::valarray<double> c_ = alpha.at("c");
            Eigen::Array3d c = Eigen::Map<Eigen::Array3d>(&(c_[0]), c_.size());
            alphas.emplace_back(TwuAlphaFunction(Tc_K[i], c));
        }
        else if (type == "Mathias-Copeman"){
            std::valarray<double> c_ = alpha.at("c");
            Eigen::Array3d c = Eigen::Map<Eigen::Array3d>(&(c_[0]), c_.size());
            alphas.emplace_back(MathiasCopemanAlphaFunction(Tc_K[i], c));
        }
        else if (type == "PR78"){
            double acentric = alpha.at("acentric");
            double mi = 0.0;
            if (acentric < 0.491) {
                mi = 0.37464 + 1.54226*acentric - 0.26992*pow2(acentric);
            }
            else {
                mi = 0.379642 + 1.48503*acentric -0.164423*pow2(acentric) + 0.016666*pow3(acentric);
            }
            alphas.emplace_back( BasicAlphaFunction(Tc_K[i], mi));
        }
        else{
            throw teqp::InvalidArgument("alpha type is not understood: "+type);
        }
        i++;
    }
    return alphas;
};

template <typename NumType, typename AlphaFunctions>
class GenericCubic {
protected:
    std::valarray<NumType> ai, bi;
    const NumType Delta1, Delta2, OmegaA, OmegaB;
    int superanc_index;
    const AlphaFunctions alphas;
    Eigen::ArrayXXd kmat;
    
    nlohmann::json meta;
    const double m_R_JmolK;
    
    template<typename TType, typename IndexType>
    auto get_ai(TType /*T*/, IndexType i) const { return ai[i]; }
    
    template<typename TType, typename IndexType>
    auto get_bi(TType /*T*/, IndexType i) const { return bi[i]; }
    
    template<typename IndexType>
    void check_kmat(IndexType N) {
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows [" + std::to_string(kmat.rows()) + "] and columns [" + std::to_string(kmat.cols()) + "] are not identical");
        }
        if (kmat.cols() == 0) {
            kmat.resize(N, N); kmat.setZero();
        }
        else if (static_cast<std::size_t>(kmat.cols()) != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components [" + std::to_string(N) + "]");
        }
    }
    
public:
    GenericCubic(NumType Delta1, NumType Delta2, NumType OmegaA, NumType OmegaB, int superanc_index, const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa, const AlphaFunctions& alphas, const Eigen::ArrayXXd& kmat, const double R_JmolK)
    : Delta1(Delta1), Delta2(Delta2), OmegaA(OmegaA), OmegaB(OmegaB), superanc_index(superanc_index), alphas(alphas), kmat(kmat), m_R_JmolK(R_JmolK)
    {
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0U; i < Tc_K.size(); ++i) {
            ai[i] = OmegaA * pow2(m_R_JmolK * Tc_K[i]) / pc_Pa[i];
            bi[i] = OmegaB * m_R_JmolK * Tc_K[i] / pc_Pa[i];
        }
        check_kmat(ai.size());
    };
    
    void set_meta(const nlohmann::json& j) { meta = j; }
    auto get_meta() const { return meta; }
    auto get_kmat() const { return kmat; }
    
    /// Return a tuple of saturated liquid and vapor densities for the EOS given the temperature
    /// Uses the superancillary equations from Bell and Deiters:
    /// \param T Temperature
    /// \param ifluid Must be provided in the case of mixtures
    auto superanc_rhoLV(double T, std::optional<std::size_t> ifluid = std::nullopt) const {
        
        std::valarray<double> molefracs(ai.size()); molefracs = 1.0;
        
        // If more than one component, must provide the ifluid argument
        if(ai.size() > 1){
            if (!ifluid){
                throw teqp::InvalidArgument("For mixtures, the argument ifluid must be provided");
            }
            if (ifluid.value() > ai.size()-1){
                throw teqp::InvalidArgument("ifluid must be less than "+std::to_string(ai.size()));
            }
            molefracs = 0.0;
            molefracs[ifluid.value()] = 1.0;
        }
        
        auto b = get_b(T, molefracs);
        auto a = get_a(T, molefracs);
        auto Ttilde = R(molefracs)*T*b/a;
        return std::make_tuple(
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOL_CODE, Ttilde)/b,
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOV_CODE, Ttilde)/b
                               );
    }
    
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const {
        return m_R_JmolK;
    }
    
    template<typename TType, typename CompType>
    auto get_a(TType T, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> a_ = 0.0;
        for (auto i = 0U; i < molefracs.size(); ++i) {
            auto alphai = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[i]));
            auto ai_ = forceeval(this->ai[i] * alphai);
            for (auto j = 0U; j < molefracs.size(); ++j) {
                auto alphaj = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[j]));
                auto aj_ = this->ai[j] * alphaj;
                auto aij = forceeval((1 - kmat(i,j)) * sqrt(ai_ * aj_));
                a_ = a_ + molefracs[i] * molefracs[j] * aij;
            }
        }
        return forceeval(a_);
    }
    
    template<typename TType, typename CompType>
    auto get_b(TType /*T*/, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> b_ = 0.0;
        for (auto i = 0U; i < molefracs.size(); ++i) {
            b_ = b_ + molefracs[i] * bi[i];
        }
        return forceeval(b_);
    }
    
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T,
                const RhoType& rho,
                const MoleFracType& molefrac) const
    {
        if (static_cast<std::size_t>(molefrac.size()) != alphas.size()) {
            throw std::invalid_argument("Sizes do not match");
        }
        auto b = get_b(T, molefrac);
        auto Psiminus = -log(1.0 - b * rho);
        auto Psiplus = log((Delta1 * b * rho + 1.0) / (Delta2 * b * rho + 1.0)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - get_a(T, molefrac) / (m_R_JmolK * T) * Psiplus;
        return forceeval(val);
    }
};

template <typename TCType, typename PCType, typename AcentricType>
auto canonical_SRK(TCType Tc_K, PCType pc_Pa, AcentricType acentric, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt, const std::optional<double> R_JmolK = std::nullopt) {
    double Delta1 = 1;
    double Delta2 = 0;
    AcentricType m = 0.48 + 1.574 * acentric - 0.176 * acentric * acentric;
    
    std::vector<AlphaFunctionOptions> alphas;
    for (auto i = 0U; i < Tc_K.size(); ++i) {
        alphas.emplace_back(BasicAlphaFunction(Tc_K[i], m[i]));
    }
    
    // See https://doi.org/10.1021/acs.iecr.1c00847
    double OmegaA = 1.0 / (9.0 * (cbrt(2) - 1));
    double OmegaB = (cbrt(2) - 1) / 3;
    
    nlohmann::json meta = {
        {"Delta1", Delta1},
        {"Delta2", Delta2},
        {"OmegaA", OmegaA},
        {"OmegaB", OmegaB},
        {"kind", "Soave-Redlich-Kwong"},
        {"R / J/mol/K", R_JmolK.value_or(constants::R_CODATA2017)}
    };
    const std::size_t N = m.size();
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, CubicSuperAncillary::SRK_CODE, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)), R_JmolK.value_or(constants::R_CODATA2017));
    cub.set_meta(meta);
    return cub;
}

/// A JSON-based factory function for the canonical SRK model
inline auto make_canonicalSRK(const nlohmann::json& spec){
    std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
    Eigen::ArrayXXd kmat(0, 0);
    if (spec.contains("kmat")){
        kmat = build_square_matrix(spec.at("kmat"));
    }
    return canonical_SRK(Tc_K, pc_Pa, acentric, kmat);
}

template <typename TCType, typename PCType, typename AcentricType>
auto canonical_PR(TCType Tc_K, PCType pc_Pa, AcentricType acentric, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt, const std::optional<double> R_JmolK = std::nullopt) {
    double Delta1 = 1+sqrt(2.0);
    double Delta2 = 1-sqrt(2.0);
    AcentricType m = acentric*0.0;
    std::vector<AlphaFunctionOptions> alphas;
    for (auto i = 0U; i < Tc_K.size(); ++i) {
        if (acentric[i] < 0.491) {
            m[i] = 0.37464 + 1.54226*acentric[i] - 0.26992*pow2(acentric[i]);
        }
        else {
            m[i] = 0.379642 + 1.48503*acentric[i] -0.164423*pow2(acentric[i]) + 0.016666*pow3(acentric[i]);
        }
        alphas.emplace_back(BasicAlphaFunction(Tc_K[i], m[i]));
    }

    // See https://doi.org/10.1021/acs.iecr.1c00847
    double OmegaA = 0.45723552892138218938;
    double OmegaB = 0.077796073903888455972;

    nlohmann::json meta = {
        {"Delta1", Delta1},
        {"Delta2", Delta2},
        {"OmegaA", OmegaA},
        {"OmegaB", OmegaB},
        {"kind", "Peng-Robinson"},
        {"R / J/mol/K", R_JmolK.value_or(constants::R_CODATA2017)}
    };
    
    const std::size_t N = m.size();
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, CubicSuperAncillary::PR_CODE, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)), R_JmolK.value_or(constants::R_CODATA2017));
    cub.set_meta(meta);
    return cub;
}

/// A JSON-based factory function for the canonical SRK model
inline auto make_canonicalPR(const nlohmann::json& spec){
    std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
    Eigen::ArrayXXd kmat(0, 0);
    if (spec.contains("kmat")){
        kmat = build_square_matrix(spec.at("kmat"));
    }
    return canonical_PR(Tc_K, pc_Pa, acentric, kmat);
}

/// A JSON-based factory function for the generalized cubic + alpha
inline auto make_generalizedcubic(const nlohmann::json& spec){
    // Tc, pc, and acentric factor must always be provided
    std::valarray<double> Tc_K = spec.at("Tcrit / K"),
    pc_Pa = spec.at("pcrit / Pa"),
    acentric = spec.at("acentric");
    
    // If kmat is provided, then collect it
    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat")){
        kmat = build_square_matrix(spec.at("kmat"));
    }
    std::optional<double> R_JmolK;
    if (spec.contains("R / J/mol/K")){
        R_JmolK = spec.at("R / J/mol/K");
    }
    
    int superanc_code = CubicSuperAncillary::UNKNOWN_CODE;
    
    // Build the list of alpha functions, one per component
    std::vector<AlphaFunctionOptions> alphas;
    
    double Delta1, Delta2, OmegaA, OmegaB;
    std::string kind = "custom";
    
    if (spec.at("type") == "PR" ){
        Delta1 = 1+sqrt(2.0);
        Delta2 = 1-sqrt(2.0);
        // See https://doi.org/10.1021/acs.iecr.1c00847
        OmegaA = 0.45723552892138218938;
        OmegaB = 0.077796073903888455972;
        superanc_code = CubicSuperAncillary::PR_CODE;
        kind = "Peng-Robinson";
        
        if (!spec.contains("alpha")){
            for (auto i = 0U; i < Tc_K.size(); ++i) {
                double mi;
                if (acentric[i] < 0.491) {
                    mi = 0.37464 + 1.54226*acentric[i] - 0.26992*pow2(acentric[i]);
                }
                else {
                    mi = 0.379642 + 1.48503*acentric[i] -0.164423*pow2(acentric[i]) + 0.016666*pow3(acentric[i]);
                }
                alphas.emplace_back(BasicAlphaFunction(Tc_K[i], mi));
            }
        }
        else{
            if (!spec["alpha"].is_array()){
                throw teqp::InvalidArgument("alpha must be array of objects");
            }
            alphas = build_alpha_functions(Tc_K, spec.at("alpha"));
        }
    }
    else if (spec.at("type") == "SRK"){
        Delta1 = 1;
        Delta2 = 0;
        if (!spec.contains("alpha")){
            for (auto i = 0U; i < Tc_K.size(); ++i) {
                double mi = 0.48 + 1.574 * acentric[i] - 0.176 * acentric[i] * acentric[i];
                alphas.emplace_back(BasicAlphaFunction(Tc_K[i], mi));
            }
        }
        else{
            if (!spec["alpha"].is_array()){
                throw teqp::InvalidArgument("alpha must be array of objects");
            }
            alphas = build_alpha_functions(Tc_K, spec.at("alpha"));
        }
        // See https://doi.org/10.1021/acs.iecr.1c00847
        OmegaA = 1.0 / (9.0 * (cbrt(2) - 1));
        OmegaB = (cbrt(2) - 1) / 3;
        superanc_code = CubicSuperAncillary::SRK_CODE;
        kind = "Soave-Redlich-Kwong";
    }
    else{
        // Generalized handling of generic cubics (not yet implemented)
        throw teqp::InvalidArgument("Generic cubic EOS are not yet supported (open an issue on github if you want this)");
    }
    
    const std::size_t N = Tc_K.size();
    nlohmann::json meta = {
        {"Delta1", Delta1},
        {"Delta2", Delta2},
        {"OmegaA", OmegaA},
        {"OmegaB", OmegaB},
        {"kind", kind},
        {"R / J/mol/K", R_JmolK.value_or(constants::R_CODATA2017)}
    };
    if (spec.contains("alpha")){
        meta["alpha"] = spec.at("alpha");
    }
    
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, superanc_code, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)), R_JmolK.value_or(constants::R_CODATA2017));
    cub.set_meta(meta);
    return cub;
}

/**
 The quantum corrected Peng-Robinson model as developed in
 
 Ailo Aasen, Morten Hammer, Silvia Lasala, Jean-Noël Jaubert, Øivind Wilhelmsen
 "Accurate quantum-corrected cubic equations of state for helium, neon, hydrogen, deuterium and their mixtures"
 Fluid Phase Equilibria 524 (2020) 112790
 https://doi.org/10.1016/j.fluid.2020.112790
 */
class QuantumCorrectedPR{
    
private:
    const std::vector<double> Tc_K, pc_Pa;
    const std::vector<AlphaFunctionOptions> alphas;
    const std::vector<double> As, Bs, cs_m3mol;
    const Eigen::ArrayXXd kmat, lmat;
    const double Ru;
    
    auto build_alphas(const nlohmann::json& j){
        std::vector<AlphaFunctionOptions> alphas_;
        std::vector<double> L = j.at("Ls"), M = j.at("Ms"), N = j.at("Ns");
        if (L.size() != M.size() || M.size() != N.size()){
            throw teqp::InvalidArgument("L,M,N must all be the same length");
        }
        for (auto i = 0U; i < L.size(); ++i){
            auto coeffs = (Eigen::Array3d() << L[i], M[i], N[i]).finished();
            alphas_.emplace_back(TwuAlphaFunction(Tc_K[i], coeffs));
        }
        return alphas_;
    }
    /// A convenience function to save some typing
    std::vector<double> get_(const nlohmann::json &j, const std::string& k) const { return j.at(k).get<std::vector<double>>(); }
public:
    
    QuantumCorrectedPR(const nlohmann::json &j) : Tc_K(get_(j, "Tcrit / K")), pc_Pa(get_(j, "pcrit / Pa")), alphas(build_alphas(j)), As(get_(j, "As")), Bs(get_(j, "Bs")), cs_m3mol(get_(j, "cs / m^3/mol")), kmat(build_square_matrix(j.at("kmat"))), lmat(build_square_matrix(j.at("lmat"))), Ru(j.value("R / J/mol/K", constants::R_CODATA2017)) {}
    
    
    
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const {
        return Ru;
    }
    auto get_kmat() const { return kmat; }
    auto get_lmat() const { return lmat; }
    auto get_Tc_K() const { return Tc_K; }
    auto get_pc_Pa() const { return pc_Pa; }
    
    template<typename TType>
    auto get_bi(std::size_t i, const TType& T) const {
        auto beta = POW3(1.0 + As[i]/(T+Bs[i]))/POW3(1.0+As[i]/(Tc_K[i]+Bs[i]));
        // See https://doi.org/10.1021/acs.iecr.1c00847 for the exact value: OmegaB = 0.077796073903888455972;
        auto b = 0.07780*Tc_K[i]*Ru/pc_Pa[i];
        return forceeval(b*beta);
    }
    
    template<typename TType>
    auto get_ai(std::size_t i, const TType& T) const {
        auto alphai = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[i]));
        // See https://doi.org/10.1021/acs.iecr.1c00847
        auto OmegaA = 0.45723552892138218938;
        auto a = OmegaA*POW2(Tc_K[i]*Ru)/pc_Pa[i];
        return forceeval(a*alphai);
    }
    
    template<typename TType, typename FractionsType>
    auto get_ab(const TType& T, const FractionsType& z) const{
        using numtype = std::common_type_t<TType, decltype(z[0])>;
        numtype b = 0.0;
        numtype a = 0.0;
        std::size_t N = alphas.size();
        for (auto i = 0U; i < N; ++i){
            auto bi = get_bi(i, T);
            auto ai = get_ai(i, T);
            for (auto j = 0U; j < N; ++j){
                auto bj = get_bi(j, T);
                auto aj = get_ai(j, T);
                b += z[i]*z[j]*(bi + bj)/2.0*(1.0 - lmat(i,j));
                a += z[i]*z[j]*sqrt(ai*aj)*(1.0 - kmat(i,j));
            }
        }
        return std::make_tuple(a, b);
    }
    template<typename TType, typename FractionsType>
    auto get_c(const TType& /*T*/, const FractionsType& z) const{
        using numtype = std::common_type_t<TType, decltype(z[0])>;
        numtype c = 0.0;
        std::size_t N = alphas.size();
        for (auto i = 0U; i < N; ++i){
            c += z[i]*cs_m3mol[i];
        }
        return c;
    }
    
    template<typename TType, typename RhoType, typename FractionsType>
    auto alphar(const TType& T, const RhoType& rhoinit, const FractionsType& molefrac) const {
        if (static_cast<std::size_t>(molefrac.size()) != alphas.size()) {
            throw std::invalid_argument("Sizes do not match");
        }
        // First shift the volume by the volume translation
        auto c = get_c(T, molefrac);
        auto rho = 1.0/(1.0/rhoinit + c);
        auto Delta1 = 1.0 + sqrt(2.0);
        auto Delta2 = 1.0 - sqrt(2.0);
        auto [a, b] = get_ab(T, molefrac);
        auto Psiminus = -log(1.0 - b * rho);
        auto Psiplus = log((Delta1 * b * rho + 1.0) / (Delta2 * b * rho + 1.0)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - a / (Ru * T) * Psiplus;
        return forceeval(val);
    }
    
    /// Return a tuple of saturated liquid and vapor densities for the EOS given the temperature
    /// Uses the superancillary equations from Bell and Deiters:
    ///
    /// \param T Temperature, in K
    /// \param ifluid Index of fluid you want to get superancillary for. Must be provided in the case of mixtures
    auto superanc_rhoLV(double T, const std::optional<std::size_t> ifluid = std::nullopt) const {
        if (Tc_K.size() != 1) {
            throw std::invalid_argument("function only available for pure species");
        }
        
        std::valarray<double> molefracs(Tc_K.size()); molefracs = 1.0;
        // If more than one component, must provide the ifluid argument
        if(Tc_K.size() > 1){
            if (!ifluid){
                throw teqp::InvalidArgument("For mixtures, the argument ifluid must be provided");
            }
            if (ifluid.value() > Tc_K.size()-1){
                throw teqp::InvalidArgument("ifluid must be less than "+std::to_string(Tc_K.size()));
            }
            molefracs = 0.0;
            molefracs[ifluid.value()] = 1.0;
        }
        
        auto [a, b] = get_ab(T, molefracs);
        auto Ttilde = R(molefracs)*T*b/a;
        auto superanc_index = CubicSuperAncillary::PR_CODE;
        return std::make_tuple(
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOL_CODE, Ttilde)/b,
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOV_CODE, Ttilde)/b
                               );
    }
};


/**
 The RK-PR method as defined in:

 Martín Cismondi, Jørgen Mollerup,
 Development and application of a three-parameter RK–PR equation of state,
 Fluid Phase Equilibria,
 Volume 232, Issues 1–2,
 2005,
 Pages 74-89,
 https://doi.org/10.1016/j.fluid.2005.03.020.
 */
class RKPRCismondi2005{
private:
    const std::vector<double> delta_1, Tc_K, pc_Pa, k;
    const Eigen::ArrayXXd kmat, lmat;
    const double Ru; ///< Universal gas constant, in J/mol/K
    const std::vector<double> a_c, b_c;
    
    /// A convenience function to save some typing
    std::vector<double> get_(const nlohmann::json &j, const std::string& key) const { return j.at(key).get<std::vector<double>>(); }
    
    /// Calculate the parameters \f$y\f$ and \f$d_1\f$
    auto get_yd1(double delta_1_){
        return std::make_tuple(1 + cbrt(2*(1+delta_1_)) + cbrt(4/(1+delta_1_)), (1+delta_1_*delta_1_)/(1+delta_1_));
    }
    
    auto build_ac(){
        std::vector<double> a_c_(delta_1.size());
        for (auto i = 0U; i < delta_1.size(); ++i){
            auto [y, d_1] = get_yd1(delta_1[i]);
            auto Omega_a = (3*y*y + 3*y*d_1 + d_1*d_1 + d_1 - 1.0)/pow(3.0*y + d_1 - 1.0, 2);
            a_c_[i] = Omega_a*pow(Ru*Tc_K[i], 2)/pc_Pa[i];
        }
        return a_c_;
    }
    auto build_bc(){
        std::vector<double> b_c_(delta_1.size());
        for (auto i = 0U; i < delta_1.size(); ++i){
            auto [y, d_1] = get_yd1(delta_1[i]);
            auto Omega_b = 1.0/(3.0*y + d_1 - 1.0);
            b_c_[i] = Omega_b*Ru*Tc_K[i]/pc_Pa[i];
        }
        return b_c_;
    }
public:
    
    RKPRCismondi2005(const nlohmann::json &j) : delta_1(get_(j, "delta_1")), Tc_K(get_(j, "Tcrit / K")), pc_Pa(get_(j, "pcrit / Pa")), k(get_(j, "k")), kmat(build_square_matrix(j.at("kmat"))), lmat(build_square_matrix(j.at("lmat"))), Ru(j.value("R / J/mol/K", constants::R_CODATA2017)), a_c(build_ac()), b_c(build_bc()) {}
    
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const {
        return Ru;
    }
    auto get_delta_1() const { return delta_1; }
    auto get_Tc_K() const { return Tc_K; }
    auto get_pc_Pa() const { return pc_Pa; }
    auto get_k() const { return k; }
    auto get_kmat() const { return kmat; }
    auto get_lmat() const { return lmat; }
    
    template<typename TType>
    auto get_bi(std::size_t i, const TType& /*T*/) const {
        return b_c[i];
    }
    
    template<typename TType>
    auto get_ai(std::size_t i, const TType& T) const {
        return forceeval(a_c[i]*pow(3.0/(2.0+T/Tc_K[i]), k[i]));
    }
    
    template<typename TType, typename FractionsType>
    auto get_ab(const TType& T, const FractionsType& z) const{
        using numtype = std::common_type_t<TType, decltype(z[0])>;
        numtype b = 0.0;
        numtype a = 0.0;
        std::size_t N = delta_1.size();
        for (auto i = 0U; i < N; ++i){
            auto bi = get_bi(i, T);
            auto ai = get_ai(i, T);
            for (auto j = 0U; j < N; ++j){
                auto aj = get_ai(j, T);
                auto bj = get_bi(j, T);
                
                a += z[i]*z[j]*sqrt(ai*aj)*(1.0 - kmat(i,j));
                b += z[i]*z[j]*(bi + bj)/2.0*(1.0 - lmat(i,j));
                
            }
        }
        return std::make_tuple(a, b);
    }
    
    template<typename TType, typename RhoType, typename FractionsType>
    auto alphar(const TType& T, const RhoType& rho, const FractionsType& molefrac) const {
        if (static_cast<std::size_t>(molefrac.size()) != delta_1.size()) {
            throw std::invalid_argument("Sizes do not match");
        }
        
        // The mixture Delta_1 is a mole-fraction-weighted average of the pure values of delta_1
        const auto delta_1_view = Eigen::Map<const Eigen::ArrayXd>(&(delta_1[0]), delta_1.size());
        const decltype(molefrac[0]) Delta1 = (molefrac*delta_1_view).eval().sum();
        
        auto Delta2 = (1.0-Delta1)/(1.0+Delta1);
        auto [a, b] = get_ab(T, molefrac);
        
        auto Psiminus = -log(1.0 - b*rho);
        auto Psiplus = log((Delta1 * b * rho + 1.0) / (Delta2 * b * rho + 1.0)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - a/(this->Ru*T)*Psiplus;
        return forceeval(val);
    }
};
using RKPRCismondi2005_t = decltype(RKPRCismondi2005({}));



}; // namespace teqp

