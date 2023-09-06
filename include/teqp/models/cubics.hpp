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

using AlphaFunctionOptions = std::variant<BasicAlphaFunction<double>, TwuAlphaFunction<double>>;

template <typename NumType, typename AlphaFunctions>
class GenericCubic {
protected:
    std::valarray<NumType> ai, bi;
    const NumType Delta1, Delta2, OmegaA, OmegaB;
    int superanc_index;
    const AlphaFunctions alphas;
    Eigen::ArrayXXd kmat;
    
    nlohmann::json meta;
    
    template<typename TType, typename IndexType>
    auto get_ai(TType T, IndexType i) const { return ai[i]; }
    
    template<typename TType, typename IndexType>
    auto get_bi(TType T, IndexType i) const { return bi[i]; }
    
    template<typename IndexType>
    void check_kmat(IndexType N) {
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows [" + std::to_string(kmat.rows()) + "] and columns [" + std::to_string(kmat.cols()) + "] are not identical");
        }
        if (kmat.cols() == 0) {
            kmat.resize(N, N); kmat.setZero();
        }
        else if (kmat.cols() != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components [" + std::to_string(N) + "]");
        }
    };
    
public:
    GenericCubic(NumType Delta1, NumType Delta2, NumType OmegaA, NumType OmegaB, int superanc_index, const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa, const AlphaFunctions& alphas, const Eigen::ArrayXXd& kmat)
    : Delta1(Delta1), Delta2(Delta2), OmegaA(OmegaA), OmegaB(OmegaB), superanc_index(superanc_index), alphas(alphas), kmat(kmat)
    {
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0; i < Tc_K.size(); ++i) {
            ai[i] = OmegaA * pow2(Ru * Tc_K[i]) / pc_Pa[i];
            bi[i] = OmegaB * Ru * Tc_K[i] / pc_Pa[i];
        }
        check_kmat(ai.size());
    };
    
    void set_meta(const nlohmann::json& j) { meta = j; }
    auto get_meta() const { return meta; }
    auto get_kmat() const { return kmat; }
    
    /// Return a tuple of saturated liquid and vapor densities for the EOS given the temperature
    /// Uses the superancillary equations from Bell and Deiters:
    auto superanc_rhoLV(double T) const {
        if (ai.size() != 1) {
            throw std::invalid_argument("function only available for pure species");
        }
        const std::valarray<double> z = { 1.0 };
        auto b = get_b(T, z);
        auto Ttilde = R(z)*T*b/get_a(T,z);
        return std::make_tuple(
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOL_CODE, Ttilde)/b,
                               CubicSuperAncillary::supercubic(superanc_index, CubicSuperAncillary::RHOV_CODE, Ttilde)/b
                               );
    }
    
    const NumType Ru = get_R_gas<double>(); /// Universal gas constant, exact number
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return Ru;
    }
    
    template<typename TType, typename CompType>
    auto get_a(TType T, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> a_ = 0.0;
        auto ai = this->ai;
        for (auto i = 0; i < molefracs.size(); ++i) {
            auto alphai = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[i]));
            auto ai_ = forceeval(ai[i] * alphai);
            for (auto j = 0; j < molefracs.size(); ++j) {
                auto alphaj = forceeval(std::visit([&](auto& t) { return t(T); }, alphas[j]));
                auto aj_ = ai[j] * alphaj;
                auto aij = forceeval((1 - kmat(i,j)) * sqrt(ai_ * aj_));
                a_ = a_ + molefracs[i] * molefracs[j] * aij;
            }
        }
        return forceeval(a_);
    }
    
    template<typename TType, typename CompType>
    auto get_b(TType /*T*/, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> b_ = 0.0;
        for (auto i = 0; i < molefracs.size(); ++i) {
            b_ = b_ + molefracs[i] * bi[i];
        }
        return forceeval(b_);
    }
    
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType& T,
                const RhoType& rho,
                const MoleFracType& molefrac) const
    {
        if (molefrac.size() != alphas.size()) {
            throw std::invalid_argument("Sizes do not match");
        }
        auto b = get_b(T, molefrac);
        auto Psiminus = -log(1.0 - b * rho);
        auto Psiplus = log((Delta1 * b * rho + 1.0) / (Delta2 * b * rho + 1.0)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - get_a(T, molefrac) / (Ru * T) * Psiplus;
        return forceeval(val);
    }
};

template <typename TCType, typename PCType, typename AcentricType>
auto canonical_SRK(TCType Tc_K, PCType pc_Pa, AcentricType acentric, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt) {
    double Delta1 = 1;
    double Delta2 = 0;
    AcentricType m = 0.48 + 1.574 * acentric - 0.176 * acentric * acentric;
    
    std::vector<AlphaFunctionOptions> alphas;
    for (auto i = 0; i < Tc_K.size(); ++i) {
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
        {"kind", "Soave-Redlich-Kwong"}
    };
    const std::size_t N = m.size();
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, CubicSuperAncillary::SRK_CODE, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)));
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
auto canonical_PR(TCType Tc_K, PCType pc_Pa, AcentricType acentric, const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt) {
    double Delta1 = 1+sqrt(2.0);
    double Delta2 = 1-sqrt(2.0);
    AcentricType m = acentric*0.0;
    std::vector<AlphaFunctionOptions> alphas; 
    for (auto i = 0; i < Tc_K.size(); ++i) {
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
        {"kind", "Peng-Robinson"}
    };
    
    const std::size_t N = m.size();
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, CubicSuperAncillary::PR_CODE, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)));
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
    
    int superanc_code = CubicSuperAncillary::UNKNOWN_CODE;
    
    // Build the list of alpha functions, one per component
    std::vector<AlphaFunctionOptions> alphas;
    
    double Delta1, Delta2, OmegaA, OmegaB;
    std::string kind = "custom";
    
    auto add_alphas = [&](const nlohmann::json& jalphas){
        std::size_t i = 0;
        if (jalphas.size() != Tc_K.size()){
            throw teqp::InvalidArgument("alpha must be the same length as components");
        }
        for (auto alpha : jalphas){
            std::string type = alpha.at("type");
            std::valarray<double> c_ = alpha.at("c");
            Eigen::Array3d c = Eigen::Map<Eigen::Array3d>(&(c_[0]), c_.size());
            if (type == "Twu"){
                alphas.emplace_back(TwuAlphaFunction(Tc_K[i], c));
            }
            else{
                throw teqp::InvalidArgument("alpha type is not understood: "+type);
            }
            i++;
        }
    };
    
    if (spec.at("type") == "PR" ){
        Delta1 = 1+sqrt(2.0);
        Delta2 = 1-sqrt(2.0);
        // See https://doi.org/10.1021/acs.iecr.1c00847
        OmegaA = 0.45723552892138218938;
        OmegaB = 0.077796073903888455972;
        superanc_code = CubicSuperAncillary::PR_CODE;
        kind = "Peng-Robinson";
        
        if (!spec.contains("alpha")){
            for (auto i = 0; i < Tc_K.size(); ++i) {
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
            add_alphas(spec.at("alpha"));
        }
    }
    else if (spec.at("type") == "SRK"){
        Delta1 = 1;
        Delta2 = 0;
        if (!spec.contains("alpha")){
            for (auto i = 0; i < Tc_K.size(); ++i) {
                double mi = 0.48 + 1.574 * acentric[i] - 0.176 * acentric[i] * acentric[i];
                alphas.emplace_back(BasicAlphaFunction(Tc_K[i], mi));
            }
        }
        else{
            if (!spec["alpha"].is_array()){
                throw teqp::InvalidArgument("alpha must be array of objects");
            }
            add_alphas(spec.at("alpha"));
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
        {"kind", kind}
    };
    if (spec.contains("alpha")){
        meta["alpha"] = spec.at("alpha");
    }
    
    auto cub = GenericCubic(Delta1, Delta2, OmegaA, OmegaB, superanc_code, Tc_K, pc_Pa, std::move(alphas), kmat.value_or(Eigen::ArrayXXd::Zero(N,N)));
    cub.set_meta(meta);
    return cub;
}



}; // namespace teqp
