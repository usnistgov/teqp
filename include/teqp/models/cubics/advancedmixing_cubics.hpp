#pragma once

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

#include "teqp/models/cubics/simple_cubics.hpp"
#include "teqp/models/activity/activity_models.hpp"
using namespace teqp::activity::activity_models;

namespace teqp {

enum class AdvancedPRaEMixingRules {knotspecified, kLinear, kQuadratic};

NLOHMANN_JSON_SERIALIZE_ENUM( AdvancedPRaEMixingRules, {
    {AdvancedPRaEMixingRules::knotspecified, nullptr},
    {AdvancedPRaEMixingRules::kLinear, "Linear"},
    {AdvancedPRaEMixingRules::kQuadratic, "Quadratic"},
})

struct AdvancedPRaEOptions{
    AdvancedPRaEMixingRules brule = AdvancedPRaEMixingRules::kQuadratic;
    double s = 2.0;
    double CEoS = -sqrt(2.0)/2.0*log(1.0 + sqrt(2.0));
    double R_JmolK = constants::R_CODATA2017;
};

inline void from_json(const json& j, AdvancedPRaEOptions& o) {
    j.at("brule").get_to(o.brule);
    j.at("s").get_to(o.s);
    j.at("CEoS").get_to(o.CEoS);
    if (j.contains("R / J/mol/K")){
        o.R_JmolK = j.at("R / J/mol/K");
    }
}

/**
 Cubic EOS with advanced mixing rules, the EoS/aE method of Jaubert and co-workers
 
 */
template <typename NumType, typename AlphaFunctions = std::vector<AlphaFunctionOptions>>
class AdvancedPRaEres {
public:
    // Hard-coded values for Peng-Robinson
    const NumType Delta1 = 1+sqrt(2.0);
    const NumType Delta2 = 1-sqrt(2.0);
    // See https://doi.org/10.1021/acs.iecr.1c00847
    const NumType OmegaA = 0.45723552892138218938;
    const NumType OmegaB = 0.077796073903888455972;
    const int superanc_code = CubicSuperAncillary::PR_CODE;
    
    
protected:
    
    std::valarray<NumType> Tc_K, pc_Pa;
    
    std::valarray<NumType> ai, bi;
    
    const AlphaFunctions alphas;
    const ResidualHelmholtzOverRTVariant ares;
    Eigen::ArrayXXd lmat;
    
    const AdvancedPRaEMixingRules brule;
    const double s;
    const double CEoS;
    const double R_JmolK;
    
    nlohmann::json meta;
    
    template<typename TType, typename IndexType>
    auto get_ai(TType& T, IndexType i) const {
        auto alphai = std::visit([&](auto& t) { return t(T); }, alphas[i]);
        return forceeval(ai[i]*alphai);
    }
    
    template<typename TType, typename IndexType>
    auto get_bi(TType& /*T*/, IndexType i) const { return bi[i]; }
    
    template<typename IndexType>
    void check_lmat(IndexType N) {
        if (lmat.cols() != lmat.rows()) {
            throw teqp::InvalidArgument("lmat rows [" + std::to_string(lmat.rows()) + "] and columns [" + std::to_string(lmat.cols()) + "] are not identical");
        }
        if (lmat.cols() == 0) {
            lmat.resize(N, N); lmat.setZero();
        }
        else if (lmat.cols() != static_cast<Eigen::Index>(N)) {
            throw teqp::InvalidArgument("lmat needs to be a square matrix the same size as the number of components [" + std::to_string(N) + "]");
        }
    }
    
public:
    AdvancedPRaEres(const std::valarray<NumType>& Tc_K, const std::valarray<NumType>& pc_Pa, const AlphaFunctions& alphas, const ResidualHelmholtzOverRTVariant& ares, const Eigen::ArrayXXd& lmat, const AdvancedPRaEOptions& options = {})
    : Tc_K(Tc_K), pc_Pa(pc_Pa), alphas(alphas), ares(ares), lmat(lmat), brule(options.brule), s(options.s), CEoS(options.CEoS), R_JmolK(options.R_JmolK)
    {
        ai.resize(Tc_K.size());
        bi.resize(Tc_K.size());
        for (auto i = 0U; i < Tc_K.size(); ++i) {
            ai[i] = OmegaA * pow2(R_JmolK * Tc_K[i]) / pc_Pa[i];
            bi[i] = OmegaB * R_JmolK * Tc_K[i] / pc_Pa[i];
        }
        check_lmat(ai.size());
    };
    
    void set_meta(const nlohmann::json& j) { meta = j; }
    auto get_meta() const { return meta; }
    auto get_lmat() const { return lmat; }
    auto get_Tc_K() const { return Tc_K; }
    auto get_pc_Pa() const { return pc_Pa; }
    
    static double get_bi(double Tc_K, double pc_Pa){
        const NumType OmegaB = 0.077796073903888455972;
        const NumType R = 8.31446261815324;
        return OmegaB*R*Tc_K/pc_Pa;
    }
    
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
        auto a = get_am_over_bm(T, molefracs)*b;
        auto Ttilde = R(molefracs)*T*b/a;
        return std::make_tuple(
           CubicSuperAncillary::supercubic(superanc_code, CubicSuperAncillary::RHOL_CODE, Ttilde)/b,
           CubicSuperAncillary::supercubic(superanc_code, CubicSuperAncillary::RHOV_CODE, Ttilde)/b
        );
    }
    
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const {
        return R_JmolK;
    }
    
    template<typename TType, typename CompType>
    auto get_a(TType T, const CompType& molefracs) const {
        return forceeval(get_am_over_bm(T, molefracs)*get_b(T, molefracs));
    }
    
    template<typename TType, typename CompType>
    auto get_am_over_bm(TType T, const CompType& molefracs) const {
        auto aEresRT = std::visit([&](auto& aresRTfunc) { return aresRTfunc(T, molefracs); }, ares); // aEres/RT, so a non-dimensional quantity
        std::common_type_t<TType, decltype(molefracs[0])> summer = aEresRT*R_JmolK*T/CEoS;
        for (auto i = 0U; i < molefracs.size(); ++i) {
            summer += molefracs[i]*get_ai(T,i)/get_bi(T,i);
        }
        return forceeval(summer);
    }
    
    template<typename TType, typename CompType>
    auto get_b(TType T, const CompType& molefracs) const {
        std::common_type_t<TType, decltype(molefracs[0])> b_ = 0.0;
        
        switch (brule){
            case AdvancedPRaEMixingRules::kQuadratic:
                for (auto i = 0U; i < molefracs.size(); ++i) {
                    auto bi_ = get_bi(T, i);
                    for (auto j = 0U; j < molefracs.size(); ++j) {
                        auto bj_ = get_bi(T, j);
                        
                        auto bij = (1 - lmat(i,j)) * pow((pow(bi_, 1.0/s) + pow(bj_, 1.0/s))/2.0, s);
                        b_ += molefracs[i] * molefracs[j] * bij;
                    }
                }
                break;
            case AdvancedPRaEMixingRules::kLinear:
                for (auto i = 0U; i < molefracs.size(); ++i) {
                    b_ += molefracs[i] * get_bi(T, i);
                }
                break;
            default:
                throw teqp::InvalidArgument("Mixing rule for b is invalid");
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
        auto a = get_am_over_bm(T, molefrac)*b;
        auto Psiminus = -log(1.0 - b * rho);
        auto Psiplus = log((Delta1 * b * rho + 1.0) / (Delta2 * b * rho + 1.0)) / (b * (Delta1 - Delta2));
        auto val = Psiminus - a / (R_JmolK * T) * Psiplus;
        return forceeval(val);
    }
};

inline auto make_AdvancedPRaEres(const nlohmann::json& j){
    
    std::valarray<double> Tc_K = j.at("Tcrit / K");
    std::valarray<double> pc_Pa = j.at("pcrit / Pa");
    
    std::vector<AlphaFunctionOptions> alphas = build_alpha_functions(Tc_K, j.at("alphas"));
    
    auto get_ares_model = [&](const nlohmann::json& armodel) -> ResidualHelmholtzOverRTVariant {
        
        std::string type = armodel.at("type");
        if (type == "Wilson"){
            std::vector<double> b;
            for (auto i = 0U; i < Tc_K.size(); ++i){
                b.push_back(teqp::AdvancedPRaEres<double>::get_bi(Tc_K[i], pc_Pa[i]));
            }
            auto mWilson = build_square_matrix(armodel.at("m"));
            auto nWilson = build_square_matrix(armodel.at("n"));
            return WilsonResidualHelmholtzOverRT<double>(b, mWilson, nWilson);
        }
        else{
            throw teqp::InvalidArgument("bad type of ares model: " + type);
        }
    };
    auto aresmodel = get_ares_model(j.at("aresmodel"));
    
    AdvancedPRaEOptions options = j.at("options");
    auto model = teqp::AdvancedPRaEres<double>(Tc_K, pc_Pa, alphas, aresmodel, Eigen::ArrayXXd::Zero(2, 2), options);
    return model;
}

using advancedPRaEres_t = decltype(make_AdvancedPRaEres({}));


}; // namespace teqp
