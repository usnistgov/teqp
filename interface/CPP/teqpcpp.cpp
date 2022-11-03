#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/json_builder.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/VLE.hpp"

using namespace teqp;
using namespace teqp::cppinterface;

namespace teqp {
    namespace cppinterface {

        class ModelImplementer : public AbstractModel {
        private:
            template<typename cls>
            const cls& get_or_fail(const std::string& typestr) const{
                if (std::holds_alternative<cls>(m_model)){
                    return std::get<cls>(m_model);
                }
                else{
                    throw std::invalid_argument("This method is only available for models of the type " + std::string(typestr));
                }
            }
        protected:
            const AllowedModels m_model;
            using RAX = Eigen::Ref<const Eigen::ArrayXd>;
            
        public:
            ModelImplementer(AllowedModels&& model) : m_model(model) {};

            double get_Arxy(const int NT, const int ND, const double T, const double rho, const EArrayd& molefracs) const override {
                return std::visit([&](const auto& model) {
                    using tdx = teqp::TDXDerivatives<decltype(model), double, EArrayd>;
                    return tdx::template get_Ar(NT, ND, model, T, rho, molefracs);
                }, m_model);
            }
            nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>& filename_, const std::optional<TCABOptions> &options_) const override {
                return std::visit([&](const auto& model) {
                    using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec0)>>;
                    return crit::trace_critical_arclength_binary(model, T0, rhovec0, "");
                }, m_model);
            }
            EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const override {
                return std::visit([&](const auto& model) {
                    return teqp::pure_VLE_T(model, T, rhoL, rhoV, maxiter);
                }, m_model);
            }
            EArrayd get_fugacity_coefficients(const double T, const EArrayd& rhovec) const override {
                return std::visit([&](const auto& model) {
                    using id = IsochoricDerivatives<decltype(model), double, EArrayd>;
                    return id::get_fugacity_coefficients(model, T, rhovec);
                }, m_model);
            }
            EArrayd get_partial_molar_volumes(const double T, const EArrayd& rhovec) const override {
                return std::visit([&](const auto& model) {
                    using id = IsochoricDerivatives<decltype(model), double, EArrayd>;
                    return id::get_partial_molar_volumes(model, T, rhovec);
                }, m_model);
            }
            EArray33d get_deriv_mat2(const double T, double rho, const EArrayd& z) const override {
                return std::visit([&](const auto& model) {
                    // Although the template argument suggests that only residual terms
                    // are returned, also the ideal-gas ones are returned because the
                    // ideal-gas term is required to implement alphar which just redirects
                    // to alphaig
                    return DerivativeHolderSquare<2, AlphaWrapperOption::residual>(model, T, rho, z).derivs;
                }, m_model);
            }
            double get_B2vir(const double T, const EArrayd& molefrac) const override {
                return std::visit([&](const auto& model) {
                    using vd = VirialDerivatives<decltype(model), double, RAX>;
                    return vd::get_B2vir(model, T, molefrac);
                }, m_model);
            }
            double get_B12vir(const double T, const EArrayd& molefrac) const override {
                return std::visit([&](const auto& model) {
                    using vd = VirialDerivatives<decltype(model), double, RAX>;
                    return vd::get_B12vir(model, T, molefrac);
                }, m_model);
            }
            std::map<int, double> get_Bnvir(const int Nderiv, const double T, const EArrayd& molefrac) const override {
                return std::visit([&](const auto& model) {
                    using vd = VirialDerivatives<decltype(model), double, RAX>;
                    return vd::get_Bnvir_runtime(Nderiv, model, T, molefrac);
                }, m_model);
            }
            double get_dmBnvirdTm(const int Nderiv, const int NTderiv, const double T, const EArrayd& molefrac) const override {
                return std::visit([&](const auto& model) {
                    using vd = VirialDerivatives<decltype(model), double, RAX>;
                    return vd::get_dmBnvirdTm_runtime(Nderiv, NTderiv, model, T, molefrac);
                }, m_model);
            }
            
            // Here XMacros are used to create functions like get_Ar00, get_Ar01, ....
            #define X(i,j) \
            double get_Ar ## i ## j(const double T, const double rho, const EArrayd& molefracs) const override { \
                return std::visit([&](const auto& model) { \
                    using tdx = teqp::TDXDerivatives<decltype(model), double, EArrayd>; \
                    return tdx::template get_Arxy<i,j>(model, T, rho, molefracs); \
                }, m_model); \
            }
            ARXY_args
            #undef X
            
            // Here XMacros are used to create functions like get_Ar01n, get_Ar02n, ....
            #define X(i) \
            EArrayd get_Ar0 ## i ## n(const double T, const double rho, const EArrayd& molefracs) const override { \
                return std::visit([&](const auto& model) { \
                    using tdx = teqp::TDXDerivatives<decltype(model), double, EArrayd>; \
                    auto vals = tdx::template get_Ar0n<i>(model, T, rho, molefracs); \
                    return Eigen::Map<Eigen::ArrayXd>(&(vals[0]), vals.size());\
                }, m_model); \
            }
            AR0N_args
            #undef X
            
            // Methods only available for PC-SAFT
            EArrayd get_m() const override {
                return get_or_fail<PCSAFT_t>("PCSAFT").get_m();
            }
            EArrayd get_sigma_Angstrom() const override {
                return get_or_fail<PCSAFT_t>("PCSAFT").get_sigma_Angstrom();
            }
            EArrayd get_epsilon_over_k_K() const override {
                return get_or_fail<PCSAFT_t>("PCSAFT").get_m();
            }
            double max_rhoN(const double T, const EArrayd& z) const override {
                return get_or_fail<PCSAFT_t>("PCSAFT").max_rhoN(T, z);
            }
        };

        std::unique_ptr<AbstractModel> make_model(const nlohmann::json& j) {
            return std::make_unique<ModelImplementer>(build_model(j));
        }

        std::unique_ptr<AbstractModel> make_multifluid_model(const std::vector<std::string>& components, const std::string& coolprop_root, const std::string& BIPcollectionpath, const nlohmann::json& flags, const std::string& departurepath) {
            return std::make_unique<ModelImplementer>(build_multifluid_model(components, coolprop_root, BIPcollectionpath, flags, departurepath));
        }
        std::unique_ptr<AbstractModel> make_vdW1(double a, double b){
            nlohmann::json j = {{"kind", "vdW1"}, {"model", {{"a", a}, {"b", b}}}};
            return std::make_unique<ModelImplementer>(build_model(j));
        }
    }
}
