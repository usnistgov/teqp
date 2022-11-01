#include "teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/json_builder.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/VLE.hpp"

using namespace teqp;
using namespace teqp::cppinterface;

namespace teqp {
    namespace cppinterface {

        class ModelImplementer : public AbstractModel {
        protected:
            const AllowedModels m_model;
            
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
            
            
            // Methods only available for PC-SAFT
            EArrayd get_m() const override {
                return std::get<PCSAFT_t>(m_model).get_m();
            }
            EArrayd get_sigma_Angstrom() const override {
                return std::get<PCSAFT_t>(m_model).get_sigma_Angstrom();
            }
            EArrayd get_epsilon_over_k_K() const override {
                return std::get<PCSAFT_t>(m_model).get_m();
            }
            double max_rhoN(const double T, const EArrayd& z) const override {
                return std::get<PCSAFT_t>(m_model).max_rhoN(T, z);
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
