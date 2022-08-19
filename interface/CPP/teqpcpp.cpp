#include "teqpcpp.hpp"
#include "teqp/derivs.hpp"
#include "teqp/json_builder.hpp"
#include "teqp/algorithms/critical_tracing.hpp"

using namespace teqp;
using namespace teqp::cppinterface;

namespace teqp {
    namespace cppinterface {

        class ModelImplementer : public AbstractModel {
        protected:
            const AllowedModels m_model;
        public:
            ModelImplementer(AllowedModels&& model) : m_model(model) {};

            double get_Arxy(const int NT, const int ND, const double T, const double rho, const Eigen::ArrayXd& molefracs) const override {
                return std::visit([&](const auto& model) {
                    using tdx = teqp::TDXDerivatives<decltype(model), double, Eigen::ArrayXd>;
                    return tdx::template get_Ar(NT, ND, model, T, rho, molefracs);
                }, m_model);
            }
            nlohmann::json trace_critical_arclength_binary(const double T0, const Eigen::ArrayXd& rhovec0) const override {
                return std::visit([&](const auto& model) {
                    using crit = teqp::CriticalTracing<decltype(model), double, std::decay_t<decltype(rhovec0)>>;
                    return crit::trace_critical_arclength_binary(model, T0, rhovec0, "");
                }, m_model);
            }
        };

        std::unique_ptr<AbstractModel> make_model(const nlohmann::json& j) {
            return std::make_unique<ModelImplementer>(build_model(j));
        }

        std::unique_ptr<AbstractModel> make_multifluid_model(const std::vector<std::string>& components, const std::string& coolprop_root, const std::string& BIPcollectionpath, const nlohmann::json& flags, const std::string& departurepath) {
            return std::make_unique<ModelImplementer>(build_multifluid_model(components, coolprop_root, BIPcollectionpath, flags, departurepath));
        }
    }
}