#pragma once 
#include <memory>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

// The only headers that can be included here are
// ones that define and use POD (plain ole' data) types
#include "teqp/algorithms/critical_tracing_types.hpp"

using EArray2 = Eigen::Array<double, 2, 1>;
using EArrayd = Eigen::ArrayX<double>;

namespace teqp {
    namespace cppinterface {

        class AbstractModel {
        public:
            virtual double get_Arxy(const int, const int, const double, const double, const EArrayd&) const = 0;
            virtual nlohmann::json trace_critical_arclength_binary(const double T0, const EArrayd& rhovec0, const std::optional<std::string>&, const std::optional<TCABOptions> &) const = 0;
            virtual EArray2 pure_VLE_T(const double T, const double rhoL, const double rhoV, int maxiter) const = 0;
            virtual EArrayd get_fugacity_coefficients(const double T, const EArrayd& rhovec) const = 0;
            
            // Methods only available for PCSAFT
            virtual EArrayd get_m() const = 0;
            virtual EArrayd get_sigma_Angstrom() const = 0;
            virtual EArrayd get_epsilon_over_k_K() const = 0;
            virtual double max_rhoN(const double, const EArrayd&) const = 0;
            
            virtual ~AbstractModel() = default;
        };
        
        // Generic JSON-based interface where the model description is encoded as JSON
        std::unique_ptr<AbstractModel> make_model(const nlohmann::json &);

        // Expose factory functions for different models
        // ....
        std::unique_ptr<AbstractModel> make_multifluid_model(
            const std::vector<std::string>& components, 
            const std::string& coolprop_root, 
            const std::string& BIPcollectionpath = {}, 
            const nlohmann::json& flags = {}, 
            const std::string& departurepath = {}
        );
    }
}
