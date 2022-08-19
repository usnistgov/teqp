#pragma once 
#include <memory>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

namespace teqp {
    namespace cppinterface {

        class AbstractModel {
        public:
            virtual double get_Arxy(const int, const int, const double, const double, const Eigen::ArrayXd&) const = 0;
            virtual nlohmann::json trace_critical_arclength_binary(const double T0, const Eigen::ArrayXd& rhovec0) const = 0;
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