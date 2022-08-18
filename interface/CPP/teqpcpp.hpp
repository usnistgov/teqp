#pragma once 

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

namespace teqp {
    namespace cppinterface {

        // Wrapper functions
        double get_Ar01(const void* model, const double T, const double rho, const Eigen::ArrayXd& molefracs); 
        double get_Arxy(const void* model, const int NT, const int ND, const double T, const double rho, const Eigen::ArrayXd& molefracs);
        nlohmann::json trace_critical_arclength_binary(const void* model, const double T0, const Eigen::ArrayXd& rhovec0);
    }
}