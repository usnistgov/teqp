#pragma once 

#include <valarray>

// Pulls in the AllowedModels variant
#include "teqp/models/fwd.hpp"

namespace teqp {
	namespace cppinterface {
		// Wrapper functions (STL types only)
		double get_Arxy(const AllowedModels& model, const int NT, const int ND, const double T, const double rho, const std::valarray<double>& molefracs);
		nlohmann::json trace_critical_arclength_binary(const AllowedModels& model, const double T0, const std::valarray<double>& rhovec0);
	}
}