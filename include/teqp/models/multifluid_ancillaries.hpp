#pragma once

#include <valarray>
#include "nlohmann/json.hpp"
#include "teqp/exceptions.hpp"

namespace teqp{

	struct VLEAncillary{
		const double T_r, Tmax, Tmin;
		const std::string type, description;
		const std::valarray<double> n, t;
		const double reducing_value;
		const bool using_tau_r, noexp;
	
		VLEAncillary(const nlohmann::json &j) :
			T_r(j.at("T_r")),
            Tmax(j.at("Tmax")),
            Tmin(j.at("Tmin")),
            type(j.at("type")),
            description(j.at("description")),
            n(j.at("n").get<std::valarray<double>>()),
            t(j.at("t").get<std::valarray<double>>()),
            reducing_value(j.at("reducing_value")),
            using_tau_r(j.at("using_tau_r")),
            noexp(type == "rhoLnoexp"){};

		double operator() (double T) const{
			if (T > T_r) {
				throw teqp::InvalidArgument("Input temperature of " + std::to_string(T) + " K is above the reducing temperature of " + std::to_string(T_r) + " K");
			}
			auto Theta = 1-T/T_r;
			auto RHS = (pow(Theta, t)*n).sum();
			if (using_tau_r){
				RHS *= T_r/T;
			}
			if (noexp){
	            return reducing_value*(1+RHS);
			}
	        else{
	            return exp(RHS)*reducing_value;
	        }
		};
	};

	struct MultiFluidVLEAncillaries {
		const VLEAncillary rhoL, rhoV, pL, pV;
		MultiFluidVLEAncillaries(const nlohmann::json& j) :
			rhoL(VLEAncillary(j.at("rhoL"))),
			rhoV(VLEAncillary(j.at("rhoV"))),
			pL((j.contains("pS")) ? VLEAncillary(j.at("pS")) : VLEAncillary(j.at("pL"))),
			pV((j.contains("pS")) ? VLEAncillary(j.at("pS")) : VLEAncillary(j.at("pV"))){}
	};
}