#pragma once

#include <valarray>
#include "nlohmann/json.hpp"

namespace teqp{

	struct VLEAncillary{
		const double T_r, Tmax, Tmin, reducing_value;
		const std::string type, description;
		const std::valarray<double> n, t;
		const bool using_tau_r, noexp;
	
		VLEAncillary(const nlohmann::json &j) :
			T_r(j.at("T_r")),
            Tmax(j.at("Tmax")),
            Tmin(j.at("Tmin")),
            description(j.at("description")),
            n(j.at("n").get<std::valarray<double>>()),
            t(j.at("t").get<std::valarray<double>>()),
            reducing_value(j.at("reducing_value")),
            type(j.at("type")),
            using_tau_r(j.at("using_tau_r")),
            noexp(type == "rhoLnoexp"){};

		double operator() (double T) const{
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