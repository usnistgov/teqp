#pragma once

#include "teqp/models/fwd.hpp"

#include "teqp/exceptions.hpp"

#include "nlohmann/json.hpp"

namespace teqp {

    static AllowedModels build_model(const nlohmann::json& json) {

        // Extract the name of the model and the model parameters
        std::string kind = json.at("kind");
        auto spec = json.at("model");

        if (kind == "vdW1") {
            return vdWEOS1(spec.at("a"), spec.at("b"));
        }
        else if (kind == "PR") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            return canonical_PR(Tc_K, pc_Pa, acentric);
        }
        else if (kind == "SRK") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            return canonical_SRK(Tc_K, pc_Pa, acentric);
        }
        else if (kind == "CPA") {
            return CPA::CPAfactory(spec);
        }
        else if (kind == "PCSAFT") {
            return PCSAFT::PCSAFTfactory(spec);
        }
        else if (kind == "multifluid") {
            return multifluidfactory(spec);
        }
        else {
            throw teqpcException(30, "Unknown kind:" + kind);
        }
    }

};