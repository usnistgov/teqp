#pragma once

#include "teqp/models/vdW.hpp"
#include "teqp/models/cubics.hpp"
#include "teqp/models/CPA.hpp"

#include "nlohmann/json.hpp"

using vad = std::valarray<double>;
using cub = decltype(canonical_PR(vad{}, vad{}, vad{}));
using cpatype = decltype(CPA::CPAfactory(nlohmann::json{})); // The type returned by the factory function

using AllowedModels = std::variant<vdWEOS1, cub, cpatype>;

AllowedModels build_model(const nlohmann::json& json) {

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
    else {
        throw teqpcException(30, "Unknown kind:" + kind);
    }
}
