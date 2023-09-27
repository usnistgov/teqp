#pragma once

#pragma message("Header `teqp/json_builder.hpp` is deprecated!")

#include "teqp/models/fwd.hpp"
#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/models/mie/mie.hpp"

#include "teqp/exceptions.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "nlohmann/json.hpp"
#include <memory>

namespace teqp {

/*
    inline AllowedModels build_model(const nlohmann::json& json) {

        // Extract the name of the model and the model parameters
        std::string kind = json.at("kind");
        auto spec = json.at("model");

        if (kind == "vdW1") {
            return vdWEOS1(spec.at("a"), spec.at("b"));
        }
        else if (kind == "vdW") {
            return vdWEOS<double>(spec.at("Tcrit / K"), spec.at("pcrit / Pa"));
        }
        else if (kind == "PR") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            Eigen::ArrayXXd kmat(0, 0);
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec.at("kmat"));
            }
            return canonical_PR(Tc_K, pc_Pa, acentric, kmat);
        }
        else if (kind == "SRK") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            Eigen::ArrayXXd kmat(0, 0);
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec.at("kmat"));
            }
            return canonical_SRK(Tc_K, pc_Pa, acentric, kmat);
        }
        else if (kind == "CPA") {
            return CPA::CPAfactory(spec);
        }
        else if (kind == "PCSAFT") {
            return PCSAFT::PCSAFTfactory(spec);
        }
        else if (kind == "SAFT-VR-Mie") {
            return SAFTVRMie::SAFTVRMiefactory(spec);
        }
        else if (kind == "multifluid") {
            return multifluidfactory(spec);
        }
        else if (kind == "SW_EspindolaHeredia2009"){
            return squarewell::EspindolaHeredia2009(spec.at("lambda"));
        }
        else if (kind == "EXP6_Kataoka1992"){
            return exp6::Kataoka1992(spec.at("alpha"));
        }
        else if (kind == "AmmoniaWaterTillnerRoth"){
            return AmmoniaWaterTillnerRoth();
        }
        else if (kind == "LJ126_TholJPCRD2016"){
            return build_LJ126_TholJPCRD2016();
        }
        else if (kind == "LJ126_KolafaNezbeda1994"){
            return LJ126KolafaNezbeda1994();
        }
        else if (kind == "LJ126_Johnson1993"){
            return LJ126Johnson1993();
        }
        else if (kind == "Mie_Pohl2023"){
            return Mie::Mie6Pohl2023(spec.at("lambda_a"));
        }
        else if (kind == "2CLJF-Dipole"){
            return twocenterljf::build_two_center_model_dipole(spec.at("author"), spec.at("L^*"), spec.at("(mu^*)^2"));
        }
        else if (kind == "2CLJF-Quadrupole"){
            return twocenterljf::build_two_center_model_quadrupole(spec.at("author"), spec.at("L^*"), spec.at("(mu^*)^2"));
        }
        else if (kind == "IdealHelmholtz"){
            return IdealHelmholtz(spec);
        }
        else {
            throw teqpcException(30, "Unknown kind:" + kind);
        }
    }
*/

};
