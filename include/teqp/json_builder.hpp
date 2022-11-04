#pragma once

#include "teqp/models/fwd.hpp"

#include "teqp/exceptions.hpp"

#include "nlohmann/json.hpp"

namespace teqp {

    inline AllowedModels build_model(const nlohmann::json& json) {
        
        auto build_square_matrix = [](const std::valarray<std::valarray<double>>& m){
            // First assume that the matrix is square, resize
            Eigen::ArrayXXd mat(m.size(), m.size());
            if (m.size() == 0){
                return mat;
            }
            // Then copy elements over
            for (auto i = 0; i < m.size(); ++i){
                auto row = m[i];
                if (row.size() != mat.rows()){
                    throw std::invalid_argument("provided matrix is not square");
                }
                for (auto j = 0; j < row.size(); ++j){
                    mat(i, j) = row[j];
                }
            }
            return mat;
        };

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
                kmat = build_square_matrix(spec["kmat"]);
            }
            return canonical_PR(Tc_K, pc_Pa, acentric, kmat);
        }
        else if (kind == "SRK") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            Eigen::ArrayXXd kmat(0, 0);
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec["kmat"]);
            }
            return canonical_SRK(Tc_K, pc_Pa, acentric, kmat);
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
        else if (kind == "SW_EspindolaHeredia2009"){
            return squarewell::EspindolaHeredia2009(spec.at("lambda"));
        }
        else if (kind == "EXP6_Kataoka1992"){
            return exp6::Kataoka1992(spec.at("alpha"));
        }
        else {
            throw teqpcException(30, "Unknown kind:" + kind);
        }
    }

};
