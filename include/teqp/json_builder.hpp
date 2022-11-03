#pragma once

#include "teqp/models/fwd.hpp"

#include "teqp/exceptions.hpp"

#include "nlohmann/json.hpp"

namespace teqp {

    inline AllowedModels build_model(const nlohmann::json& json) {
        
        auto build_square_matrix = [](const std::valarray<std::valarray<double>>& m){
            Eigen::ArrayXXd mat;
            // First assume that the matrix is square, resize
            mat.resize(m.size(), m.size());
            // Then copy elements over
            for (auto i = 0; i < m.size(); ++i){
                if (m[i].size() != mat.rows()){
                    throw std::invalid_argument("provided matrix is not square");
                }
                for (auto j = 0; i < m[i].size(); ++j){
                    mat(i,j) = m[i][j];
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
        else if (kind == "PR") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            Eigen::ArrayXXd kmat;
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec["kmat"]);
            }
            return canonical_PR(Tc_K, pc_Pa, acentric, kmat);
        }
        else if (kind == "SRK") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            Eigen::ArrayXXd kmat;
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
        else {
            throw teqpcException(30, "Unknown kind:" + kind);
        }
    }

};
