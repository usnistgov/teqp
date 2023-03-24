#pragma once

#include "teqp/models/fwd.hpp"
#include "teqp/models/mie/lennardjones.hpp"
#include "teqp/models/mie/mie.hpp"

#include "teqp/exceptions.hpp"

#include "nlohmann/json.hpp"

namespace teqp {

    inline AllowedModels build_model(const nlohmann::json& json) {
        
        auto build_square_matrix = [](const nlohmann::json& j){
            try{
                const std::valarray<std::valarray<double>> m = j;
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
            }
            catch(const nlohmann::json::exception&){
                throw teqp::InvalidArgument("Unable to convert this kmat to a 2x2 matrix of doubles:" + j.dump(2));
            }
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
            using namespace PCSAFT;
            std::optional<Eigen::ArrayXXd> kmat;
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec["kmat"]);
            }
            
            if (spec.contains("names")){
                std::vector<std::string> names = spec["names"];
                if (kmat && kmat.value().rows() != names.size()){
                    throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
                }
                return PCSAFTMixture(names, kmat.value_or(Eigen::ArrayXXd{}));
            }
            else if (spec.contains("coeffs")){
                std::vector<SAFTCoeffs> coeffs;
                for (auto j : spec["coeffs"]) {
                    SAFTCoeffs c;
                    c.name = j.at("name");
                    c.m = j.at("m");
                    c.sigma_Angstrom = j.at("sigma_Angstrom");
                    c.epsilon_over_k = j.at("epsilon_over_k");
                    c.BibTeXKey = j.at("BibTeXKey");
                    if (j.contains("(mu^*)^2") && j.contains("nmu")){
                        c.mustar2 = j.at("(mu^*)^2");
                        c.nmu = j.at("nmu");
                    }
                    if (j.contains("(Q^*)^2") && j.contains("nQ")){
                        c.Qstar2 = j.at("(Q^*)^2");
                        c.nQ = j.at("nQ");
                    }
                    coeffs.push_back(c);
                }
                if (kmat && kmat.value().rows() != coeffs.size()){
                    throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
                }
                return PCSAFTMixture(coeffs, kmat.value_or(Eigen::ArrayXXd{}));
            }
            else{
                throw std::invalid_argument("you must provide names or coeffs, but not both");
            }
        }
        else if (kind == "SAFT-VR-Mie") {
            using namespace SAFTVRMie;
            std::optional<Eigen::ArrayXXd> kmat;
            if (spec.contains("kmat")){
                kmat = build_square_matrix(spec["kmat"]);
            }
            
            if (spec.contains("names")){
                std::vector<std::string> names = spec["names"];
                if (kmat && kmat.value().rows() != names.size()){
                    throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
                }
                return SAFTVRMieMixture(names, kmat);
            }
            else if (spec.contains("coeffs")){
                std::vector<SAFTVRMieCoeffs> coeffs;
                for (auto j : spec["coeffs"]) {
                    SAFTVRMieCoeffs c;
                    c.name = j.at("name");
                    c.m = j.at("m");
                    c.sigma_m = (j.contains("sigma_m")) ? j.at("sigma_m").get<double>() : j.at("sigma_Angstrom").get<double>()/1e10;
                    c.epsilon_over_k = j.at("epsilon_over_k");
                    c.lambda_r = j.at("lambda_r");
                    c.lambda_a = j.at("lambda_a");
                    c.BibTeXKey = j.at("BibTeXKey");
                    coeffs.push_back(c);
                }
                if (kmat && kmat.value().rows() != coeffs.size()){
                    throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " +  std::to_string(kmat.value().rows()));
                }
                return SAFTVRMieMixture(coeffs, kmat);
            }
            else{
                throw std::invalid_argument("you must provide names or coeffs, but not both");
            }
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

};
