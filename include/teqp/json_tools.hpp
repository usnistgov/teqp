#pragma once
#include "nlohmann/json.hpp"

#include <set>
#include <filesystem>
#include <fstream>
#include "teqp/exceptions.hpp"

namespace teqp{
    
    /// Load a JSON file from a specified file
    inline nlohmann::json load_a_JSON_file(const std::string& path) {
        if (!std::filesystem::is_regular_file(path)) {
            throw std::invalid_argument("Path to be loaded does not exist: " + path);
        }
        auto stream = std::ifstream(path);
        if (!stream) {
            throw std::invalid_argument("File stream cannot be opened from: " + path);
        }
        try {
            return nlohmann::json::parse(stream);
        }
        catch (...) {
            throw std::invalid_argument("File at " + path + " is not valid JSON");
        }
    }

    inline auto all_same_length(const nlohmann::json& j, const std::vector<std::string>& ks) {
        std::set<decltype(j[0].size())> lengths;
        for (auto k : ks) { lengths.insert(j.at(k).size()); }
        return lengths.size() == 1;
    }

    auto build_square_matrix = [](const nlohmann::json& j){
        if (j.is_null() || (j.is_array() && j.size() == 0)){
            return Eigen::ArrayXXd(0, 0);
        }
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
}
