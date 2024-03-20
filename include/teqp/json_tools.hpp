#pragma once
#include "nlohmann/json.hpp"
#include <nlohmann/json-schema.hpp>

#include <set>
#include <filesystem>
#include <fstream>
#include "teqp/exceptions.hpp"

#include <Eigen/Dense>

using nlohmann::json;
using nlohmann::json_schema::json_validator;

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

    inline void JSON_to_file(const nlohmann::json& jsondata, const std::string& path){
        std::ofstream file(path);
        file << jsondata;
    }

    inline auto all_same_length(const nlohmann::json& j, const std::vector<std::string>& ks) {
        std::set<decltype(j[0].size())> lengths;
        for (auto k : ks) { lengths.insert(j.at(k).size()); }
        return lengths.size() == 1;
    }

    inline auto build_square_matrix = [](const nlohmann::json& j){
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
            for (auto i = 0U; i < m.size(); ++i){
                auto row = m[i];
                if (row.size() != static_cast<std::size_t>(mat.rows())){
                    throw std::invalid_argument("provided matrix is not square");
                }
                for (auto k = 0U; k < row.size(); ++k){
                    mat(i, k) = row[k];
                }
            }
            return mat;
        }
        catch(const nlohmann::json::exception&){
            throw teqp::InvalidArgument("Unable to convert this kmat to a NxN matrix of doubles:" + j.dump(2));
        }
    };

    /**
    A method for loading something from a nlohmann::json node. Thing to be operated on can be:
    
     0. empty, in which case the file at the default_path is loaded
     1. An object node, in which case it is returned
     2. A path as a string. If this file exists, it will be loaded
     3. A JSON-encoded string
     
     */
    inline auto multilevel_JSON_load(const nlohmann::json &j, const std::optional<std::string>& default_path = std::nullopt){
        
        auto is_valid_path = [](const std::string & s){
            try{
                return std::filesystem::is_regular_file(s) || true; // this will return true if the function CAN BE CALLED without exception, indicating it could be a path
            }
            catch(...){
                return false;
            }
        };
        
        // If not provided (NULL, empty array or empty string), load from the default path provided
        if (j.is_null() || (j.is_array() && j.empty()) || (j.is_string() && j.get<std::string>().empty())){
            if (default_path){
                return load_a_JSON_file(default_path.value());
            }
            else{
                throw teqp::InvalidArgument("default path was not provided, and cannot load this thing: " + j.dump(1));
            }
        }
        else if (j.is_object()){
            // Assume we are already providing the thing
            return j;
        }
        else if (j.is_array() && j.size() > 0){
            return j;
        }
        else if (j.is_string()){
            // If a string, either data in JSON format, or a path-like thing
            std::string s = j.get<std::string>();
            
            // If path to existing file, use it
            if (is_valid_path(s) && std::filesystem::is_regular_file(s)){
                return load_a_JSON_file(s);
            }
            // Or assume it is a string in JSON format
            else{
                return nlohmann::json::parse(s);
            }
        }
        else{
            throw teqp::InvalidArgument("Unable to load the argument to multilevel_JSON_load");
        }
    }

    /**
    This class is not thread-safe for construction because the validator is not
     */
    class JSONValidator{
    public:
        const nlohmann::json schema;
        
        json_validator validator; // create validator
        
        // Instantiate the validator object, will throw if the schema is invalid
        JSONValidator(const nlohmann::json& schema) : schema(schema) {
            validator.set_root_schema(schema); // insert root-schema
        }
        
        // Return the validation errors when trying to validate the JSON
        std::vector<std::string> get_validation_errors(const nlohmann::json& j) const{

            /* Custom error handler */
            class custom_error_handler : public nlohmann::json_schema::basic_error_handler
            {
            public:
                std::vector<std::string> errors;
                void error(const nlohmann::json::json_pointer &ptr, const json &instance, const std::string &message) override
                {
                    nlohmann::json_schema::basic_error_handler::error(ptr, instance, message);
                    std::stringstream ss;
                    ss << ptr << ":" << instance << "': " << message << "\n";
                    errors.push_back(ss.str());
                }
            } handler;
            validator.validate(j, handler); // validate the document
            return handler.errors;
        }
        
        // A quick checker for validation of the JSON
        bool is_valid(const nlohmann::json&j ) const { return get_validation_errors(j).empty(); }
    };

}
