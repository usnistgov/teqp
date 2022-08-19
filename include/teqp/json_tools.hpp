#include "nlohmann/json.hpp"

#include <set>
#include <filesystem>
#include <fstream>

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
}