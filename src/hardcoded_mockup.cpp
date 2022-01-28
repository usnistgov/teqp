#include "teqp/models/multifluid.hpp"

#include <valarray>
#include <iostream>
#include <sstream>

/// Load a JSON file from a specified file
inline std::string get_file_contents(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("Path to be loaded does not exist: " + path);
    }
    auto stream = std::ifstream(path);
    if (!stream) {
        throw std::invalid_argument("File stream cannot be opened from: " + path);
    }
    std::stringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
};

const std::map<std::string, std::string> hardcoded_models{
    {"internal://n-Propane.json", get_file_contents("../mycp/dev/fluids/n-Propane.json")},
    {"internal://n-Butane.json", get_file_contents("../mycp/dev/fluids/n-Butane.json")},
};
const std::string hardcoded_BIP = get_file_contents("../mycp/dev/mixtures/mixture_binary_pairs.json");
const std::string hardcoded_dep = get_file_contents("../mycp/dev/mixtures/mixture_departure_functions.json");

int main(){

    auto model = teqp::build_multifluid_JSONstr(
        { hardcoded_models.at("internal://n-Propane.json"), hardcoded_models.at("internal://n-Butane.json") }, 
        hardcoded_BIP, hardcoded_dep
    );

    return EXIT_SUCCESS;
}