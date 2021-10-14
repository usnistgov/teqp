/***
 * Although this is a C-language interface, the code itself is of course written in C++
 * to allow for features only available in C++ and avoiding C memory management
 */

#include <unordered_map>
#include <variant>

#include "teqp/models/eos.hpp"
#include "teqp/models/cubics.hpp"
#include "teqp/derivs.hpp"

// TODO: actually make new UUID for each model instance
// TODO: catch tests

using vad = std::valarray<double>;
using cub = decltype(canonical_PR(vad{}, vad{}, vad{}));

using AllowedModels = std::variant<vdWEOS1, cub>;
std::unordered_map<std::string, AllowedModels> library;

class teqpcException : public std::exception {
public:
    const int code;
    const std::string msg;
    teqpcException(int code, const std::string& msg) : code(code), msg(msg) {}
};

void HandleException(int& errcode, char* message_buffer, const int buffer_length)
{
    try{
        throw; // Rethrow the error so that we can handle it here
    }
    catch (teqpcException& e) {
        errcode = e.code;
        strcpy(message_buffer, e.msg.c_str());
    }
    catch (std::exception e) {
        errcode = 9999;
        strcpy(message_buffer, e.what());
    }
}

int build_model(const char* j, char* uuid, char* errmsg, int errmsg_length){
    int errcode = 0;
    try{
        std::string uid = "100"; // TODO: this needs to be unique, not hardcoded
        nlohmann::json json = nlohmann::json::parse(j);
        
        // Extract the name of the model and the parameters
        std::string kind = json.at("kind");
        auto spec = json.at("model");

        if (kind == "vdW1") {
            library.emplace(std::make_pair(uid, vdWEOS1(spec.at("a"), spec.at("b"))));
        }
        else if (kind == "PR") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            library.emplace(std::make_pair(uid, canonical_PR(Tc_K, pc_Pa, acentric)));
        }
        else if (kind == "SRK") {
            std::valarray<double> Tc_K = spec.at("Tcrit / K"), pc_Pa = spec.at("pcrit / Pa"), acentric = spec.at("acentric");
            library.emplace(std::make_pair(uid, canonical_SRK(Tc_K, pc_Pa, acentric)));
        }
        else {
            throw teqpcException(30, "Unknown kind:" + kind);
        }
        strcpy(uuid, uid.c_str());
    }
    catch (...) {
        HandleException(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

int free_model(char* uuid, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        library.erase(std::string(uuid));
    }
    catch (...) {
        HandleException(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

int get_Arxy(char* uuid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);

        // Lambda function to extract the given derivative from the thing contained in the variant
        auto f = [&](auto& model) {
            using tdx = TDXDerivatives<decltype(model), double, decltype(molefrac_)>;  
            return tdx::get_Ar(NT, ND, model, T, rho, molefrac_);
        };

        // Now call the visitor function to get the value
        *val = std::visit(f, library.at(std::string(uuid)));
    }
    catch (...) {
        HandleException(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

int main() {
    constexpr int errmsg_length = 300;
    char uuid[300] = "", errmsg[errmsg_length] = "";
    double val = -1;
    std::valarray<double> molefrac = { 1.0 };
    {
        std::string j = R"({"kind":"vdW1", "model":{"a":1.0, "b":2.0}})";
        int errcode1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int errcode11 = get_Arxy(uuid, 0, 0, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int errcode2 = free_model(uuid, errmsg, errmsg_length);
        int rrr = 0;
    }
    {
        std::string j = R"(
            {
                "kind": "PR", 
                "model": {
                    "Tcrit / K": [190], 
                    "pcrit / Pa": [3.5e6], 
                    "acentric": [0.11]
                }
            }
        )";
        int errcode1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int errcode11 = get_Arxy(uuid, 0, 0, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int errcode2 = free_model(uuid, errmsg, errmsg_length);
        int rrr = 0;
    }
    {
        std::string j = R"(
            {
                "kind": "SRK", 
                "model": {
                    "Tcrit / K": [190], 
                    "pcrit / Pa": [3.5e6], 
                    "acentric": [0.11]
                }
            }
        )";
        int errcode1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int errcode11 = get_Arxy(uuid, 0, 1, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int errcode2 = free_model(uuid, errmsg, errmsg_length);
        int rrr = 0;
    }
}