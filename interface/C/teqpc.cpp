/***
 * Although this is a C-language interface, the code itself is of course written in C++
 * to allow for features only available in C++ and avoiding C memory management
 */

#include <unordered_map>
#include <variant>
#include <atomic>

#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/json_builder.hpp"

// Define empty macros so that no exporting happens
#if defined(TEQPC_CATCH)
#define EXPORT_CODE
#define CONVENTION

#else

#if defined(EXTERN_C_DLLEXPORT)
#define EXPORT_CODE extern "C" __declspec(dllexport) 
#endif
#if defined(EXTERN_C)
#define EXPORT_CODE extern "C" 
#endif
#ifndef CONVENTION
#  define CONVENTION
#endif
#endif


// An atomic is used here for thread safety
// The max possible index is 18,446,744,073,709,551,615
std::atomic<unsigned long long int> next_index{ 0 };

/// A function for returning a sequential index of the next 
std::string get_uid(int N) {
    auto s = std::to_string(next_index);
    next_index++;
    return std::string(N - s.size(), '0') + s;
}

std::unordered_map<std::string, AllowedModels> library;

void exception_handler(int& errcode, char* message_buffer, const int buffer_length)
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

EXPORT_CODE int CONVENTION build_model(const char* j, char* uuid, char* errmsg, int errmsg_length){
    int errcode = 0;
    try{
        nlohmann::json json = nlohmann::json::parse(j);
        std::string uid = get_uid(32);
        try {
            library.emplace(std::make_pair(uid, build_model(json)));
        }
        catch (std::exception &e) {
            throw teqpcException(30, "Unable to load with error:" + std::string(e.what()));
        }
        strcpy(uuid, uid.c_str());
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION free_model(char* uuid, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        library.erase(std::string(uuid));
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_Arxy(const char* uuid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);

        // Lambda function to extract the given derivative from the thing contained in the variant
        auto f = [&](const auto& model) {
            using tdx = TDXDerivatives<decltype(model), double, decltype(molefrac_)>;
            return tdx::get_Ar(NT, ND, model, T, rho, molefrac_);
        };

        // Now call the visitor function to get the value
        *val = std::visit(f, library.at(std::string(uuid)));
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

#if defined(TEQPC_CATCH)

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch/catch.hpp"

TEST_CASE("Use of C interface","[teqpc]") {

    constexpr int errmsg_length = 300;
    char uuid[33] = "", uuidPR[33] = "", errmsg[errmsg_length] = "";
    double val = -1;
    std::valarray<double> molefrac = { 1.0 };

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
    build_model(j.c_str(), uuidPR, errmsg, errmsg_length);
    
    BENCHMARK("vdW1") {
        std::string j = R"({"kind":"vdW1", "model":{"a":1.0, "b":2.0}})";
        int e1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 0, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };
    BENCHMARK("PR") {
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
        int e1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 0, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };
    BENCHMARK("PR call") {
        int e = get_Arxy(uuidPR, 0, 0, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        REQUIRE(e == 0);
        return val;
    };
    BENCHMARK("SRK") {
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
        int e1 = build_model(j.c_str(), uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("CPA") {
        nlohmann::json water = {
            {"a0i / Pa m^6/mol^2",0.12277 }, {"bi / m^3/mol", 0.000014515}, {"c1", 0.67359}, {"Tc / K", 647.096},
            {"epsABi / J/mol", 16655.0}, {"betaABi", 0.0692}, {"class", "4C"}
        };
        nlohmann::json jCPA = {
            {"cubic","SRK"},
            {"pures", {water}},
            {"R_gas / J/mol/K", 8.3144598}
        };
        nlohmann::json j = {
            {"kind", "CPA"},
            {"model", jCPA}
        };
        std::string jstring = j.dump();
        int e1 = build_model(jstring.c_str(), uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("PCSAFT") {
        nlohmann::json jmodel = nlohmann::json::array();
        std::valarray<double> molefrac = { 0.4, 0.6 };
        jmodel.push_back({ {"name", "Methane"}, { "m", 1.0 }, { "sigma_Angstrom", 3.7039},{"epsilon_over_k", 150.03}, {"BibTeXKey", "Gross-IECR-2001"} });
        jmodel.push_back({ {"name", "Ethane"}, { "m", 1.6069 }, { "sigma_Angstrom", 3.5206},{"epsilon_over_k", 191.42}, {"BibTeXKey", "Gross-IECR-2001"} });
        nlohmann::json j = {
            {"kind", "PCSAFT"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(j.dump(2).c_str(), uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300, 3.0e-6, &(molefrac[0]), molefrac.size(), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };
    
}
#else 
int main() {
}
#endif