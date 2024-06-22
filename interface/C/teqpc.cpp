/***
 * Although this is a C-language interface, the code itself is of course written in C++
 * to allow for features only available in C++ and avoiding C memory management
 */

#include <unordered_map>
#include <variant>
#include <atomic>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/exceptions.hpp"

// Define empty macros so that no exporting happens
#if defined(TEQPC_CATCH)
#define EXPORT_CODE
#define CONVENTION

# else

#if !defined(EXPORT_CODE)
#  if defined(EXTERN_C_DLLEXPORT)
#    define EXPORT_CODE extern "C" __declspec(dllexport)
#  endif
#  if defined(EXTERN_C)
#    define EXPORT_CODE extern "C"
#  endif
#endif

#if !defined(CONVENTION)
#  define CONVENTION
#endif

#endif

#include "teqpc.h"

using namespace teqp;

// An atomic is used here for thread safety
// The max possible index is 18,446,744,073,709,551,615
std::atomic<long long int> next_index{ 0 };

std::unordered_map<unsigned long long int, std::shared_ptr<teqp::cppinterface::AbstractModel>> library;

void exception_handler(int& errcode, char* message_buffer, const int buffer_length)
{
    auto write_error = [&](const std::string& msg){
        if (msg.size() < static_cast<std::size_t>(buffer_length)){
            strcpy(message_buffer, msg.c_str());
        }
        else{
            std::string toolong_message = "Error message too long for buffer";
            if (toolong_message.size() < static_cast<std::size_t>(buffer_length)){
                strcpy(message_buffer, toolong_message.c_str());
            }
            else if (buffer_length > 1){
                strcpy(message_buffer, "?");
            }
        }
    };
    
    try{
        throw; // Rethrow the error so that we can handle it here
    }
    catch (teqpcException& e) {
        errcode = e.code;
        write_error(e.msg);
    }
    catch (std::exception e) {
        errcode = 9999;
        write_error(e.what());
    }
}

EXPORT_CODE int CONVENTION build_model(const char* j, long long int* uuid, char* errmsg, int errmsg_length){
    int errcode = 0;
    try{
        nlohmann::json json = nlohmann::json::parse(j);
        long long int uid = next_index++;
        try {
            library.emplace(std::make_pair(uid, cppinterface::make_model(json)));
        }
        catch (std::exception &e) {
            throw teqpcException(30, "Unable to load with error:" + std::string(e.what()));
        }
        *uuid = uid;
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION free_model(const long long int uuid, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        library.erase(uuid);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_Arxy(const long long int uuid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_Arxy(NT, ND, T, rho, molefrac_);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_ATrhoXi(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_ATrhoXi(T, NT, rhomolar, ND, molefrac_, i, NXi);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_ATrhoXiXj(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_ATrhoXiXj(T, NT, rhomolar, ND, molefrac_, i, NXi, j, NXj);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_ATrhoXiXjXk(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_ATrhoXiXjXk(T, NT, rhomolar, ND, molefrac_, i, NXi, j, NXj, k, NXk);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}


EXPORT_CODE int CONVENTION get_AtaudeltaXi(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_AtaudeltaXi(tau, Ntau, delta, Ndelta, molefrac_, i, NXi);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_AtaudeltaXiXj(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_AtaudeltaXiXj(tau, Ntau, delta, Ndelta, molefrac_, i, NXi, j, NXj);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_AtaudeltaXiXjXk(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk, double *val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_AtaudeltaXiXjXk(tau, Ntau, delta, Ndelta, molefrac_, i, NXi, j, NXj, k, NXk);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

EXPORT_CODE int CONVENTION get_dmBnvirdTm(const long long int uuid, const int Nvir, const int NT, const double T, const double* molefrac, const int Ncomp, double* val, char* errmsg, int errmsg_length) {
    int errcode = 0;
    try {
        // Make an Eigen view of the double buffer
        Eigen::Map<const Eigen::ArrayXd> molefrac_(molefrac, Ncomp);
        // Call the function
        *val = library.at(uuid)->get_dmBnvirdTm(Nvir, NT, T, molefrac_);
    }
    catch (...) {
        exception_handler(errcode, errmsg, errmsg_length);
    }
    return errcode;
}

#if defined(TEQPC_CATCH)

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include "teqp/json_tools.hpp"

const std::string FLUIDDATAPATH = "../teqp/fluiddata"; // normally defined in src/test/test_common.in

TEST_CASE("Use of C interface","[teqpc]") {

    constexpr int errmsg_length = 3000;
    long long int uuid, uuidPR, uuidMF;
    char errmsg[errmsg_length] = "";
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
    build_model(j.c_str(), &uuidPR, errmsg, errmsg_length);
    {
        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["departure"] = "";
        jmodel["BIP"] = "";
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back(FLUIDDATAPATH+"/dev/fluids/Argon.json");

        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(js.c_str(), &uuidMF, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        CAPTURE(errmsg);
    }
    {
        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["departure"] = "";
        jmodel["BIP"] = "";
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back(load_a_JSON_file(FLUIDDATAPATH+"/dev/fluids/Argon.json"));
        
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(js.c_str(), &uuid, errmsg, errmsg_length);
        CAPTURE(errmsg);
        REQUIRE(e1 == 0);
    }
    {
        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["departure"] = "";
        jmodel["BIP"] = "";
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back("Ethane");
        jmodel["components"].push_back("Nitrogen");
        jmodel["root"] = FLUIDDATAPATH;
        
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(js.c_str(), &uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
    }
    
    BENCHMARK("vdW1 parse string") {
        std::string j = R"({"kind":"vdW1", "model":{"a":1.0, "b":2.0}})";
        return nlohmann::json::parse(j);
    };
    BENCHMARK("vdW1 construct only") {
        std::string j = R"({"kind":"vdW1", "model":{"a":1.0, "b":2.0}})";
        int e1 = build_model(j.c_str(), &uuid, errmsg, errmsg_length);
        return e1;
    };
    BENCHMARK("vdW1") {
        std::string j = R"({"kind":"vdW1", "model":{"a":1.0, "b":2.0}})";
        int e1 = build_model(j.c_str(), &uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 0, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
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
        int e1 = build_model(j.c_str(), &uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 0, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };
    BENCHMARK("PR call") {
        int e = get_Arxy(uuidPR, 0, 0, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
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
        int e1 = build_model(j.c_str(), &uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
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
            {"radial_dist", "KG"}, 
            {"pures", {water}},
            {"R_gas / J/mol/K", 8.3144598}
        };
        nlohmann::json j = {
            {"kind", "CPA"},
            {"model", jCPA}
        };
        std::string jstring = j.dump();
        int e1 = build_model(jstring.c_str(), &uuid, errmsg, errmsg_length);
        CAPTURE(errmsg);
        int e2 = get_Arxy(uuid, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        CAPTURE(jstring);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("PCSAFT") {
        nlohmann::json jcoeffs = nlohmann::json::array();
        std::valarray<double> molefrac = { 0.4, 0.6 };
        jcoeffs.push_back({ {"name", "Methane"}, { "m", 1.0 }, { "sigma_Angstrom", 3.7039},{"epsilon_over_k", 150.03}, {"BibTeXKey", "Gross-IECR-2001"} });
        jcoeffs.push_back({ {"name", "Ethane"}, { "m", 1.6069 }, { "sigma_Angstrom", 3.5206},{"epsilon_over_k", 191.42}, {"BibTeXKey", "Gross-IECR-2001"} });
        nlohmann::json model = {
            {"coeffs", jcoeffs}
        };
        nlohmann::json j = {
            {"kind", "PCSAFT"},
            {"model", model}
        };
        std::string js = j.dump(1);
        int e1 = build_model(js.c_str(), &uuid, errmsg, errmsg_length);
        CAPTURE(errmsg);
        CAPTURE(js);
        REQUIRE(e1 == 0);
        int e2 = get_Arxy(uuid, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("multifluid pure with fluid path") {
        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["departure"] = nlohmann::json::array();
        jmodel["BIP"] = nlohmann::json::array();
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back(FLUIDDATAPATH+"/dev/fluids/Argon.json");
        
        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(js.c_str(), &uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("multifluid pure with fluid contents") {
        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back(load_a_JSON_file(FLUIDDATAPATH+"/dev/fluids/Argon.json"));
        jmodel["departure"] = nlohmann::json::array();
        jmodel["BIP"] = nlohmann::json::array();
        jmodel["flags"] = nlohmann::json::object();

        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        std::string js = j.dump(2);
        int e1 = build_model(js.c_str(), &uuid, errmsg, errmsg_length);
        int e2 = get_Arxy(uuid, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        int e3 = free_model(uuid, errmsg, errmsg_length);
        REQUIRE(e1 == 0);
        REQUIRE(e2 == 0);
        REQUIRE(e3 == 0);
        return val;
    };

    BENCHMARK("multifluid call") {
        int e2 = get_Arxy(uuidMF, 0, 1, 300.0, 3.0e-6, &(molefrac[0]), static_cast<int>(molefrac.size()), &val, errmsg, errmsg_length);
        REQUIRE(e2 == 0);
        return val;
    };
    
}
#else 
int main() {
}
#endif
