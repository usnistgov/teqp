/// *********************************************************************************
///                     EMSCRIPTEN (for javascript)
/// *********************************************************************************

#include <emscripten/bind.h>
using namespace emscripten;

#include <optional>

#include "teqp/models/vdW.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/json_builder.hpp"

//#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/derivs.hpp"
#include "teqp/algorithms/VLE.hpp"

std::string VLE(const std::string &JSON_model_string, const std::string &JSON_problem_string)
{
    using modeltype = decltype(teqp::build_model(nlohmann::json{})); // get the variant type returned from the factory

    try{
        // Unpack the model spec and build the model
        nlohmann::json j = nlohmann::json::parse(JSON_model_string);
        modeltype model = teqp::build_model(j);

        // Unpack the model spec
        nlohmann::json jprob = nlohmann::json::parse(JSON_problem_string);

        // Do the tracing
        double T = jprob.at("T / K");
        std::valarray<double> rhovecL_ = jprob.at("rhoL / m^3/mol");
        std::valarray<double> rhovecV_ = jprob.at("rhoV / m^3/mol");
        auto vec2eig = [](const std::valarray<double>&a){ return Eigen::Map<const Eigen::ArrayXd>(&(a[0]), a.size()).eval(); };
        auto rhovecL = vec2eig(rhovecL_);
        auto rhovecV = vec2eig(rhovecV_);

        // Lambda function to do the trace for the thing contained in the variant
        auto f = [&](const auto& model) -> nlohmann::json {
            return teqp::trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV);
        };
        auto out = std::visit(f, model);
        return out.dump();
    }
    catch(std::exception &e){
        //std::cout << "JSON parsing err: " << e.what() << std::endl;
        return nlohmann::json{{{"err", e.what()}}}.dump();
    }

    return "OK";
}

// Main binding code
EMSCRIPTEN_BINDINGS(teqp) {
    function("isotherm", &VLE);
}