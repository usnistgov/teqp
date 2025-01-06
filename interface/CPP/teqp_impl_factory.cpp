#include "teqp/cpp/teqpcpp.hpp"

#include "teqp/models/vdW.hpp"
#include "teqp/models/cubics/simple_cubics.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

// This large block of schema definitions is populated by cmake
// at cmake configuration time
extern const nlohmann::json model_schema_library;

namespace teqp {
    namespace cppinterface {

        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRMie(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFT(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFTPureGrossSadowski2001(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SOFTSAFT(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_genericSAFT(const nlohmann::json &);
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004resid(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008resid(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2004idealgas(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_GERG2008idealgas(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LKP(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_association(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_activity(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_multifluid_ECS_HuberEly1994(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_AmmoniaWaterTillnerRoth();
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_TholJPCRD2016();
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_KolafaNezbeda1994();
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_LJ126_Johnson1993();
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SW_EspindolaHeredia2009(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_EXP6_Kataoka1992(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Pohl2023(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Chaparro2023(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Dipole(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Quadrupole(const nlohmann::json &);
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_CPA(const nlohmann::json &);
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_advancedPRaEres(const nlohmann::json &);
    
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_IdealHelmholtz(const nlohmann::json &);
    
        using makefunc = ModelPointerFactoryFunction;
        using namespace teqp::cppinterface::adapter;
    
        nlohmann::json get_model_schema(const std::string& kind) { return model_schema_library.at(kind); }

        // A list of factory functions that maps from EOS kind to factory function
        // The factory function returns a pointer to an AbstractModel (but which is an instance of a derived class)
        static std::unordered_map<std::string, makefunc> pointer_factory = {
            {"vdW1", [](const nlohmann::json& spec){ return make_owned(vdWEOS1(spec.at("a"), spec.at("b"))); }},
            {"vdW", [](const nlohmann::json& spec){ return make_owned(vdWEOS<double>(spec.at("Tcrit / K"), spec.at("pcrit / Pa"))); }},
            {"PR", [](const nlohmann::json& spec){ return make_owned(make_canonicalPR(spec));}},
            {"SRK", [](const nlohmann::json& spec){ return make_owned(make_canonicalSRK(spec));}},
            {"cubic", [](const nlohmann::json& spec){ return make_owned(make_generalizedcubic(spec));}},
            {"QCPRAasen", [](const nlohmann::json& spec){ return make_owned(QuantumCorrectedPR(spec));}},
            {"RKPRCismondi2005", [](const nlohmann::json& spec){ return make_owned(RKPRCismondi2005(spec));}},
            
            {"advancedPRaEres", [](const nlohmann::json& spec){ return make_advancedPRaEres(spec);}},
            
            // Implemented in their own compilation units to help with compilation time and memory
            // use. Having all the template instantations in one file is handy, but requires a huge amount of RAM
            // ---------
            {"SAFT-VR-Mie", [](const nlohmann::json& spec){ return make_SAFTVRMie(spec); }},
            
            {"PCSAFT", [](const nlohmann::json& spec){ return make_PCSAFT(spec); }},
            {"PCSAFTPureGrossSadowski2001", [](const nlohmann::json& spec){ return make_PCSAFTPureGrossSadowski2001(spec); }},
            {"SoftSAFT", [](const nlohmann::json& spec){ return make_SOFTSAFT(spec); }},
            {"genericSAFT", [](const nlohmann::json& spec){ return make_genericSAFT(spec); }},
            
            {"GERG2004resid", [](const nlohmann::json& spec){ return make_GERG2004resid(spec);}},
            {"GERG2008resid", [](const nlohmann::json& spec){ return make_GERG2008resid(spec);}},
            {"GERG2004idealgas", [](const nlohmann::json& spec){ return make_GERG2004idealgas(spec);}},
            {"GERG2008idealgas", [](const nlohmann::json& spec){ return make_GERG2008idealgas(spec);}},
            
            {"LKP", [](const nlohmann::json& spec){ return make_LKP(spec);}},
            
            {"multifluid", [](const nlohmann::json& spec){ return make_multifluid(spec);}},
            {"multifluid-ECS-HuberEly1994", [](const nlohmann::json& spec){ return make_multifluid_ECS_HuberEly1994(spec);}},
            {"multifluid-association", [](const nlohmann::json& spec){ return make_multifluid_association(spec);}},
            {"multifluid-activity", [](const nlohmann::json& spec){ return make_multifluid_activity(spec);}},
            {"AmmoniaWaterTillnerRoth", [](const nlohmann::json& /*spec*/){ return make_AmmoniaWaterTillnerRoth();}},
            
            {"LJ126_TholJPCRD2016", [](const nlohmann::json& /*spec*/){ return make_LJ126_TholJPCRD2016();}},
            {"LJ126_KolafaNezbeda1994", [](const nlohmann::json& /*spec*/){ return make_LJ126_KolafaNezbeda1994();}},
            {"LJ126_Johnson1993", [](const nlohmann::json& /*spec*/){ return make_LJ126_Johnson1993();}},
            {"SW_EspindolaHeredia2009",  [](const nlohmann::json& spec){ return make_SW_EspindolaHeredia2009(spec);}},
            {"EXP6_Kataoka1992", [](const nlohmann::json& spec){ return make_EXP6_Kataoka1992(spec); }},
            {"Mie_Pohl2023", [](const nlohmann::json& spec){ return make_Mie_Pohl2023(spec); }},
            {"Mie_Chaparro2023", [](const nlohmann::json& spec){ return make_Mie_Chaparro2023(spec); }},
            {"2CLJF", [](const nlohmann::json& spec){ return make_2CLJF(spec); }},
            {"2CLJF-Dipole", [](const nlohmann::json& spec){ return make_2CLJF_Dipole(spec); }},
            {"2CLJF-Quadrupole", [](const nlohmann::json& spec){ return make_2CLJF_Quadrupole(spec); }},
            
            {"CPA", [](const nlohmann::json& spec){ return make_CPA(spec); }},
            
            {"IdealHelmholtz", [](const nlohmann::json& spec){ return make_IdealHelmholtz(spec); }},
        };

        std::unique_ptr<teqp::cppinterface::AbstractModel> build_model_ptr(const nlohmann::json& json, const bool validate) {
            
            // Extract the name of the model and the model parameters
            std::string kind = json.at("kind");
            auto spec = json.at("model");
            
            auto itr = pointer_factory.find(kind);
            if (itr != pointer_factory.end()){
                bool do_validation = validate;
                if (json.contains("validate")){
                    do_validation = json["validate"];
                }
                if (do_validation){
                    if (model_schema_library.contains(kind)){
                        // This block is not thread-safe, needs a mutex or something
                        JSONValidator validator(model_schema_library.at(kind));
                        if (!validator.is_valid(spec)){
                            throw teqp::JSONValidationError(validator.get_validation_errors(spec));
                        }
                    }
                }
                return (itr->second)(spec);
            }
            else{
                throw std::invalid_argument("Don't understand \"kind\" of: " + kind);
            }
        }
    
        std::unique_ptr<AbstractModel> make_multifluid_model(const std::vector<std::string>& components, const std::string& root, const std::string& BIP, const nlohmann::json& flags, const std::string& departurepath) {
            return make_multifluid({{"components", components}, {"root",root}, {"BIP", BIP}, {"flags", flags}, {"departure", departurepath}});
        }
    
        std::unique_ptr<AbstractModel> make_model(const nlohmann::json& j, const bool validate) {
            return build_model_ptr(j, validate);
        }
    
        void add_model_pointer_factory_function(const std::string& key, ModelPointerFactoryFunction& func){
            if (pointer_factory.find(key) == pointer_factory.end()){
                pointer_factory[key] = func;
            }
            else{
                throw teqp::InvalidArgument("key is already present, overwriting is not currently allowed");
            }
        }
    }
}
