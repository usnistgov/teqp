#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_SQUAREWELL
#include "teqp/models/model_potentials/squarewell.hpp"
#endif
#ifndef DISABLE_EXP6
#include "teqp/models/model_potentials/exp6.hpp"
#endif
#ifndef DISABLE_2CLJF
#include "teqp/models/model_potentials/2center_ljf.hpp"
#endif
#ifndef DISABLE_MIE
#include "teqp/models/mie/mie.hpp"
#endif

namespace teqp{
    namespace cppinterface{
        using teqp::cppinterface::adapter::make_owned;
    
#ifndef DISABLE_SQUAREWELL
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SW_EspindolaHeredia2009(const nlohmann::json &spec){
            return make_owned(squarewell::EspindolaHeredia2009(spec.at("lambda")));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SW_EspindolaHeredia2009(const nlohmann::json &){
            throw teqp::NotImplementedError("The squarewell model from Espindola-Heredia has been disabled");
        }
#endif
    
#ifndef DISABLE_EXP6
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_EXP6_Kataoka1992(const nlohmann::json &spec){
            return make_owned(exp6::Kataoka1992(spec.at("alpha")));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_EXP6_Kataoka1992(const nlohmann::json &){
            throw teqp::NotImplementedError("The EXP-6 model from Espindola-Heredia has been disabled");
        }
#endif
    
#ifndef DISABLE_MIE
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Pohl2023(const nlohmann::json &spec){
            return make_owned(Mie::Mie6Pohl2023(spec.at("lambda_r")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Chaparro2023(const nlohmann::json &spec){
            return make_owned(FEANN::ChaparroJCP2023(spec.at("lambda_r"), spec.at("lambda_a")));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Pohl2023(const nlohmann::json &){
            throw teqp::NotImplementedError("The Mie model from Pohl has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_Mie_Chaparro2023(const nlohmann::json &){
            throw teqp::NotImplementedError("The Mie model from Chaparro has been disabled");
        }
#endif
    
#ifndef DISABLE_2CLJF
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model(spec.at("author"), spec.at("L^*")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Dipole(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model_dipole(spec.at("author"), spec.at("L^*"), spec.at("(mu^*)^2")));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Quadrupole(const nlohmann::json &spec){
            return make_owned(twocenterljf::build_two_center_model_quadrupole(spec.at("author"), spec.at("L^*"), spec.at("(Q^*)^2")));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF(const nlohmann::json &){
            throw teqp::NotImplementedError("The 2CLJF model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Dipole(const nlohmann::json &){
            throw teqp::NotImplementedError("The 2CLJF+dipole model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_2CLJF_Quadrupole(const nlohmann::json &){
            throw teqp::NotImplementedError("The 2CLJF+quadrupole model has been disabled");
        }
#endif
    }
}
