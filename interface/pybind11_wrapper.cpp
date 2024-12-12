#include "nlohmann/json.hpp"
#include "pybind11_json/pybind11_json.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "teqp/ideal_eosterms.hpp"
#include "teqp/cpp/derivs.hpp"
#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/iteration.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/fwd.hpp"
#include "teqp/algorithms/ancillary_builder.hpp"
#include "teqp/models/multifluid_ecs_mutant.hpp"
#include "teqp/models/multifluid_association.hpp"
#include "teqp/models/multifluid/multifluid_activity.hpp"
#include "teqp/models/saft/genericsaft.hpp"
#include "teqp/algorithms/phase_equil.hpp"

#include "teqp/algorithms/pure_param_optimization.hpp"
using namespace teqp::algorithms::pure_param_optimization;

namespace py = pybind11;
using namespace py::literals;

#define stringify(A) #A
using namespace teqp;

void add_multifluid(py::module& m){
    // A single ancillary curve
    py::class_<VLEAncillary>(m, "VLEAncillary")
        .def(py::init<const nlohmann::json&>())
        .def("__call__", &VLEAncillary::operator())
        .def_readonly("T_r", &VLEAncillary::T_r)
        .def_readonly("Tmax", &VLEAncillary::Tmax)
        .def_readonly("Tmin", &VLEAncillary::Tmin)
        ;

    // The collection of VLE ancillary curves
    py::class_<MultiFluidVLEAncillaries>(m, "MultiFluidVLEAncillaries")
        .def(py::init<const nlohmann::json&>())
        .def_readonly("rhoL", &MultiFluidVLEAncillaries::rhoL)
        .def_readonly("rhoV", &MultiFluidVLEAncillaries::rhoV)
        .def_readonly("pL", &MultiFluidVLEAncillaries::pL)
        .def_readonly("pV", &MultiFluidVLEAncillaries::pV)
        ;

    // Expose some additional functions for working with the JSON data structures and resolving aliases
    m.def("get_BIPdep", &reducing::get_BIPdep, py::arg("BIPcollection"), py::arg("identifiers"), py::arg("flags") = nlohmann::json{});
    m.def("build_alias_map", &build_alias_map, py::arg("root"));
    m.def("collect_component_json", &collect_component_json, py::arg("identifiers"), py::arg("root"));
    m.def("get_departure_json", &get_departure_json, py::arg("name"), py::arg("root"));
}

template<typename TYPE>
const TYPE& get_typed(const py::object& o){
    using namespace teqp::cppinterface;
    using namespace teqp::cppinterface::adapter;
    // Cast Python-wrapped AbstractModel to the C++ AbstractModel
    const AbstractModel* am = o.cast<const AbstractModel *>();
    // Cast the C++ AbstractModel to the derived adapter class
    return get_model_cref<TYPE>(am);
}

template<typename TYPE>
TYPE& get_mutable_typed(py::object& o){
    using namespace teqp::cppinterface;
    using namespace teqp::cppinterface::adapter;
    // Cast Python-wrapped AbstractModel to the C++ AbstractModel
    AbstractModel* am = o.cast<AbstractModel *>();
    // Cast the C++ AbstractModel to the derived adapter class
    return get_model_ref<TYPE>(am);
}

void add_multifluid_mutant(py::module& m) {
    using namespace teqp;
    using namespace teqp::cppinterface;

    // A typedef for the base model
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));

    // Wrap the function for generating a multifluid mutant
    m.def("_build_multifluid_mutant", [](const py::object& o, const nlohmann::json &j){
        const MultiFluid& model = get_typed<MultiFluid>(o);
        auto mutant{build_multifluid_mutant(model, j)};
        return teqp::cppinterface::adapter::make_owned(mutant);
    });
}

void add_multifluid_ecs_mutant(py::module& m) {
    using namespace teqp;
    using namespace teqp::cppinterface;

    // A typedef for the base model
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));

    // Wrap the function for generating a multifluid mutant
    m.def("_build_multifluid_ecs_mutant", [](const py::object& o, const nlohmann::json& j) {
        const MultiFluid& model = get_typed<MultiFluid>(o);
        auto mutant{ build_multifluid_ecs_mutant(model, j) };
        return teqp::cppinterface::adapter::make_owned(mutant);
    });
}

template<typename TYPE>
void attach_multifluid_methods(py::object&obj){
    auto setattr = py::getattr(obj, "__setattr__");
    auto MethodType = py::module_::import("types").attr("MethodType");
    setattr("get_Tcvec", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).redfunc.Tc; }), obj));
    setattr("get_vcvec", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).redfunc.vc; }), obj));
    setattr("get_Tr", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<TYPE>(o).redfunc.get_Tr(molefrac); }, "self"_a, "molefrac"_a.noconvert()), obj));
    setattr("get_rhor", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<TYPE>(o).redfunc.get_rhor(molefrac); }, "self"_a, "molefrac"_a.noconvert()), obj));
    setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).get_meta(); }), obj));
    setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<TYPE>(o).set_meta(s); }, "self"_a, "s"_a), obj));
    setattr("get_alpharij", MethodType(py::cpp_function([](py::object& o, const int i, const int j, const double tau, const double delta){ return get_typed<TYPE>(o).dep.get_alpharij(i,j,tau,delta); }, "self"_a, "i"_a, "j"_a, "tau"_a, "delta"_a), obj));
    setattr("get_BIP", MethodType(py::cpp_function([](py::object& o, const std::size_t& i, const std::size_t& j, const std::string& key){ return get_typed<TYPE>(o).get_BIP(i,j,key); }, "self"_a, "i"_a, "j"_a, "key"_a), obj));
}
template<typename TYPE>
void attach_GERG_methods(py::object&obj){
    auto setattr = py::getattr(obj, "__setattr__");
    auto MethodType = py::module_::import("types").attr("MethodType");
    setattr("get_Tcvec", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).red.get_Tcvec(); }), obj));
    setattr("get_vcvec", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).red.get_vcvec(); }), obj));
    setattr("get_Tr", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<TYPE>(o).red.get_Tr(molefrac); }, "self"_a, "molefrac"_a.noconvert()), obj));
    setattr("get_rhor", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<TYPE>(o).red.get_rhor(molefrac); }, "self"_a, "molefrac"_a.noconvert()), obj));
//    setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<TYPE>(o).get_meta(); }), obj));
//    setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<TYPE>(o).set_meta(s); }, "self"_a, "s"_a), obj));
//    setattr("get_alpharij", MethodType(py::cpp_function([](py::object& o, const int i, const int j, const double tau, const double delta){ return get_typed<TYPE>(o).dep.get_alpharij(i,j,tau,delta); }, "self"_a, "i"_a, "j"_a, "tau"_a, "delta"_a), obj));
}

// Type index variables matching the model types, used for runtime attachment of model-specific methods
const std::type_index vdWEOS1_i{std::type_index(typeid(vdWEOS1))};
const std::type_index PCSAFT_i{std::type_index(typeid(PCSAFT_t))};
const std::type_index SAFTVRMie_i{std::type_index(typeid(SAFTVRMie_t))};
const std::type_index canonical_cubic_i{std::type_index(typeid(canonical_cubic_t))};
const std::type_index AmmoniaWaterTillnerRoth_i{std::type_index(typeid(AmmoniaWaterTillnerRoth))};
const std::type_index idealgas_i{std::type_index(typeid(idealgas_t))};
const std::type_index multifluid_i{std::type_index(typeid(multifluid_t))};
const std::type_index multifluidmutant_i{std::type_index(typeid(multifluidmutant_t))};
const std::type_index SW_EspindolaHeredia2009_i{std::type_index(typeid(SW_EspindolaHeredia2009_t))};
const std::type_index EXP6_Kataoka1992_i{std::type_index(typeid(EXP6_Kataoka1992_t))};
const std::type_index twocenterLJF_i{std::type_index(typeid(twocenterLJF_t))};
const std::type_index QuantumPR_i{std::type_index(typeid(QuantumPR_t))};
const std::type_index advancedPRaEres_i{std::type_index(typeid(advancedPRaEres_t))};
const std::type_index RKPRCismondi2005_i{std::type_index(typeid(RKPRCismondi2005_t))};
const std::type_index GERG2004ResidualModel_i{std::type_index(typeid(GERG2004::GERG2004ResidualModel))};
const std::type_index GERG2008ResidualModel_i{std::type_index(typeid(GERG2008::GERG2008ResidualModel))};
using CPA_t = decltype(teqp::CPA::CPAfactory(""));
const std::type_index CPA_i{std::type_index(typeid(CPA_t))};
const std::type_index genericSAFT_i{std::type_index(typeid(teqp::saft::genericsaft::GenericSAFT))};
const std::type_index MultiFluidAssociation_i{std::type_index(typeid(MultifluidPlusAssociation))};
const std::type_index MultiFluidActivity_i{std::type_index(typeid(teqp::multifluid::multifluid_activity::MultifluidPlusActivity))};

/**
 At runtime we can add additional model-specific methods that only apply for a particular model.  We take in a Python-wrapped
 object and use runtime instrospection to figure out what kind of model it is. Then, we attach methods that are evaluated
 ad *runtime* to call methods of the instance. This is tricky, although it works just fine.
 
 */
// You cannot know at runtime what is contained in the model so you must iterate
// over possible model types and attach methods accordingly
void attach_model_specific_methods(py::object& obj){
    using namespace teqp::cppinterface;
    
    // Get things from Python (these are just for convenience to save typing)
    auto MethodType = py::module_::import("types").attr("MethodType");
    auto setattr = py::getattr(obj, "__setattr__");
    
    AbstractModel* am = obj.cast<AbstractModel *>();
    if (am == nullptr){
        throw std::invalid_argument("Bad cast of argument to C++ AbstractModel type");
    }
    const auto& index = am->get_type_index();
    
    if (index == vdWEOS1_i){
        setattr("get_a", MethodType(py::cpp_function([](py::object& o){ return get_typed<vdWEOS1>(o).get_a(); }), obj));
        setattr("get_b", MethodType(py::cpp_function([](py::object& o){ return get_typed<vdWEOS1>(o).get_b(); }), obj));
    }
    else if (index == PCSAFT_i){
        setattr("get_m", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_m(); }), obj));
        setattr("get_sigma_Angstrom", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_sigma_Angstrom(); }), obj));
        setattr("get_epsilon_over_k_K", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_epsilon_over_k_K(); }), obj));
        setattr("get_names", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_names(); }), obj));
        setattr("get_BibTeXKeys", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_BibTeXKeys(); }), obj));
        setattr("max_rhoN", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<PCSAFT_t>(o).max_rhoN(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_kmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<PCSAFT_t>(o).get_kmat(); }), obj));
    }
    else if (index == SAFTVRMie_i){
        setattr("get_names", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_names(); }), obj));
        setattr("get_BibTeXKeys", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_BibTeXKeys(); }), obj));
        setattr("get_m", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_m(); }), obj));
        setattr("get_sigma_Angstrom", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_sigma_Angstrom(); }), obj));
        setattr("get_sigma_m", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_sigma_m(); }), obj));
        setattr("get_epsilon_over_k_K", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_epsilon_over_k_K(); }), obj));
        setattr("get_lambda_r", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_lambda_r(); }), obj));
        setattr("get_lambda_a", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_lambda_a(); }), obj));
        
//        setattr("max_rhoN", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<SAFTVRMie_t>(o).max_rhoN(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_kmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).get_kmat(); }), obj));
        setattr("has_polar", MethodType(py::cpp_function([](py::object& o){ return get_typed<SAFTVRMie_t>(o).has_polar(); }), obj));
        setattr("get_core_calcs", MethodType(py::cpp_function([](py::object& o, double T, double rhomolar, REArrayd& molefrac){ return get_typed<SAFTVRMie_t>(o).get_core_calcs(T, rhomolar, molefrac); }, "self"_a, "T"_a, "rhomolar"_a, "molefrac"_a), obj));
        
    }
    else if (index == canonical_cubic_i){
        setattr("get_a", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<canonical_cubic_t>(o).get_a(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_b", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<canonical_cubic_t>(o).get_b(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("superanc_rhoLV", MethodType(py::cpp_function([](py::object& o, double T, std::optional<std::size_t> ifluid){ return get_typed<canonical_cubic_t>(o).superanc_rhoLV(T, ifluid); }, "self"_a, "T"_a, py::arg_v("ifluid", std::nullopt, "None")), obj));
        setattr("get_kmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<canonical_cubic_t>(o).get_kmat(); }), obj));
        setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<canonical_cubic_t>(o).get_meta(); }), obj));
        setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<canonical_cubic_t>(o).set_meta(s); }, "self"_a, "s"_a), obj));
    }
    else if (index == QuantumPR_i){
        setattr("get_ab", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<QuantumPR_t>(o).get_ab(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_c", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<QuantumPR_t>(o).get_c(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("superanc_rhoLV", MethodType(py::cpp_function([](py::object& o, double T, std::optional<std::size_t> ifluid){ return get_typed<QuantumPR_t>(o).superanc_rhoLV(T, ifluid); }, "self"_a, "T"_a, py::arg_v("ifluid", std::nullopt, "None")), obj));
        setattr("get_kmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<QuantumPR_t>(o).get_kmat(); }), obj));
        setattr("get_lmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<QuantumPR_t>(o).get_lmat(); }), obj));
        setattr("get_Tc_K", MethodType(py::cpp_function([](py::object& o){ return get_typed<QuantumPR_t>(o).get_Tc_K(); }), obj));
        setattr("get_pc_Pa", MethodType(py::cpp_function([](py::object& o){ return get_typed<QuantumPR_t>(o).get_pc_Pa(); }), obj));
//        setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<canonical_cubic_t>(o).get_meta(); }), obj));
//        setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<canonical_cubic_t>(o).set_meta(s); }, "self"_a, "s"_a), obj));
    }
    else if (index == advancedPRaEres_i){
        setattr("get_a", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<advancedPRaEres_t>(o).get_a(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_b", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<advancedPRaEres_t>(o).get_b(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("superanc_rhoLV", MethodType(py::cpp_function([](py::object& o, double T, std::size_t i){ return get_typed<advancedPRaEres_t>(o).superanc_rhoLV(T, i); }, "self"_a, "T"_a, "ifluid"_a), obj));
        setattr("get_Tc_K", MethodType(py::cpp_function([](py::object& o){ return get_typed<advancedPRaEres_t>(o).get_Tc_K(); }), obj));
        setattr("get_pc_Pa", MethodType(py::cpp_function([](py::object& o){ return get_typed<advancedPRaEres_t>(o).get_pc_Pa(); }), obj));
//        setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<advancedPRaEres_t>(o).get_meta(); }), obj));
//        setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<advancedPRaEres_t>(o).set_meta(s); }, "self"_a, "s"_a), obj));
    }
    else if (index == RKPRCismondi2005_i){
        setattr("get_ab", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){ return get_typed<RKPRCismondi2005_t>(o).get_ab(T, molefrac); }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("get_delta_1", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_delta_1(); }), obj));
        setattr("get_Tc_K", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_Tc_K(); }), obj));
        setattr("get_pc_Pa", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_pc_Pa(); }), obj));
        setattr("get_k", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_k(); }), obj));
        
        setattr("get_kmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_kmat(); }), obj));
        setattr("get_lmat", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_lmat(); }), obj));
//        setattr("get_meta", MethodType(py::cpp_function([](py::object& o){ return get_typed<RKPRCismondi2005_t>(o).get_meta(); }), obj));
//        setattr("set_meta", MethodType(py::cpp_function([](py::object& o, const std::string& s){ return get_mutable_typed<RKPRCismondi2005_t>(o).set_meta(s); }, "self"_a, "s"_a), obj));
    }
    else if (index == AmmoniaWaterTillnerRoth_i){
        setattr("TcNH3", get_typed<AmmoniaWaterTillnerRoth>(obj).TcNH3);
        setattr("vcNH3", get_typed<AmmoniaWaterTillnerRoth>(obj).vcNH3);
        setattr("get_Tr", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<AmmoniaWaterTillnerRoth>(o).get_Treducing(molefrac); }, "self"_a, "molefrac"_a), obj));
        setattr("get_rhor", MethodType(py::cpp_function([](py::object& o, REArrayd& molefrac){ return get_typed<AmmoniaWaterTillnerRoth>(o).get_rhoreducing(molefrac); }, "self"_a, "molefrac"_a), obj));
        setattr("alphar_departure", MethodType(py::cpp_function([](py::object& o, const double tau, const double delta, const double xNH3){ return get_typed<AmmoniaWaterTillnerRoth>(o).alphar_departure(tau, delta, xNH3); }, "self"_a, "tau"_a, "delta"_a, "xNH3"_a), obj));
        setattr("dalphar_departure_ddelta", MethodType(py::cpp_function([](py::object& o, const double tau, const double delta, const double xNH3){
            // Calculate with complex step derivatives
            double h = 1e-100;
            const auto& m = get_typed<AmmoniaWaterTillnerRoth>(o);
            std::complex<double> delta_(delta, h);
            return m.alphar_departure(tau, delta_, xNH3).imag()/h;
        }, "self"_a, "tau"_a, "delta"_a, "xNH3"_a), obj));
    }
    else if (index == multifluid_i){
        attach_multifluid_methods<multifluid_t>(obj);
        setattr("build_ancillaries", MethodType(py::cpp_function([](py::object& o, std::optional<int> i = std::nullopt){
            const auto& c = get_typed<multifluid_t>(o);
            auto N = c.redfunc.Tc.size();
            if (!i && c.redfunc.Tc.size() != 1) {
                throw teqp::InvalidArgument("Can only build ancillaries for pure fluids, or provide the index of fluid you would like to construct");
            }
            auto k = i.value_or(0);
            if (k > N-1) {
                throw teqp::InvalidArgument("Cannot obtain the EOS at index"+std::to_string(k)+"; length is "+std::to_string(N));
            }
            auto jancillaries = nlohmann::json::parse(c.get_meta()).at("pures")[k].at("ANCILLARIES");
            return teqp::MultiFluidVLEAncillaries(jancillaries);
        }, "self"_a, py::arg_v("i", std::nullopt, "None")), obj));
    }
    else if (index == GERG2004ResidualModel_i){
        attach_GERG_methods<GERG2004::GERG2004ResidualModel>(obj);
    }
    else if (index == GERG2008ResidualModel_i){
        attach_GERG_methods<GERG2008::GERG2008ResidualModel>(obj);
    }
    else if (index == idealgas_i){
        // Here X-Macros are used to create functions like get_Aig00, get_Aig01, ....
        #define X(i,j) setattr(stringify(get_Aig ## i ## j), MethodType(py::cpp_function([](py::object& o, double T, double rho, REArrayd& molefrac){ \
                using tdx = teqp::TDXDerivatives<idealgas_t, double, REArrayd>; \
                return tdx::template get_Aigxy<i,j>(get_typed<idealgas_t>(o), T, rho, molefrac); \
                }, "self"_a, "T"_a, "rho"_a, "molefrac"_a.noconvert()), obj));
            ARXY_args
        #undef X
    }
    else if (index == multifluidmutant_i){
        attach_multifluid_methods<multifluidmutant_t>(obj);
    }
    else if (index == SW_EspindolaHeredia2009_i){
        // Have to use a method because lambda is a reserved word in Python
        setattr("get_lambda", MethodType(py::cpp_function([](py::object& o){ return get_typed<SW_EspindolaHeredia2009_t>(o).get_lambda(); }, "self"_a), obj));
    }
    else if (index == EXP6_Kataoka1992_i){
        setattr("alpha", get_typed<EXP6_Kataoka1992_t>(obj).get_alpha());
    }
    else if (index == twocenterLJF_i){
        setattr("Lstar", get_typed<twocenterLJF_t>(obj).L);
        setattr("mustar_sq", get_typed<twocenterLJF_t>(obj).mu_sq);
    }
    else if (index == CPA_i){
        setattr("get_assoc_calcs", MethodType(py::cpp_function([](py::object& o, double T, double rhomolar, REArrayd& molefrac){ return get_typed<CPA_t>(o).assoc.get_assoc_calcs(T, rhomolar, molefrac); }, "self"_a, "T"_a, "rhomolar"_a, "molefrac"_a), obj));
    }
    else if (index == genericSAFT_i){
        setattr("get_assoc_calcs", MethodType(py::cpp_function([](py::object& o, double T, double rhomolar, REArrayd& molefrac){
            const auto& assocoptvariant = get_typed<teqp::saft::genericsaft::GenericSAFT>(o).association;
            if (!assocoptvariant){
                throw teqp::InvalidArgument("No association term is available");
            }
            return std::visit([&](const auto& a){ return a.get_assoc_calcs(T, rhomolar, molefrac); }, assocoptvariant.value());
        }, "self"_a, "T"_a, "rhomolar"_a, "molefrac"_a), obj));
    }
    else if (index == MultiFluidAssociation_i){
        setattr("get_assoc_calcs", MethodType(py::cpp_function([](py::object& o, double T, double rhomolar, REArrayd& molefrac){
            return get_typed<MultifluidPlusAssociation>(o).get_association().get_assoc_calcs(T, rhomolar, molefrac);
        }, "self"_a, "T"_a, "rhomolar"_a, "molefrac"_a), obj));
    }
    else if (index == MultiFluidActivity_i){
        setattr("calc_gER_over_RT", MethodType(py::cpp_function([](py::object& o, double T, REArrayd& molefrac){
            return get_typed<teqp::multifluid::multifluid_activity::MultifluidPlusActivity>(o).calc_gER_over_RT(T, molefrac);
        }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("calc_lngamma_resid", MethodType(py::cpp_function([](py::object& o, double T, EArrayd& molefrac){
            return get_typed<teqp::multifluid::multifluid_activity::MultifluidPlusActivity>(o).calc_lngamma_resid(T, molefrac);
        }, "self"_a, "T"_a, "molefrac"_a), obj));
        setattr("calc_lngamma_comb", MethodType(py::cpp_function([](py::object& o, double T, EArrayd& molefrac) -> EArrayd{
            return get_typed<teqp::multifluid::multifluid_activity::MultifluidPlusActivity>(o).calc_lngamma_comb(T, molefrac);
        }, "self"_a, "T"_a, "molefrac"_a), obj));
    }
};

/// Instantiate "instances" of models (really wrapped Python versions of the models), and then attach all derivative methods
void init_teqp(py::module& m) {
    
    // The options class for critical tracer, not tied to a particular model
    py::class_<TCABOptions>(m, "TCABOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &TCABOptions::abs_err)
        .def_readwrite("rel_err", &TCABOptions::rel_err)
        .def_readwrite("init_dt", &TCABOptions::init_dt)
        .def_readwrite("init_c", &TCABOptions::init_c)
        .def_readwrite("max_dt", &TCABOptions::max_dt)
        .def_readwrite("T_tol", &TCABOptions::T_tol)
        .def_readwrite("small_T_count", &TCABOptions::small_T_count)
        .def_readwrite("max_step_count", &TCABOptions::max_step_count)
        .def_readwrite("skip_dircheck_count", &TCABOptions::skip_dircheck_count)
        .def_readwrite("integration_order", &TCABOptions::integration_order)
        .def_readwrite("calc_stability", &TCABOptions::calc_stability)
        .def_readwrite("stability_rel_drho", &TCABOptions::stability_rel_drho)
        .def_readwrite("verbosity", &TCABOptions::verbosity)
        .def_readwrite("polish", &TCABOptions::polish)
        .def_readwrite("polish_reltol_rho", &TCABOptions::polish_reltol_rho)
        .def_readwrite("polish_reltol_T", &TCABOptions::polish_reltol_T)
        .def_readwrite("pure_endpoint_polish", &TCABOptions::pure_endpoint_polish)
        .def_readwrite("polish_exception_on_fail", &TCABOptions::polish_exception_on_fail)
    ;
    
    // The options class for isotherm tracer, not tied to a particular model
    py::class_<TVLEOptions>(m, "TVLEOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &TVLEOptions::abs_err)
        .def_readwrite("rel_err", &TVLEOptions::rel_err)
        .def_readwrite("init_dt", &TVLEOptions::init_dt)
        .def_readwrite("init_c", &TVLEOptions::init_c)
        .def_readwrite("max_dt", &TVLEOptions::max_dt)
        .def_readwrite("p_termination", &TVLEOptions::p_termination)
        .def_readwrite("crit_termination", &TVLEOptions::crit_termination)
        .def_readwrite("max_steps", &TVLEOptions::max_steps)
        .def_readwrite("integration_order", &TVLEOptions::integration_order)
        .def_readwrite("polish", &TVLEOptions::polish)
        .def_readwrite("polish_reltol_rho", &TVLEOptions::polish_reltol_rho)
        .def_readwrite("polish_exception_on_fail", &TVLEOptions::polish_exception_on_fail)
        .def_readwrite("verbosity", &TVLEOptions::verbosity)
        .def_readwrite("calc_criticality", &TVLEOptions::calc_criticality)
        .def_readwrite("terminate_unstable", &TVLEOptions::terminate_unstable)
    ;
    
    // The options class for isobar tracer, not tied to a particular model
    py::class_<PVLEOptions>(m, "PVLEOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &PVLEOptions::abs_err)
        .def_readwrite("rel_err", &PVLEOptions::rel_err)
        .def_readwrite("init_dt", &PVLEOptions::init_dt)
        .def_readwrite("init_c", &PVLEOptions::init_c)
        .def_readwrite("max_dt", &PVLEOptions::max_dt)
        .def_readwrite("crit_termination", &PVLEOptions::crit_termination)
        .def_readwrite("max_steps", &PVLEOptions::max_steps)
        .def_readwrite("integration_order", &PVLEOptions::integration_order)
        .def_readwrite("polish", &PVLEOptions::polish)
        .def_readwrite("polish_reltol_rho", &PVLEOptions::polish_reltol_rho)
        .def_readwrite("polish_exception_on_fail", &PVLEOptions::polish_exception_on_fail)
        .def_readwrite("verbosity", &PVLEOptions::verbosity)
        .def_readwrite("calc_criticality", &PVLEOptions::calc_criticality)
        .def_readwrite("terminate_unstable", &PVLEOptions::terminate_unstable)
    ;
    
    // The options class for the finder of VLLE solutions from VLE tracing, not tied to a particular model
    py::class_<VLLE::VLLEFinderOptions>(m, "VLLEFinderOptions")
        .def(py::init<>())
        .def_readwrite("max_steps", &VLLE::VLLEFinderOptions::max_steps)
        .def_readwrite("rho_trivial_threshold", &VLLE::VLLEFinderOptions::rho_trivial_threshold)
    ;
    
    // The options class for the finder of VLLE solutions from VLE tracing, not tied to a particular model
    py::class_<VLLE::VLLETracerOptions>(m, "VLLETracerOptions")
        .def(py::init<>())
        .def_readwrite("max_step_count", &VLLE::VLLETracerOptions::max_step_count)
        .def_readwrite("abs_err", &VLLE::VLLETracerOptions::abs_err)
        .def_readwrite("rel_err", &VLLE::VLLETracerOptions::rel_err)
        .def_readwrite("verbosity", &VLLE::VLLETracerOptions::verbosity)
        .def_readwrite("init_dT", &VLLE::VLLETracerOptions::init_dT)
        .def_readwrite("max_dT", &VLLE::VLLETracerOptions::max_dT)
        .def_readwrite("polish", &VLLE::VLLETracerOptions::polish)
        .def_readwrite("max_polish_steps", &VLLE::VLLETracerOptions::max_polish_steps)
        .def_readwrite("terminate_composition", &VLLE::VLLETracerOptions::terminate_composition)
        .def_readwrite("terminate_composition_tol", &VLLE::VLLETracerOptions::terminate_composition_tol)
        .def_readwrite("T_limit", &VLLE::VLLETracerOptions::T_limit)
        .def_readwrite("max_step_retries", &VLLE::VLLETracerOptions::max_step_retries)
    ;
    
    py::class_<MixVLETpFlags>(m, "MixVLETpFlags")
        .def(py::init<>())
        .def_readwrite("atol", &MixVLETpFlags::atol)
        .def_readwrite("reltol", &MixVLETpFlags::reltol)
        .def_readwrite("axtol", &MixVLETpFlags::axtol)
        .def_readwrite("relxtol", &MixVLETpFlags::relxtol)
        .def_readwrite("maxiter", &MixVLETpFlags::maxiter)
    ;
    
    py::class_<MixVLEpxFlags>(m, "MixVLEpxFlags")
        .def(py::init<>())
        .def_readwrite("atol", &MixVLEpxFlags::atol)
        .def_readwrite("reltol", &MixVLEpxFlags::reltol)
        .def_readwrite("axtol", &MixVLEpxFlags::axtol)
        .def_readwrite("relxtol", &MixVLEpxFlags::relxtol)
        .def_readwrite("maxiter", &MixVLEpxFlags::maxiter)
    ;
    
    using namespace teqp::cppinterface;
    // The Jacobian and value matrices for Newton-Raphson
    py::class_<IterationMatrices>(m, "IterationMatrices")
        .def(py::init<>())
        .def_readonly("J", &IterationMatrices::J)
        .def_readonly("v", &IterationMatrices::v)
        .def_readonly("vars", &IterationMatrices::vars)
    ;
    
    py::enum_<VLE_return_code>(m, "VLE_return_code")
        .value("unset", VLE_return_code::unset)
        .value("xtol_satisfied", VLE_return_code::xtol_satisfied)
        .value("functol_satisfied", VLE_return_code::functol_satisfied)
        .value("maxiter_met", VLE_return_code::maxiter_met)
        .value("maxfev_met", VLE_return_code::maxfev_met)
        .value("notfinite_step", VLE_return_code::notfinite_step)
    ;
    
    py::class_<MixVLEReturn>(m, "MixVLEReturn")
        .def(py::init<>())
        .def_readonly("success", &MixVLEReturn::success)
        .def_readonly("message", &MixVLEReturn::message)
        .def_readonly("rhovecL", &MixVLEReturn::rhovecL)
        .def_readonly("rhovecV", &MixVLEReturn::rhovecV)
        .def_readonly("return_code", &MixVLEReturn::return_code)
        .def_readonly("num_iter", &MixVLEReturn::num_iter)
        .def_readonly("T", &MixVLEReturn::T)
        .def_readonly("num_fev", &MixVLEReturn::num_fev)
        .def_readonly("r", &MixVLEReturn::r)
        .def_readonly("initial_r", &MixVLEReturn::initial_r)
    ;
    
    using namespace teqp::PCSAFT;
    py::class_<SAFTCoeffs>(m, "SAFTCoeffs")
        .def(py::init<>())
        .def_readwrite("name", &SAFTCoeffs::name)
        .def_readwrite("m", &SAFTCoeffs::m)
        .def_readwrite("sigma_Angstrom", &SAFTCoeffs::sigma_Angstrom)
        .def_readwrite("epsilon_over_k", &SAFTCoeffs::epsilon_over_k)
        .def_readwrite("BibTeXKey", &SAFTCoeffs::BibTeXKey)
        .def_readwrite("mustar2", &SAFTCoeffs::mustar2)
        .def_readwrite("nmu", &SAFTCoeffs::nmu)
        .def_readwrite("Qstar2", &SAFTCoeffs::Qstar2)
        .def_readwrite("nQ", &SAFTCoeffs::nQ)
    ;
    
    m.def("convert_CoolProp_idealgas", [](const std::string &path, int index){return convert_CoolProp_idealgas(path, index);});
    
    add_multifluid(m);
    add_multifluid_mutant(m);
    add_multifluid_ecs_mutant(m);
    
    using am = teqp::cppinterface::AbstractModel;
    py::class_<AbstractModel, std::unique_ptr<AbstractModel>>(m, "AbstractModel", py::dynamic_attr())
    
        .def("get_R", &am::get_R, "molefrac"_a.noconvert())
    
        .def("get_reducing_density", &am::get_reducing_density,  "molefrac"_a.noconvert())
        .def("get_reducing_temperature", &am::get_reducing_temperature,  "molefrac"_a.noconvert())
    
        .def("get_B2vir", &am::get_B2vir, "T"_a, "molefrac"_a.noconvert())
        .def("get_Bnvir", &am::get_Bnvir, "Nderiv"_a, "T"_a, "molefrac"_a.noconvert())
        .def("get_dmBnvirdTm", &am::get_dmBnvirdTm, "Nderiv"_a, "NTderiv"_a, "T"_a, "molefrac"_a.noconvert())
        .def("get_B12vir", &am::get_B12vir, "T"_a, "molefrac"_a.noconvert())
    
        .def("get_ATrhoXi", &am::get_ATrhoXi, "T"_a, "NT"_a, "rhomolar"_a, "Nrho"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a)
        .def("get_ATrhoXiXj", &am::get_ATrhoXiXj, "T"_a, "NT"_a, "rhomolar"_a, "Nrho"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a, "j"_a, "NXj"_a)
        .def("get_ATrhoXiXjXk", &am::get_ATrhoXiXjXk, "T"_a, "NT"_a, "rhomolar"_a, "Nrho"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a, "j"_a, "NXj"_a, "k"_a, "NXk"_a)
        .def("get_AtaudeltaXi", &am::get_AtaudeltaXi, "tau"_a, "Ntau"_a, "delta"_a, "Ndelta"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a)
        .def("get_AtaudeltaXiXj", &am::get_AtaudeltaXiXj, "tau"_a, "Ntau"_a, "delta"_a, "Ndelta"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a, "j"_a, "NXj"_a)
        .def("get_AtaudeltaXiXjXk", &am::get_AtaudeltaXiXjXk, "tau"_a, "Ntau"_a, "delta"_a, "Ndelta"_a, "molefrac"_a.noconvert(), "i"_a, "NXi"_a, "j"_a, "NXj"_a, "k"_a, "NXk"_a)
    
        .def("get_Arxy", &am::get_Arxy, "NT"_a, "ND"_a, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    // Here X-Macros are used to create functions like get_Ar00, get_Ar01, ....
#define X(i,j) .def(stringify(get_Ar ## i ## j), &am::get_Ar ## i ## j, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    ARXY_args
#undef X
    // And like get_Ar01n, get_Ar02n, ....
#define X(i) .def(stringify(get_Ar0 ## i ## n), &am::get_Ar0 ## i ## n, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    AR0N_args
#undef X
    // And like get_Ar10n, get_Ar20n, ....
#define X(i) .def(stringify(get_Ar ## i ## 0n), &am::get_Ar ## i ## 0n, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    ARN0_args
#undef X
        .def("get_neff", &am::get_neff, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    
    // Methods that come from the isochoric derivatives formalism
        .def("get_pr", &am::get_pr, "T"_a, "rhovec"_a.noconvert())
        .def("get_splus", &am::get_splus, "T"_a, "rhovec"_a.noconvert())
        .def("build_Psir_Hessian_autodiff", &am::build_Psir_Hessian_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("build_Psi_Hessian_autodiff", &am::build_Psi_Hessian_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("build_Psir_gradient_autodiff", &am::build_Psir_gradient_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("build_d2PsirdTdrhoi_autodiff", &am::build_d2PsirdTdrhoi_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("get_chempotVLE_autodiff", &am::get_chempotVLE_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("get_dchempotdT_autodiff", &am::get_dchempotdT_autodiff, "T"_a, "rhovec"_a.noconvert())
        .def("get_fugacity_coefficients", &am::get_fugacity_coefficients, "T"_a, "rhovec"_a.noconvert())
        .def("get_partial_molar_volumes", &am::get_partial_molar_volumes, "T"_a, "rhovec"_a.noconvert())
    
        .def("get_deriv_mat2", &am::get_deriv_mat2, "T"_a, "rho"_a, "molefrac"_a.noconvert())
    
    // Routines related to pure fluid critical point calculation
        .def("get_pure_critical_conditions_Jacobian", &am::get_pure_critical_conditions_Jacobian, "T"_a, "rho"_a, py::arg_v("alternative_pure_index", std::nullopt, "None"), py::arg_v("alternative_length", std::nullopt, "None"))
        .def("solve_pure_critical", &am::solve_pure_critical, "T"_a, "rho"_a, py::arg_v("flags", std::nullopt, "None"))
        .def("extrapolate_from_critical", &am::extrapolate_from_critical, "Tc"_a, "rhoc"_a, "T"_a, py::arg_v("molefrac", std::nullopt, "None"))
    
    // Routines related to binary mixture critical curve tracing
        .def("trace_critical_arclength_binary", &am::trace_critical_arclength_binary, "T0"_a, "rhovec0"_a, py::arg_v("path", std::nullopt, "None"), py::arg_v("options", std::nullopt, "None"))
        .def("get_criticality_conditions", &am::get_criticality_conditions, "T"_a, "rhovec"_a.noconvert())
        .def("eigen_problem", &am::eigen_problem, "T"_a, "rhovec"_a, py::arg_v("alignment_v0", std::nullopt, "None"))
        .def("get_minimum_eigenvalue_Psi_Hessian", &am::get_minimum_eigenvalue_Psi_Hessian, "T"_a, "rhovec"_a.noconvert())
        .def("get_drhovec_dT_crit", &am::get_drhovec_dT_crit, "T"_a, "rhovec"_a.noconvert())
        .def("get_dp_dT_crit", &am::get_dp_dT_crit, "T"_a, "rhovec"_a.noconvert())
    
        .def("pure_VLE_T", &am::pure_VLE_T, "T"_a, "rhoL"_a, "rhoV"_a, "max_iter"_a, py::arg_v("molefrac", std::nullopt, "None"))
        .def("dpsatdT_pure", &am::dpsatdT_pure, "T"_a, "rhoL"_a, "rhoV"_a)
    
        .def("get_drhovecdp_Tsat", &am::get_drhovecdp_Tsat, "T"_a, "rhovecL"_a.noconvert(), "rhovecV"_a.noconvert())
        .def("get_drhovecdT_psat", &am::get_drhovecdT_psat, "T"_a, "rhovecL"_a.noconvert(), "rhovecV"_a.noconvert())
        .def("get_dpsat_dTsat_isopleth", &am::get_dpsat_dTsat_isopleth, "T"_a, "rhovecL"_a.noconvert(), "rhovecV"_a.noconvert())
    
        .def("trace_VLE_isotherm_binary", &am::trace_VLE_isotherm_binary, "T"_a, "rhovecL0"_a.noconvert(), "rhovecV0"_a.noconvert(), py::arg_v("options", std::nullopt, "None"))
        .def("trace_VLE_isobar_binary", &am::trace_VLE_isobar_binary, "p"_a, "T0"_a, "rhovecL0"_a.noconvert(), "rhovecV0"_a.noconvert(), py::arg_v("options", std::nullopt, "None"))
        .def("mix_VLE_Tx", &am::mix_VLE_Tx, "T"_a, "rhovecL0"_a.noconvert(), "rhovecV0"_a.noconvert(), "xspec"_a.noconvert(), "atol"_a, "reltol"_a, "axtol"_a, "relxtol"_a, "maxiter"_a)
        .def("mix_VLE_Tp", &am::mix_VLE_Tp, "T"_a, "p_given"_a, "rhovecL0"_a.noconvert(), "rhovecV0"_a.noconvert(), py::arg_v("options", std::nullopt, "None"))
        .def("mixture_VLE_px", &am::mixture_VLE_px, "p_spec"_a, "xmolar_spec"_a.noconvert(), "T0"_a, "rhovecL0"_a.noconvert(), "rhovecV0"_a.noconvert(), py::arg_v("options", std::nullopt, "None"))
    
        .def("mix_VLLE_T", &am::mix_VLLE_T, "T"_a, "rhovecVinit"_a.noconvert(), "rhovecL1init"_a.noconvert(), "rhovecL2init"_a.noconvert(), "atol"_a, "reltol"_a, "axtol"_a, "relxtol"_a, "maxiter"_a)
        .def("find_VLLE_T_binary", &am::find_VLLE_T_binary, "traces"_a, py::arg_v("options", std::nullopt, "None"))
        .def("find_VLLE_p_binary", &am::find_VLLE_p_binary, "traces"_a, py::arg_v("options", std::nullopt, "None"))
        .def("trace_VLLE_binary", &am::trace_VLLE_binary, "T"_a, "rhovecV"_a.noconvert(), "rhovecL1"_a.noconvert(), "rhovecL2"_a.noconvert(), py::arg_v("options", std::nullopt, "None"))
    ;
    
    m.def("_make_model", &teqp::cppinterface::make_model, "json_data"_a, py::arg_v("validate", true));
    m.def("attach_model_specific_methods", &attach_model_specific_methods);
    m.def("build_ancillaries", &teqp::ancillaries::build_ancillaries, "model"_a, "Tc"_a, "rhoc"_a, "Tmin"_a, py::arg_v("flags", std::nullopt, "None"));
    m.def("convert_FLD", [](const std::string& component, const std::string& name){ return RPinterop::FLDfile(component).make_json(name); },
          "component"_a, "name"_a);
    m.def("convert_HMXBNC", [](const std::string& path){ return RPinterop::HMXBNCfile(path).make_jsons(); }, "path"_a);
    
    {
        using namespace teqp::algorithms::phase_equil;
        auto m_phaseequil = m.def_submodule("phaseequil", "Routines for phase equilibrium");
        
        // Specification options
        py::class_<AbstractSpecification, std::shared_ptr<AbstractSpecification>>(m_phaseequil, "AbstractSpecification");
        py::class_<TSpecification, AbstractSpecification, std::shared_ptr<TSpecification>>(m_phaseequil, "TSpecification").def(py::init<double>());
        py::class_<PSpecification, AbstractSpecification, std::shared_ptr<PSpecification>>(m_phaseequil, "PSpecification").def(py::init<double>());
        py::class_<BetaSpecification, AbstractSpecification, std::shared_ptr<BetaSpecification>>(m_phaseequil, "BetaSpecification").def(py::init<double, std::size_t>());
        py::class_<MolarVolumeSpecification, AbstractSpecification, std::shared_ptr<MolarVolumeSpecification>>(m_phaseequil, "MolarVolumeSpecification").def(py::init<double>());
        py::class_<MolarEntropySpecification, AbstractSpecification, std::shared_ptr<MolarEntropySpecification>>(m_phaseequil, "MolarEntropySpecification").def(py::init<double>());
        py::class_<MolarEnthalpySpecification, AbstractSpecification, std::shared_ptr<MolarEnthalpySpecification>>(m_phaseequil, "MolarEnthalpySpecification").def(py::init<double>());
        py::class_<MolarInternalEnergySpecification, AbstractSpecification, std::shared_ptr<MolarInternalEnergySpecification>>(m_phaseequil, "MolarInternalEnergySpecification").def(py::init<double>());
        
        using CallResult = GeneralizedPhaseEquilibrium::CallResult;
        py::class_<CallResult>(m_phaseequil, "CallResult")
            .def_readonly("r", &CallResult::r, "r")
            .def_readonly("J", &CallResult::J, "J")
        ;
        
        using UnpackedVariables = GeneralizedPhaseEquilibrium::UnpackedVariables;
        py::class_<UnpackedVariables>(m_phaseequil, "UnpackedVariables")
            .def(py::init<const double, const std::vector<Eigen::ArrayXd>&, const Eigen::ArrayXd&>())
            .def_readonly("T", &UnpackedVariables::T, "Temperature")
            .def_readonly("rhovecs", &UnpackedVariables::rhovecs, "Vectors of molar concentrations for each phase")
            .def_readonly("betas", &UnpackedVariables::betas, "Vector of molar phase fractions for each phase")
            .def("pack", &UnpackedVariables::pack, "Convenience function to generate the array of independent variables")
        ;
        
        py::class_<GeneralizedPhaseEquilibrium>(m_phaseequil, "GeneralizedPhaseEquilibrium")
            .def(py::init<const AbstractModel&, const Eigen::ArrayXd&, const UnpackedVariables&, const std::vector<std::shared_ptr<AbstractSpecification>>&>())
            .def("call", &GeneralizedPhaseEquilibrium::call, "Call the function to build the residuals and Jacobian matrix", "x"_a)
            .def("num_Jacobian", &GeneralizedPhaseEquilibrium::num_Jacobian, "A testing function to build the Jacobian with centered differences")
            .def_readonly("res", &GeneralizedPhaseEquilibrium::res, "The data structure containing r and J")
        ;
    }
    
    using namespace teqp::iteration;
    py::class_<NRIterator>(m, "NRIterator")
        .def(py::init<const AlphaModel&, const std::vector<char>&, const Eigen::ArrayXd&, double, double, const Eigen::Ref<const Eigen::ArrayXd>&, const std::tuple<bool, bool>&, const std::vector<std::shared_ptr<StoppingCondition>>>())
        .def("calc_step", &NRIterator::calc_step)
    //        .def("take_step", &NRIterator::take_step)
    //        .def("take_step_getmaxabsr", &NRIterator::take_step_getmaxabsr)
        .def("take_steps", &NRIterator::take_steps)
        .def("get_vars", &NRIterator::get_vars)
        .def("get_vals", &NRIterator::get_vals)
        .def("get_molefrac", &NRIterator::get_molefrac)
        .def("get_T", &NRIterator::get_T)
        .def("get_rho", &NRIterator::get_rho)
    ;
    auto add_paramoptimizermodule = [](auto & m)
    {
        
        
        py::class_<SatRhoLPoint>(m, "SatRhoLPoint")
            .def(py::init<>())
            .def_readwrite("T", &SatRhoLPoint::T)
            .def_readwrite("rhoL_exp", &SatRhoLPoint::rhoL_exp)
            .def_readwrite("rhoL_guess", &SatRhoLPoint::rhoL_guess)
            .def_readwrite("rhoV_guess", &SatRhoLPoint::rhoV_guess)
            .def_readwrite("weight", &SatRhoLPoint::weight)
        ;
        py::class_<SatRhoLPPoint>(m, "SatRhoLPPoint")
            .def(py::init<>())
            .def_readwrite("T", &SatRhoLPPoint::T)
            .def_readwrite("rhoL_exp", &SatRhoLPPoint::rhoL_exp)
            .def_readwrite("p_exp", &SatRhoLPPoint::p_exp)
            .def_readwrite("rhoL_guess", &SatRhoLPPoint::rhoL_guess)
            .def_readwrite("rhoV_guess", &SatRhoLPPoint::rhoV_guess)
            .def_readwrite("weight_rho", &SatRhoLPPoint::weight_rho)
            .def_readwrite("weight_p", &SatRhoLPPoint::weight_p)
            .def_readwrite("R", &SatRhoLPPoint::R)
        ;
        py::class_<SatRhoLPWPoint>(m, "SatRhoLPWPoint")
            .def(py::init<>())
            .def_readwrite("T", &SatRhoLPWPoint::T)
            .def_readwrite("rhoL_exp", &SatRhoLPWPoint::rhoL_exp)
            .def_readwrite("p_exp", &SatRhoLPWPoint::p_exp)
            .def_readwrite("w_exp", &SatRhoLPWPoint::w_exp)
            .def_readwrite("R", &SatRhoLPWPoint::R)
            .def_readwrite("Ao20", &SatRhoLPWPoint::Ao20)
            .def_readwrite("M", &SatRhoLPWPoint::M)
            .def_readwrite("rhoL_guess", &SatRhoLPWPoint::rhoL_guess)
            .def_readwrite("rhoV_guess", &SatRhoLPWPoint::rhoV_guess)
            .def_readwrite("weight_rho", &SatRhoLPWPoint::weight_rho)
            .def_readwrite("weight_p", &SatRhoLPWPoint::weight_p)
            .def_readwrite("weight_w", &SatRhoLPWPoint::weight_w)
        ;
        py::class_<PVTNoniterativePoint>(m, "PVTNoniterativePoint")
            .def(py::init<>())
            .def_readwrite("T", &PVTNoniterativePoint::T)
            .def_readwrite("rho_exp", &PVTNoniterativePoint::rho_exp)
            .def_readwrite("p_exp", &PVTNoniterativePoint::p_exp)
            .def_readwrite("weight", &PVTNoniterativePoint::weight)
            .def_readwrite("R", &PVTNoniterativePoint::R)
        ;
        py::class_<SOSPoint>(m, "SOSPoint")
            .def(py::init<>())
            .def_readwrite("weight_w", &SOSPoint::weight_w)
    #define X(field) .def_readwrite(stringify(field), &SOSPoint::field)
        SOSPoint_fields
    #undef X
        ;
        
        py::class_<PureParameterOptimizer>(m, "PureParameterOptimizer")
            .def(py::init<const nlohmann::json&, const std::vector<std::variant<std::string, std::vector<std::string>>>&>())
            .def_readonly("contributions", &PureParameterOptimizer::contributions, py::return_value_policy::copy)
            .def("cost_function", &PureParameterOptimizer::cost_function<Eigen::ArrayXd>)
            .def("cost_function_threaded", &PureParameterOptimizer::cost_function_threaded<Eigen::ArrayXd>)
            .def("build_JSON", &PureParameterOptimizer::build_JSON<Eigen::ArrayXd>)
            .def("add_one_contribution", &PureParameterOptimizer::add_one_contribution)
        ;
    };
    auto m_paramopt = m.def_submodule("paramopt", "Tools for doing parameter optimization");
    add_paramoptimizermodule(m_paramopt);
    
//    // Some functions for timing overhead of interface
//    m.def("___mysummer", [](const double &c, const Eigen::ArrayXd &x) { return c*x.sum(); });
//    using RAX = Eigen::Ref<const Eigen::ArrayXd>;
//    using namespace pybind11::literals; // for "arg"_a
//    m.def("___mysummerref", [](const double& c, const RAX x) { return c * x.sum(); }, "c"_a, "x"_a.noconvert());
//    m.def("___myadder", [](const double& c, const double& d) { return c+d; });
}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    m.attr("__version__") = TEQPVERSION;
    init_teqp(m);
}
