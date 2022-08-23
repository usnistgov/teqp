
#include "nlohmann/json.hpp"
#include "pybind11_json/pybind11_json.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "teqp/core.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/algorithms/VLE.hpp"
#include "teqp/algorithms/VLLE.hpp"

namespace py = pybind11;
using namespace teqp;

template<typename Model, int iT, int iD, typename Class>
void add_res_deriv_impl(Class& cls) {
    using RAX = Eigen::Ref<Eigen::ArrayXd>;
    using idx = TDXDerivatives<Model, double, RAX>;
    const std::string fname = "get_Ar" + std::to_string(iT) + std::to_string(iD);
    cls.def(fname.c_str(),
        [](const Model& m, const double T, const double rho, const RAX molefrac) { return idx::template get_Arxy<iT, iD, ADBackends::autodiff>(m, T, rho, molefrac); },
        py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert()
    );
}

template<typename Model, typename Class>
void add_res_derivatives(Class& cls) {
    add_res_deriv_impl<Model, 0, 0>(cls); add_res_deriv_impl<Model, 0, 1>(cls); add_res_deriv_impl<Model, 0, 2>(cls); add_res_deriv_impl<Model, 0, 3>(cls); add_res_deriv_impl<Model, 0, 4>(cls);
    add_res_deriv_impl<Model, 1, 0>(cls); add_res_deriv_impl<Model, 1, 1>(cls); add_res_deriv_impl<Model, 1, 2>(cls); add_res_deriv_impl<Model, 1, 3>(cls); add_res_deriv_impl<Model, 1, 4>(cls);
    add_res_deriv_impl<Model, 2, 0>(cls); add_res_deriv_impl<Model, 2, 1>(cls); add_res_deriv_impl<Model, 2, 2>(cls); add_res_deriv_impl<Model, 2, 3>(cls); add_res_deriv_impl<Model, 2, 4>(cls);
}

// Note: Do NOT(!) attach methods to the module, rather attach them to the instance
// Overloaded functions in pybind11 struggle to resolve in a timely fashion, and 
// looking up a method on an instance is much(!) faster

template<typename Model, typename Wrapper>
void add_derivatives(py::module &m, Wrapper &cls) {

    using RAX = Eigen::Ref<Eigen::ArrayXd>;

    using id = IsochoricDerivatives<Model, double, RAX >;
    cls.def("get_Ar00iso", &id::get_Ar00, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_Ar10iso", &id::get_Ar10, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_Psiriso", &id::get_Psir, py::arg("T"), py::arg("rho").noconvert());

    cls.def("get_pr", &id::get_pr, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_splus", &id::get_splus, py::arg("T"), py::arg("rho").noconvert());

    cls.def("build_Psir_Hessian_autodiff", &id::build_Psir_Hessian_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("build_Psi_Hessian_autodiff", &id::build_Psi_Hessian_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("build_Psir_gradient_autodiff", &id::build_Psir_gradient_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("build_d2PsirdTdrhoi_autodiff", &id::build_d2PsirdTdrhoi_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_chempotVLE_autodiff", &id::get_chempotVLE_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_dchempotdT_autodiff", &id::get_dchempotdT_autodiff, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_fugacity_coefficients", &id::template get_fugacity_coefficients<ADBackends::autodiff>, py::arg("T"), py::arg("rho").noconvert());
    cls.def("get_partial_molar_volumes", &id::get_partial_molar_volumes, py::arg("T"), py::arg("rhovec").noconvert());

    using vd = VirialDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1>>;
    cls.def("get_B2vir", &vd::get_B2vir, py::arg("T"), py::arg("molefrac").noconvert());
    cls.def("get_Bnvir", [](const Model& m, const int Nderiv, const double T, const RAX molefrac) { return vd::get_Bnvir_runtime(Nderiv, m, T, molefrac); }, py::arg("Nderiv"), py::arg("T"), py::arg("molefrac").noconvert());
    cls.def("get_dmBnvirdTm", [](const Model& m, const int Nderiv, const int NTderiv, const double T, const RAX molefrac) { return vd::get_dmBnvirdTm_runtime(Nderiv, NTderiv, m, T, molefrac); }, py::arg("Nderiv"), py::arg("NTderiv"), py::arg("T"), py::arg("molefrac").noconvert());
    cls.def("get_B12vir", &vd::get_B12vir, py::arg("T"), py::arg("molefrac").noconvert());

    using ct = CriticalTracing<Model, double, Eigen::Array<double, Eigen::Dynamic, 1>>;
    cls.def("trace_critical_arclength_binary", &ct::trace_critical_arclength_binary, py::arg("T0"), py::arg("rhovec0").noconvert(), py::arg_v("path", std::nullopt, "None"), py::arg_v("options", std::nullopt, "None"));
    cls.def("get_criticality_conditions", &ct::get_criticality_conditions);
    cls.def("eigen_problem", &ct::eigen_problem);
    cls.def("get_minimum_eigenvalue_Psi_Hessian", &ct::get_minimum_eigenvalue_Psi_Hessian);
    cls.def("get_drhovec_dT_crit", &ct::get_drhovec_dT_crit);

    cls.def("extrapolate_from_critical", &extrapolate_from_critical<Model, double>);
    cls.def("pure_VLE_T", &pure_VLE_T<Model, double>);
    
    cls.def("get_pure_critical_conditions_Jacobian", &get_pure_critical_conditions_Jacobian<Model, double, ADBackends::autodiff>, py::arg("T"), py::arg("rho"), py::arg_v("alternative_pure_index", -1), py::arg_v("alternative_length", 2));
    cls.def("solve_pure_critical", &solve_pure_critical<Model, double, ADBackends::autodiff>, py::arg("T"), py::arg("rho"), py::arg_v("flags", std::nullopt, "None"));
    cls.def("mix_VLE_Tx", &mix_VLE_Tx<Model, double, RAX>);
    cls.def("mix_VLE_Tp", &mix_VLE_Tp<Model, double, RAX>, py::arg("T"), py::arg("p_given"), py::arg("rhovecL0").noconvert(), py::arg("rhovecV0").noconvert(), py::arg_v("flags", MixVLETpFlags{}, "None"));
    cls.def("mixture_VLE_px", &mixture_VLE_px<Model, double, Eigen::ArrayXd>, py::arg("p_spec"), py::arg("xmolar_spec").noconvert(), py::arg("T0"), py::arg("rhovecL0").noconvert(), py::arg("rhovecV0").noconvert(), py::arg_v("flags", MixVLEpxFlags{}, "None"));

    cls.def("get_drhovecdp_Tsat", &get_drhovecdp_Tsat<Model, double, RAX>, py::arg("T"), py::arg("rhovecL").noconvert(), py::arg("rhovecV").noconvert());
    cls.def("trace_VLE_isotherm_binary", &trace_VLE_isotherm_binary<Model, double, Eigen::ArrayXd>, py::arg("T"), py::arg("rhovecL0").noconvert(), py::arg("rhovecV0").noconvert(), py::arg_v("options", std::nullopt, "None"));
    cls.def("get_drhovecdT_psat", &get_drhovecdT_psat<Model, double, RAX>, py::arg("T"), py::arg("rhovecL").noconvert(), py::arg("rhovecV").noconvert());
    cls.def("trace_VLE_isobar_binary", &trace_VLE_isobar_binary<Model, double, Eigen::ArrayXd>, py::arg("p"), py::arg("T0"), py::arg("rhovecL0").noconvert(), py::arg("rhovecV0").noconvert(), py::arg_v("options", std::nullopt, "None"));
    cls.def("get_dpsat_dTsat_isopleth", &get_dpsat_dTsat_isopleth<Model, double, Eigen::ArrayXd>, py::arg("T"), py::arg("rhovecL").noconvert(), py::arg("rhovecV").noconvert());

    cls.def("mix_VLLE_T", &mix_VLLE_T<Model, double, Eigen::ArrayXd>);
    cls.def("find_VLLE_T_binary", &find_VLLE_T_binary<Model>, py::arg("traces"), py::arg_v("options", std::nullopt, "None"));

    // Temperature, density, composition derivatives
    using tdx = TDXDerivatives<Model, double, RAX>;
    
    cls.def("get_R", [](const Model& m, const RAX molefrac) { return m.R(molefrac); }, py::arg("molefrac").noconvert());
    cls.def("get_Ar00", &tdx::get_Ar00, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    add_res_derivatives<Model>(cls); // All the residual derivatives

    cls.def("get_Ar01n", &(tdx::template get_Ar0n<1>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar02n", &(tdx::template get_Ar0n<2>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar03n", &(tdx::template get_Ar0n<3>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar04n", &(tdx::template get_Ar0n<4>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar05n", &(tdx::template get_Ar0n<5>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar06n", &(tdx::template get_Ar0n<6>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_neff", &(tdx::template get_neff<ADBackends::autodiff>), py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());

}