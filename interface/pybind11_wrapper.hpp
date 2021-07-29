
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

namespace py = pybind11;

template<typename Model, typename Wrapper>
void add_derivatives(py::module &m, Wrapper &cls) {

    using RAX = Eigen::Ref<Eigen::ArrayXd>;

    using id = IsochoricDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1> >;
    m.def("get_Ar00", &id::get_Ar00, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Ar10", &id::get_Ar10, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Psir", &id::get_Psir, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("get_pr", &id::get_pr, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_splus", &id::get_splus, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("build_Psir_Hessian_autodiff", &id::build_Psir_Hessian_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_Psi_Hessian_autodiff", &id::build_Psi_Hessian_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_Psir_gradient_autodiff", &id::build_Psir_gradient_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_d2PsirdTdrhoi_autodiff", &id::build_d2PsirdTdrhoi_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_chempotVLE_autodiff", &id::get_chempotVLE_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_dchempotdT_autodiff", &id::get_dchempotdT_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));

    using vd = VirialDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1>>;
    m.def("get_B2vir", &vd::get_B2vir, py::arg("model"), py::arg("T"), py::arg("molefrac").noconvert());
    cls.def("get_B2vir", [](const Model& m, const double T, const RAX molefrac) { return vd::get_B2vir(m, T, molefrac); }, py::arg("T"), py::arg("molefrac").noconvert());
    m.def("get_B12vir", &vd::get_B12vir, py::arg("model"), py::arg("T"), py::arg("molefrac").noconvert());

    using ct = CriticalTracing<Model, double, Eigen::Array<double, Eigen::Dynamic, 1>>;
    m.def("trace_critical_arclength_binary", &ct::trace_critical_arclength_binary);
    m.def("get_drhovec_dT_crit", &ct::get_drhovec_dT_crit);

    m.def("extrapolate_from_critical", &extrapolate_from_critical<Model, double>);
    m.def("pure_VLE_T", &pure_VLE_T<Model, double>);
    m.def("get_drhovecdp_Tsat", &get_drhovecdp_Tsat<Model, double, RAX>, py::arg("model"), py::arg("T"), py::arg("rhovecL").noconvert(), py::arg("rhovecV").noconvert());

    //cls.def("get_Ar01", [](const Model& m, const double T, const Eigen::ArrayXd& rhovec) { return id::get_Ar01(m, T, rhovec); });
    //cls.def("get_Ar10", [](const Model& m, const double T, const Eigen::ArrayXd& rhovec) { return id::get_Ar10(m, T, rhovec); });
    using tdx = TDXDerivatives<Model, double, Eigen::Array<double, Eigen::Dynamic, 1> >;
    
    cls.def("get_R", [](const Model& m, const RAX molefrac) { return m.R(molefrac); }, py::arg("molefrac").noconvert());
    cls.def("get_Ar00", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::get_Ar00(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar01", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar01<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar10", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar10<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar11", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar11<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar12", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar12<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());

    cls.def("get_Ar01n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<1>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar02n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<2>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar03n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<3>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar04n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<4>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar05n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<5>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_Ar06n", [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::template get_Ar0n<6>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
    cls.def("get_neff",  [](const Model& m, const double T, const double rho, const RAX molefrac) { return tdx::get_neff(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert());
}