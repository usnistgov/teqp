#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "teqp/core.hpp"
#include "teqp/models/pcsaft.hpp"

namespace py = pybind11;

template<typename Model>
void add_virials(py::module& m) {
    using vd = VirialDerivatives<Model>;
    m.def("get_B2vir", &vd::get_B2vir, py::arg("model"), py::arg("T"), py::arg("molefrac"));
    m.def("get_B12vir", &vd::get_B12vir, py::arg("model"), py::arg("T"), py::arg("molefrac"));
}

template<typename Model>
void add_derivatives(py::module &m) {
    using id = IsochoricDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1> >;
    m.def("get_Ar00", &id::get_Ar00, py::arg("model"), py::arg("T"), py::arg("rho")); 
    m.def("get_Ar10", &id::get_Ar10, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Psir", &id::get_Psir, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("get_pr", &id::get_pr, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_splus", &id::get_splus, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("build_Psir_Hessian_autodiff", &id::build_Psir_Hessian_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_Psir_gradient_autodiff", &id::build_Psir_gradient_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));

    add_virials<Model>(m);
}


void init_teqp(py::module& m) {

    using vdWEOSd = vdWEOS<double>;
    py::class_<vdWEOSd>(m, "vdWEOS")
        .def(py::init<const std::valarray<double>&, const std::valarray<double>&>(),py::arg("Tcrit"), py::arg("pcrit"))
        ;
    add_derivatives<vdWEOSd>(m);

    py::class_<vdWEOS1>(m, "vdWEOS1")
        .def(py::init<const double&, const double&>(), py::arg("a"), py::arg("b"))
        ;
    add_derivatives<vdWEOS1>(m);

    py::class_<PCSAFTMixture>(m, "PCSAFTEOS")
        .def(py::init<const std::vector<std::string> &>(), py::arg("names"))
        ;
    add_derivatives<PCSAFTMixture>(m);

    // for timing testing
    m.def("mysummer", [](const double &c, const Eigen::ArrayXd &x) { return c*x.sum(); });
    m.def("myadder", [](const double& c, const double& d) { return c+d; });

}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    init_teqp(m);
}