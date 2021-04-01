#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "teqp/core.hpp"

namespace py = pybind11;

template<typename Model>
void add_derivatives(py::module &m) {
    using id = IsochoricDerivatives<Model>;
    m.def("get_Ar00", &id::get_Ar00, py::arg("model"), py::arg("T"), py::arg("rho")); 
    m.def("get_Ar10", &id::get_Ar10, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Psir", &id::get_Psir, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("get_pr", &id::get_pr, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_splus", &id::get_splus, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("build_Psir_Hessian_autodiff", &id::build_Psir_Hessian_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_Psir_gradient_autodiff", &id::build_Psir_gradient_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
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

}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    init_teqp(m);
}