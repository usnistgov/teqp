
#include "pybind11_wrapper.hpp"

#include "teqp/models/vdW.hpp"
#include "teqp/derivs.hpp"

void add_vdW(py::module &m){
	using vdWEOSd = vdWEOS<double>;
    auto wvdW = py::class_<vdWEOSd>(m, "vdWEOS")
        .def(py::init<const std::valarray<double>&, const std::valarray<double>&>(),py::arg("Tcrit"), py::arg("pcrit"))
        ;
    add_derivatives<vdWEOSd>(m, wvdW);

    auto wvdW1 = py::class_<vdWEOS1>(m, "vdWEOS1")
        .def(py::init<const double&, const double&>(), py::arg("a"), py::arg("b"))
        ;
    add_derivatives<vdWEOS1>(m, wvdW1);
}