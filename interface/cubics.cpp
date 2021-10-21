#include "pybind11_wrapper.hpp"

#include "teqp/models/cubics.hpp"

void add_cubics(py::module& m) {

    using va = std::valarray<double>;
    
    m.def("canonical_PR", &canonical_PR<va,va,va>, py::arg("Tc_K"), py::arg("pc_Pa"), py::arg("acentric"));
    m.def("canonical_SRK", &canonical_SRK<va, va, va>, py::arg("Tc_K"), py::arg("pc_Pa"), py::arg("acentric"));

    using cub = decltype(canonical_PR(va{}, va{}, va{}));
    auto wcub = py::class_<cub>(m, "GenericCubic")
        .def("get_meta", &cub::get_meta)
        .def("superanc_rhoLV", &cub::superanc_rhoLV)
        ;
    add_derivatives<cub>(m, wcub);
}