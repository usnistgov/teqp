#include "pybind11_wrapper.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"

void add_multifluid(py::module& m) {

    // Multifluid model
    m.def("build_multifluid_model", &build_multifluid_model, py::arg("components"), py::arg("coolprop_root"), py::arg("BIPcollectionpath") = "", py::arg("flags") = nlohmann::json{}, py::arg("departurepath") = "");
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"",""},"",""));
    auto wMF = py::class_<MultiFluid>(m, "MultiFluid")
        .def("get_Tcvec", [](const MultiFluid& c) { return c.redfunc.Tc; })
        .def("get_vcvec", [](const MultiFluid& c) { return c.redfunc.vc; })
        ;
    add_derivatives<MultiFluid>(m, wMF);
}