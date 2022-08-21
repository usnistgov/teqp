#include "pybind11_wrapper.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/models/ammonia_water.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/derivs.hpp"
#include "teqp/models/mie/lennardjones.hpp"

#include "multifluid_shared.hpp"

void add_AmmoniaWaterTillnerRoth(py::module&m ){
    auto wAW = py::class_<AmmoniaWaterTillnerRoth>(m, "AmmoniaWaterTillnerRoth")
        .def(py::init<>())
        .def_readonly("TcNH3", &AmmoniaWaterTillnerRoth::TcNH3)
        .def_readonly("vcNH3", &AmmoniaWaterTillnerRoth::vcNH3)
        .def("get_Tr", &AmmoniaWaterTillnerRoth::get_Treducing<Eigen::ArrayXd>)
        .def("get_rhor", &AmmoniaWaterTillnerRoth::get_rhoreducing<Eigen::ArrayXd>)
        .def("alphar_departure", &AmmoniaWaterTillnerRoth::alphar_departure<double, double, double>, py::arg("tau"), py::arg("delta"), py::arg("xNH3"))
    ;
    add_derivatives<AmmoniaWaterTillnerRoth>(m, wAW);
}

void add_multifluid(py::module& m) {

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

    // Multifluid model
    m.def("build_multifluid_model", &build_multifluid_model, py::arg("components"), py::arg("coolprop_root"), py::arg("BIPcollectionpath") = "", py::arg("flags") = nlohmann::json{}, py::arg("departurepath") = "");
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"",""},"",""));
    auto wMF = py::class_<MultiFluid>(m, "MultiFluid");
    add_derivatives<MultiFluid>(m, wMF);
    add_multifluid_methods<MultiFluid>(wMF);
    wMF.def("get_alpharij", [](const MultiFluid& c, const int i, const int j, const double &tau, const double &delta) { return c.dep.get_alpharij(i, j, tau, delta); });

    // Expose some additional functions for working with the JSON data structures and resolving aliases
    m.def("get_BIPdep", &reducing::get_BIPdep, py::arg("BIPcollection"), py::arg("identifiers"), py::arg("flags") = nlohmann::json{});
    m.def("build_alias_map", &build_alias_map, py::arg("root"));
    m.def("collect_component_json", &collect_component_json, py::arg("identifiers"), py::arg("root"));
    m.def("get_departure_json", &get_departure_json, py::arg("name"), py::arg("root"));

    m.def("build_LJ126_TholJPCRD2016", &teqp::build_LJ126_TholJPCRD2016);
    add_AmmoniaWaterTillnerRoth(m);
}