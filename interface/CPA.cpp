#include "pybind11_wrapper.hpp"

#include "teqp/models/CPA.hpp"
#include "teqp/derivs.hpp"

void add_CPA(py::module& m) {

    // CPA model
    using CPAEOS_ = decltype(CPA::CPAfactory(nlohmann::json()));
    m.def("CPAfactory", &CPA::CPAfactory);
    auto wCPA = py::class_<CPAEOS_>(m, "CPAEOS");
    add_derivatives<CPAEOS_>(m, wCPA);
}