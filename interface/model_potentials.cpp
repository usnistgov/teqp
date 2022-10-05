//
//  model_potentials.cpp
//  teqp
//
//  Created by Bell, Ian H. (Fed) on 10/5/22.
//

#include "pybind11_wrapper.hpp"

#include "teqp/models/squarewell.hpp"

using namespace teqp;

void add_model_potentials(py::module& m) {
    using namespace teqp::squarewell;
    
    auto cls = py::class_<EspindolaHeredia2009>(m, "SW_EspindolaHeredia2009")
        .def(py::init<double>(), py::arg("lambda_"))
        ;
    add_derivatives<EspindolaHeredia2009>(m, cls);
}
