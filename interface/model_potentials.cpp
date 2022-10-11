//
//  model_potentials.cpp
//  teqp
//
//  Created by Bell, Ian H. (Fed) on 10/5/22.
//

#include "pybind11_wrapper.hpp"

#include "teqp/models/model_potentials/squarewell.hpp"
#include "teqp/models/model_potentials/exp6.hpp"

using namespace teqp;

void add_model_potentials(py::module& m) {
    using namespace teqp::squarewell;
    {
        auto cls = py::class_<EspindolaHeredia2009>(m, "SW_EspindolaHeredia2009")
            .def(py::init<double>(), py::arg("lambda_"))
        ;
        add_derivatives<EspindolaHeredia2009>(m, cls);
    }
    
    using namespace teqp::exp6;
    {
        auto cls = py::class_<Kataoka1992>(m, "EXP6_Kataoka1992")
            .def(py::init<double>(), py::arg("alpha"))
        ;
        add_derivatives<Kataoka1992>(m, cls);
    }
}
