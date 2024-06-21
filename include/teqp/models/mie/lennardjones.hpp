#pragma once

#include "teqp/models/multifluid.hpp"
#include "teqp/types.hpp"
#include "teqp/math/pow_templates.hpp"
#include "teqp/models/mie/lennardjones/johnson.hpp"

namespace teqp {

    /**
     * The EOS of Monika Thol and colleagues. DOI:10.1063/1.4945000
     */
    inline auto build_LJ126_TholJPCRD2016() {
        std::string contents = R"(

        {
          "EOS": [
            {
              "BibTeX_CP0": "",
              "BibTeX_EOS": "Thol-THESIS-2015",
              "STATES": {
                "reducing": {
                  "T": 1.32,
                  "T_units": "LJ units",
                  "rhomolar": 0.31,
                  "rhomolar_units": "LJ units"
                }
              },
              "T_max": 1200,
              "T_max_units": "LJ units",
              "Ttriple": 0.661,
              "Ttriple_units": "LJ units",
              "alphar": [
                {
                  "d": [4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5],
                  "l": [0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1],
                  "n": [0.52080730e-2, 0.21862520e+1, -0.21610160e+1, 0.14527000e+1, -0.20417920e+1, 0.18695286e+0, -0.62086250e+0, -0.56883900e+0, -0.80055922e+0, 0.10901431e+0, -0.49745610e+0, -0.90988445e-1],
                  "t": [1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294],
                  "type": "ResidualHelmholtzPower"
                },
                {
                  "beta": [0.625, 0.638, 3.91, 0.156, 0.157, 0.153, 1.16, 1.73, 383, 0.112, 0.119],
                  "d": [1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1],
                  "epsilon": [ 0.2053, 0.409, 0.6, 1.203, 1.829, 1.397, 1.39, 0.539, 0.934, 2.369, 2.43],
                  "eta": [2.067, 1.522, 8.82, 1.722, 0.679, 1.883, 3.925, 2.461, 28.2, 0.753, 0.82],
                  "gamma": [0.71, 0.86, 1.94, 1.48, 1.49, 1.945, 3.02, 1.11, 1.17, 1.33, 0.24],
                  "n": [-0.14667177e+1, 0.18914690e+1, -0.13837010e+0, -0.38696450e+0, 0.12657020e+0, 0.60578100e+0, 0.11791890e+1, -0.47732679e+0, -0.99218575e+1, -0.57479320e+0, 0.37729230e-2],
                  "t": [2.830, 2.548, 4.650, 1.385, 1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970],
                  "type": "ResidualHelmholtzGaussian"
                }
              ],
              "gas_constant": 1.0,
              "gas_constant_units": "LJ units",
              "molar_mass": 1.0,
              "molar_mass_units": "LJ units",
              "p_max": 100000,
              "p_max_units": "LJ units",
              "pseudo_pure": false
            }
          ],
          "INFO":{
            "NAME": "LennardJones",
            "REFPROP_NAME": "LJF",
            "CAS": "N/A"
            }
        }

        )";

        return teqp::build_multifluid_JSONstr({ contents }, "{}", "{}");
    };
};

using namespace teqp::mie::lennardjones::Johnson;
using namespace teqp::mie::lennardjones::Kolafa;
