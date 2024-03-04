/**
 * Demonstration of how to specify your own EOS at runtime. In this case, 
 * the EOS of Monika Thol and colleagues: https://doi.org/10.1063/1.4945000
 */

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"

#include <array>

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
      "Ttriple": 290.25,
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
          "n": [-0.14667177e+1, 0.18914690e+1, -0.13837010e+0, -0.38696450e+0, 0.12657020e+0, 0.60578100e+0, 0.11791890e+1, -0.47732679e+0, -0.99218575e-1, -0.57479320e+0, 0.37729230e-2],
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

int main() {
    std::vector <std::string> componentJSON = { contents };
    auto model = teqp::build_multifluid_JSONstr(componentJSON, "{}", "{}");

    // Test values from https://aip.scitation.org/doi/suppl/10.1063/1.4945000/suppl_file/additional_information_ljf.pdf
    std::vector<std::array<double, 7> > data = {
        {0.8, 0.005, 3.8430053e-3, -5.4597389e-2, 5.5672903e-2, 1.1324263e0, 2.7768170e-1},
        {0.8, 0.8, 1.5894013e-2, -5.7174120e0, 9.5995160e-1, 5.0522400e0, 1.1838093e0},
        {1.0, 0.02, 1.7886470e-2, -1.8772644e-1, 1.3016045e-1, 1.2290934e0, 1.8318141e0},
        {1.0, 0.71, 7.5247483e-2, -4.9564222e0, 6.8903536e-1, 4.1644650e0, 2.9792860e0},
        {2.0, 0.5, 1.0751638e0, -3.1525021e0, 3.1068090e-1, 3.5186329e0, 9.5274193e0},
        {5.0, 0.6, 6.9432008e0, -2.6956781e0, 3.1772707e-1, 6.8375197e0, 2.6122755e1},
        {7.0, 1.0, 4.1531352e1, -6.2393078e-1, 7.3348579e-1, 1.4201978e1, 4.8074394e-1},
    };
    std::valarray<double> z(1.0, 1);


    std::cout << "**************** With general JSON interface **************" << std::endl;
    std::cout << "All in L-J units:" << std::endl;
    {
        constexpr int errmsg_length = 300;
        char uuid[33] = "", errmsg[errmsg_length] = "";
        double val = -1, Ar01, Ar00;
        auto molefrac = (Eigen::ArrayXd(1) <<  1.0).finished();

        nlohmann::json jmodel = nlohmann::json::object();
        jmodel["components"] = nlohmann::json::array();
        jmodel["components"].push_back(nlohmann::json::parse(contents));
        jmodel["departure"] = nlohmann::json::array();
        jmodel["BIP"] = nlohmann::json::array();
        jmodel["flags"] = nlohmann::json::object();

        nlohmann::json j = {
            {"kind", "multifluid"},
            {"model", jmodel}
        };
        auto m = teqp::cppinterface::make_model(j);

        for (auto& el : data) {
            auto [T_, rho_, p, ur, cvr, w, a] = el;
            double T = T_, rho = rho_; // It is not possible to capture tuple-unpacked variables

            auto NT = 0, ND = 0;

            // Lambda function to extract the given derivative from the thing contained in the variant
            auto f = [&](const auto& model) {
                using tdx = teqp::TDXDerivatives<decltype(model), double, decltype(molefrac)>;
                return tdx::get_Ar(NT, ND, model, T, rho, molefrac);
            };

            // Now call the visitor function to get the value
            auto Ar00 = m->get_Ar00(T, rho, molefrac);
            auto Ar01 = m->get_Ar01(T, rho, molefrac);
            auto Ar10 = m->get_Ar10(T, rho, molefrac);
            auto Ar20 = m->get_Ar20(T, rho, molefrac);

            double pcalc = T * rho * (1 + Ar01);
            double urcalc = T * Ar10;
            double cvrcalc = -Ar20;

            std::cout << "@ (T,rho): " << T << "," << rho << std::endl;
            std::cout << "p: " << pcalc << ", " << p << std::endl;
            std::cout << "ur: " << urcalc << ", " << ur << std::endl;
            std::cout << "cvr: " << cvrcalc << ", " << cvr << std::endl;
        }
    }

    std::cout << "**************** With normal interface **************" << std::endl;
    using tdx = teqp::TDXDerivatives<decltype(model), double, decltype(z)>;
    std::cout << "All in L-J units:" << std::endl;
    for (auto &el : data) {
        auto [T, rho, p, ur, cvr, w, a] = el; // I
        double Ar01 = tdx::get_Ar01(model, T, rho, z);
        double Ar10 = tdx::get_Ar10(model, T, rho, z);
        double Ar20 = tdx::get_Ar20(model, T, rho, z);
        double pcalc = T*rho*(1 + Ar01);
        double urcalc = T*Ar10;
        double cvrcalc = -Ar20;

        std::cout << "@ (T,rho): " << T << "," << rho << std::endl;
        std::cout << "p: " << pcalc << ", " << p << std::endl;
        std::cout << "ur: " << urcalc << ", " << ur << std::endl;
        std::cout << "cvr: " << cvrcalc << ", " << cvr << std::endl;
    }
}
