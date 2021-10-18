/***
* A script for testing the loss in precision of autodiff differentiation and comparing to the lost
* precision in REFPROP
*/

// Only this file gets the implementation
#define REFPROP_IMPLEMENTATION
#define REFPROP_FUNCTION_MODIFIER
#include "REFPROP_lib.h"
#undef REFPROP_FUNCTION_MODIFIER
#undef REFPROP_IMPLEMENTATION

#include <iostream>
#include <valarray>

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;
#include "teqp/finite_derivs.hpp"

/// A standalone implementation to be more in control of type promotion.  
/// In the end this standalone implementation gives the same answer
template<typename T, typename Tau, typename Delta>
auto alphar_Lemmon2009(Tau tau, Delta delta)
{
    const static std::valarray<T> d = { 4.0,1.0,1.0,2.0,2.0,1.0,3.0,6.0,6.0,2.0,3.0,1.0,1.0,1.0,2.0,2.0,4.0,1.0 },
        n = { 0.042910051,1.7313671,-2.4516524,0.34157466,-0.46047898,-0.66847295,0.20889705,0.19421381,-0.22917851,-0.60405866,0.066680654,0.017534618,0.33874242,0.22228777,-0.23219062,-0.09220694,-0.47575718,-0.017486824 },
        t = { 1,0.33,0.8,0.43,0.9,2.46,2.09,0.88,1.09,3.25,4.62,0.76,2.5,2.75,3.05,2.55,8.4,6.75 },
        ld = { 0,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0 },
        cd = { 0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0 },
        lt = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        ct = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        beta = { 0,0,0,0,0,0,0,0,0,0,0,2.33,3.47,3.15,3.19,0.92,18.8,547.8 },
        epsilon = { 0,0,0,0,0,0,0,0,0,0,0,1.283,0.6936,0.788,0.473,0.8577,0.271,0.948 },
        eta = { 0,0,0,0,0,0,0,0,0,0,0,0.963,1.977,1.917,2.307,2.546,3.28,14.6 },
        gamma = { 0,0,0,0,0,0,0,0,0,0,0,0.684,0.829,1.419,0.817,1.5,1.426,1.093 };
    std::common_type_t<Tau, Delta> result = 0.0;
    for (auto i = 0; i < n.size(); ++i) {
        result += n[i] * pow(tau, t[i]) * pow(delta, d[i]) * exp(-cd[i] * pow(delta, ld[i]) - eta[i] * pow(delta - epsilon[i], 2) - beta[i] * pow(tau - gamma[i], 2));
    }
    return result;
}

int REFPROP_setup() {
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = std::getenv("RPPREFIX");
    std::string DLL_name = "";

    // Load the shared library and set up the fluid
    std::string err;
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop: %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP) { throw std::invalid_argument("Bad load of REFPROP"); }
    char hpath[256]; strcpy(hpath, path + std::string(254-path.size(),'\0'));
    SETPATHdll(hpath, 255);
    int ierr = 0, nc = 1;
    char herr[256], hfld[10000] = "PROPANE           ", hhmx[255] = "HMX.BNC", href[4] = "DEF";
    SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
    if (ierr != 0) { throw std::invalid_argument("Bad setup of REFPROP: "+std::string(herr)); }
}

struct REFPROP_sat_output {
    double T, rhoLmol_L, rhoVmol_L, p_kPa, rho_mol_L;
    char herr[256];
    int ierr;
    std::valarray<double> mole_fractions{ 0.0, 20 }, mole_fractions_liq{ 0.0, 20 }, mole_fractions_vap{ 0.0, 20 };
};

// Do a saturation call in REFPROP to generate the liquid and vapor densities for a given temperature
auto REFPROP_sat(double T) {
    REFPROP_sat_output o;
    o.T = T;
    int iFlsh = 0;
    SATTdll(T, &(o.mole_fractions[0]), iFlsh,
        o.p_kPa, o.rhoLmol_L, o.rhoVmol_L, &(o.mole_fractions_liq[0]), &(o.mole_fractions_vap[0]),
        o.ierr, o.herr, errormessagelength);
    return o;
}

struct calc_output {
    double Zexact, Zteqp;
};

template<typename Model, typename VECTOR>
auto with_teqp_and_boost(const Model &model, double T, double rhoL, const VECTOR &z){
    // Pressure for each phase via teqp in double precision w/ autodiff
    using tdx = TDXDerivatives<decltype(model), double, std::valarray<double>>;
    double Zteqp = 1.0 + tdx::get_Ar01(model, T, rhoL, z);

    // Calculation with ridiculous number of digits of precision (the approximation of ground truth)
    using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
    my_float Tc = model.redfunc.Tc[0];
    my_float rhoc = 5000.0;
    auto delta = static_cast<my_float>(rhoL) / rhoc;
    auto tau = Tc / static_cast<my_float>(T);
    my_float ddelta = 1e-30 * delta;
    my_float deltaplus = delta + ddelta, deltaminus = delta - ddelta;
    using coef_type = my_float; // What numerical type to use to initialize the coefficients (actually it doesn't matter since they all get upcasted to my_float)

    // Check that the function values are exactly the same
    auto ar1 = model.corr.get_EOS(0).alphar(tau, delta);
    auto ar2 = alphar_Lemmon2009<my_float>(tau, delta);
    auto dar2 = static_cast<double>((ar2 - ar1) / ar1);
    if (std::abs(dar2) > 1e-100) { // yes, we have ridiculously accurate values
        throw std::invalid_argument("Function values are not exactly the same");
    }

    // And now the derivative value in two subtly different approaches, also checl that 2nd-order-truncation and 4th-order-truncation are the same
    auto derL2_2nd = (alphar_Lemmon2009<coef_type>(tau, deltaplus) - alphar_Lemmon2009<coef_type>(tau, deltaminus)) / (2.0 * ddelta) * delta;
    auto derL2_4th = (
         1.0*alphar_Lemmon2009<coef_type>(tau, delta - 2.0*ddelta)/12.0 
        -2.0*alphar_Lemmon2009<coef_type>(tau, delta - ddelta)/3.0 
        +2.0*alphar_Lemmon2009<coef_type>(tau, delta + ddelta)/3.0 
        -1.0*alphar_Lemmon2009<coef_type>(tau, delta + 2.0*ddelta)/12.0
        ) / ddelta * delta;
    auto derL3 = (model.corr.get_EOS(0).alphar(tau, deltaplus) - model.corr.get_EOS(0).alphar(tau, deltaminus)) / (2.0 * ddelta) * delta;
    auto Zexact = derL2_4th + 1.0;

    auto d3 = static_cast<double>((derL2_2nd - derL3) / derL2_2nd);
    auto d34th = static_cast<double>((derL2_4th - derL2_2nd) / derL2_2nd);

    if (std::abs(d3) > 1e-100) { // yes, we have ridiculously accurate values
        throw std::invalid_argument("Derivatives are not exactly the same in teqp and in standalone implementation");
    }

    calc_output o;
    o.Zexact = static_cast<double>(Zexact);
    o.Zteqp = Zteqp;
    return o;
}

int main()
{
    REFPROP_setup();
    auto model = build_multifluid_model({"n-Propane"}, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");
    std::valarray<double> z = { 1.0 };
    double Tt = 85.525, Tc = 369.89;
    int NT = 200;
    nlohmann::json outputs = nlohmann::json::array();
    for (double T : Eigen::ArrayXd::LinSpaced(NT, Tt, Tc)) {
        auto o = REFPROP_sat(T);

        // Pressure for each phase via REFPROP
        double pL = -1; PRESSdll(o.T, o.rhoLmol_L, &(z[0]), pL);
        double RL = -1; RMIX2dll(&(z[0]), RL);
        double ZLRP = pL/(o.rhoLmol_L*RL*o.T); // Units cancel (factor of 1000 in pL and RL)
        double pV = -1; PRESSdll(o.T, o.rhoVmol_L, &(z[0]), pV);
        double RV = -1; RMIX2dll(&(z[0]), RV);
        double ZVRP = pV/(o.rhoVmol_L*RV*o.T); // Units cancel (factor of 1000 in pV and RV)

        double rhoL = o.rhoLmol_L * 1000.0, rhoV = o.rhoVmol_L*1000.0;

        auto c = with_teqp_and_boost(model, T, rhoL, z);
        outputs.push_back({
            {"T / K", T},
            {"Q", 0},
            {"rho / mol/m^3", rhoL},
            {"Zteqp", c.Zteqp},
            {"Zexact", c.Zexact},
            {"ratio-1", c.Zteqp/c.Zexact-1},
            {"ZRP", ZLRP},
        });

        auto cV = with_teqp_and_boost(model, T, rhoV, z);
        outputs.push_back({
            {"T / K", T},
            {"Q", 1},
            {"rho / mol/m^3", rhoV},
            {"Zteqp", cV.Zteqp},
            {"Zexact", cV.Zexact},
            {"ratio-1", cV.Zteqp/cV.Zexact-1},
            {"ZRP", ZVRP},
            });
        std::cout << "Completed:" << T << std::endl;
    }
    std::ofstream file("saturation_Z_accuracy.json");
    file << outputs;
    return EXIT_SUCCESS;
}