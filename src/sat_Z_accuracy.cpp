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

#if !defined(TEQP_MULTIPRECISION_ENABLED)
#error "TEQP_MULTIPRECISION_ENABLED must be turned on"
#endif

#include <iostream>
#include <valarray>

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
using namespace boost::multiprecision;
#include "teqp/finite_derivs.hpp"

using namespace teqp;

/// A standalone implementation to be more in control of type promotion.  
/// In the end this standalone implementation gives the same answer
/// This is for propane
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

int REFPROP_setup(const std::string &RPname) {
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = std::getenv("RPPREFIX");
    std::string DLL_name = "";

    // Load the shared library and set up the fluid
    std::string err;
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop: %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP) { throw std::invalid_argument("Bad load of REFPROP"); }
    char hpath[256]; strcpy(hpath, (path + std::string(254-path.size(),'\0')).c_str());
    SETPATHdll(hpath, 255);
    int ierr = 0, nc = 1;
    char herr[256], hfld[10000] = "                             ", hhmx[256] = "HMX.BNC", href[4] = "DEF";
    strcpy(hfld, RPname.c_str());
    SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
    if (ierr != 0) { throw std::invalid_argument("Bad setup of REFPROP: "+std::string(herr)); }
    return 0;
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
    double Zexact, Zteqp, Ar01exact, Ar01teqp, Ar02exact, Ar02teqp, Ar03exact, Ar03teqp;
};

template<typename Model, typename VECTOR>
auto with_teqp_and_boost(const Model &model, double T, double rho, const VECTOR &z, bool is_propane){
    // Pressure for each phase via teqp in double precision w/ autodiff
    using tdx = TDXDerivatives<decltype(model), double, VECTOR>;
    double Zteqp = 1.0 + tdx::get_Ar01(model, T, rho, z);
    double Ar01teqp = tdx::get_Ar01(model, T, rho, z);

    // Calculation with ridiculous number of digits of precision (the approximation of ground truth)
    using my_float = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
    my_float Tc = model.redfunc.Tc[0];
    my_float rhoc = 1.0/static_cast<my_float>(model.redfunc.vc[0]);
    auto delta = static_cast<my_float>(rho) / rhoc;
    auto tau = Tc / static_cast<my_float>(T);
    my_float ddelta = 1e-30 * delta;
    my_float deltaplus = delta + ddelta, deltaminus = delta - ddelta;
    using coef_type = my_float; // What numerical type to use to initialize the coefficients (in the end it doesn't matter since they all get upcasted to my_float)

    // Check that the function values are exactly the same
    auto ar1 = model.corr.get_EOS(0).alphar(tau, delta);
    if (is_propane) {
        // As the standalone (if we are using propane)
        auto ar2 = alphar_Lemmon2009<my_float>(tau, delta);
        auto dar2 = static_cast<double>((ar2 - ar1) / ar1);
        if (std::abs(dar2) > 1e-100) { // yes, we have ridiculously accurate values
            throw std::invalid_argument("Function values are not exactly the same");
        }
    }
    else {
        // Or from REFPROP otherwise
        int itau = 0, idelta = 0;
        double tau_ = static_cast<double>(tau), delta_ = static_cast<double>(delta);
        std::valarray<double> z(20); z = 1;
        double ar2 = -1; PHIXdll(itau, idelta, tau_, delta_, &(z[0]), ar2);
        double dar2 = static_cast<double>((ar2 - ar1) / ar1);
        if (std::abs(dar2) > 5e-14) { // basically double precision..
            std::cout << dar2 << std::endl;
            throw std::invalid_argument("Function values are not exactly the same; error (%): "+std::to_string(dar2));
        }
    }

    // And now the derivative value in two subtly different approaches, also check that 2nd-order-truncation and 4th-order-truncation are the same
    auto derL2_2nd = (alphar_Lemmon2009<coef_type>(tau, deltaplus) - alphar_Lemmon2009<coef_type>(tau, deltaminus)) / (2.0 * ddelta) * delta;
    auto derL2_4th = (
         1.0*alphar_Lemmon2009<coef_type>(tau, delta - 2.0*ddelta)/12.0 
        -2.0*alphar_Lemmon2009<coef_type>(tau, delta - ddelta)/3.0 
        +2.0*alphar_Lemmon2009<coef_type>(tau, delta + ddelta)/3.0 
        -1.0*alphar_Lemmon2009<coef_type>(tau, delta + 2.0*ddelta)/12.0
        ) / ddelta * delta;
    auto derL3 = (model.corr.get_EOS(0).alphar(tau, deltaplus) - model.corr.get_EOS(0).alphar(tau, deltaminus)) / (2.0 * ddelta) * delta;
    auto Zexact = derL3 + 1.0;

    if (is_propane) {
        auto d3 = static_cast<double>((derL2_2nd - derL3) / derL2_2nd);
        auto d34th = static_cast<double>((derL2_4th - derL2_2nd) / derL2_2nd);
        if (std::abs(d3) > 1e-100) { // yes, we have ridiculously accurate values
            throw std::invalid_argument("Derivatives are not exactly the same in teqp and in standalone implementation");
        }
    }

    calc_output o;
    o.Zexact = static_cast<double>(Zexact);
    o.Zteqp = Zteqp;
    o.Ar01exact = static_cast<double>(derL3);
    o.Ar01teqp = Ar01teqp;

    // Now do the third-order derivative of alphar, as a further test
    // Define a generic lambda function taking rho
    auto ff = [&](const my_float& rho) -> my_float {
        auto tau = forceeval(Tc/T);
        auto delta = forceeval(rho/rhoc);
        return alphar_Lemmon2009<my_float>(tau, delta);
    };
    my_float drho = 1e-30*rho;

    auto Ar02_extended = centered_diff<2,6>(ff,static_cast<my_float>(rho),drho)*pow(rho, 2);
    o.Ar02exact = static_cast<double>(Ar02_extended);
    o.Ar02teqp = tdx::template get_Ar0n<2>(model, T, rho, z)[2];

    o.Ar03exact = static_cast<double>(centered_diff<3,6>(ff,static_cast<my_float>(rho),drho)*pow(rho, 3));
    o.Ar03teqp = tdx::template get_Ar0n<3>(model, T, rho, z)[3];
    
    return o;
}

int do_one(const std::string &RPname, const std::string &teqpname)
{
    REFPROP_setup(RPname);
    auto model = build_multifluid_model({ teqpname }, "../mycp", "../mycp/dev/mixtures/mixture_binary_pairs.json");
    Eigen::ArrayXd z(1); z = 1.0;
    bool is_propane = (RPname == "PROPANE");
    double Tt = (is_propane) ? 85.525 : 273.16, 
           Tc = (is_propane) ? 369.89 : 647.096;
    int NT = 200;
    nlohmann::json outputs = nlohmann::json::array();
    for (double T : Eigen::ArrayXd::LinSpaced(NT, Tt, Tc)) {
        auto o = REFPROP_sat(T);

        // Pressure for each phase via REFPROP
        double pL = -1; PRESSdll(o.T, o.rhoLmol_L, &(z[0]), pL);
        double RL = -1; RMIX2dll(&(z[0]), RL);
        double ZLRP = pL/(o.rhoLmol_L*RL*o.T); // Units cancel (factor of 1000 in pL and RL)

        double Tr = -1, Dr = -1;
        REDXdll(&(z[0]), Tr, Dr);
        
        double pV = -1; PRESSdll(o.T, o.rhoVmol_L, &(z[0]), pV);
        double RV = -1; RMIX2dll(&(z[0]), RV);
        double ZVRP = pV/(o.rhoVmol_L*RV*o.T); // Units cancel (factor of 1000 in pV and RV)
        
        int itau = 0, idelta = 3; double tau = Tr / o.T, deltaL = o.rhoLmol_L / Dr, Ar03LRP = -1;
        double deltaV = o.rhoVmol_L / Dr, Ar03VRP = -1;
        PHIXdll(itau, idelta, tau, deltaL, &(z[0]), Ar03LRP);
        PHIXdll(itau, idelta, tau, deltaV, &(z[0]), Ar03VRP);
        
        double Ar01LRP = -1, Ar01VRP = -1;
        idelta = 1;
        PHIXdll(itau, idelta, tau, deltaL, &(z[0]), Ar01LRP);
        PHIXdll(itau, idelta, tau, deltaV, &(z[0]), Ar01VRP);

        double Ar02LRP = -1, Ar02VRP = -1;
        idelta = 2;
        PHIXdll(itau, idelta, tau, deltaL, &(z[0]), Ar02LRP);
        PHIXdll(itau, idelta, tau, deltaV, &(z[0]), Ar02VRP);

        double rhoL = o.rhoLmol_L * 1000.0, rhoV = o.rhoVmol_L*1000.0;
        for (double Q : { 0, 1 }) {
            double rho = (Q == 0) ? rhoL : rhoV;
            auto c = with_teqp_and_boost(model, T, rho, z, is_propane);
            double Zratiominus1 = c.Zteqp / c.Zexact - 1, Ar01ratiominus1 = c.Ar01teqp / c.Ar01exact - 1;
            outputs.push_back({
                {"T / K", T},
                {"Q", Q},
                {"rho / mol/m^3", rho},
                {"Zteqp", c.Zteqp},
                {"Zexact", c.Zexact},
                {"ratio-1", Zratiominus1},
                {"ZRP", ((Q == 0) ? ZLRP : ZVRP)},
                {"Ar03teqp", c.Ar03teqp},
                {"Ar03exact", c.Ar03exact},
                {"ratio03-1", c.Ar03teqp / c.Ar03exact - 1},
                {"Ar01teqp", c.Ar01teqp},
                {"Ar01exact", c.Ar01exact},
                {"ratio01-1", Ar01ratiominus1},
                {"Ar02teqp", c.Ar02teqp},
                {"Ar02exact", c.Ar02exact},
                {"Ar01RP", ((Q == 0) ? Ar01LRP : Ar01VRP)},
                {"Ar02RP", ((Q == 0) ? Ar02LRP : Ar02VRP)},
                {"Ar03RP", ((Q == 0) ? Ar03LRP : Ar03VRP)},
                });
        }
        std::cout << "Completed:" << T << std::endl;
    }
    std::ofstream file(RPname + "_saturation_Z_accuracy.json");
    file << outputs;
    return EXIT_SUCCESS;
}

int main()
{
    do_one("PROPANE", "n-Propane"); 
    do_one("WATER", "Water");
    return EXIT_SUCCESS;
}
