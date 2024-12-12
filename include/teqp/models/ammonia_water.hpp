#pragma once
#include <string>
#include <Eigen/Dense>
#include "teqp/models/multifluid.hpp"

namespace teqp{

    class AmmoniaWaterTillnerRoth {
    public:
        // The EOS are hard-coded to make sure they cannot be changed
        const nlohmann::json water_Wagner = nlohmann::json::parse(R"delim(
            {"ANCILLARIES": {"melting_line": {"BibTeX": "IAPWS-Melting-2011", "T_m": -1, "parts": [{"T_0": 273.16, "T_max": 251.165, "T_min": 273.16, "a": [-1195393.37, -80818.3159, -3338.2686], "p_0": 611.657, "t": [3.0, 25.75, 103.75]}, {"T_0": 251.165, "T_max": 256.164, "T_min": 251.165, "a": [0.299948], "p_0": 208566000.0, "t": [60]}, {"T_0": 256.164, "T_max": 273.31, "T_min": 256.164, "a": [1.18721], "p_0": 350100000.0, "t": [8]}, {"T_0": 273.31, "T_max": 355, "T_min": 273.31, "a": [1.07476], "p_0": 623400000.0, "t": [4.6]}], "type": "polynomial_in_Tr"}, "pS": {"T_r": 647.096, "Tmax": 647.0959999999985, "Tmin": 273.16, "description": "p'' = pc*exp(Tc/T*sum(n_i*theta^t_i))", "max_abserror_percentage": 0.01384518934277601, "n": [-9.75639641045262, 3.3357600887120102, -1.10029278432831, 0.02037617155190105, -2.6668589845604367, 6.676721087238668], "reducing_value": 22064000.0, "t": [1.018, 1.206, 2.327, 5.753, 4.215, 14.951], "type": "pV", "using_tau_r": true}, "rhoL": {"T_r": 647.096, "Tmax": 647.0959999999985, "Tmin": 273.16, "description": "rho' = rhoc*(1+sum(n_i*theta^t_i))", "max_abserror_percentage": 0.14434134863475778, "n": [0.8157021355019343, 2.0434712177006693, -78.58278372496308, 1026.4273940070307, -2290.5642779377695, 8420.141408210317], "reducing_value": 17873.72799560906, "t": [0.276, 0.455, 7.127, 9.846, 11.707, 17.805], "type": "rhoLnoexp", "using_tau_r": false}, "rhoV": {"T_r": 647.096, "Tmax": 647.0959999999985, "Tmin": 273.16, "description": "rho'' = rhoc*exp(Tc/T*sum(n_i*theta^t_i))", "max_abserror_percentage": 0.2457203054771817, "n": [0.9791749335365787, -2.6190679042770215, -3.9166443712365235, -20.313306821636637, 16.497589490043744, -125.36580458432083], "reducing_value": 17873.72799560906, "t": [0.21, 0.262, 0.701, 3.909, 4.076, 17.459], "type": "rhoV", "using_tau_r": true}, "surface_tension": {"BibTeX": "Mulero-JPCRD-2012", "Tc": 647.096, "a": [-0.1306, 0.2151], "description": "sigma = sum(a_i*(1-T/Tc)^n_i)", "n": [2.471, 1.233]}}, "EOS": [{"BibTeX_CP0": "", "BibTeX_EOS": "Wagner-JPCRD-2002", "STATES": {"hs_anchor": {"T": 711.8056, "T_units": "K", "hmolar": 43781.065905123054, "hmolar_units": "J/mol", "p": 38398578.56983617, "p_units": "Pa", "rhomolar": 16086.355196048155, "rhomolar_units": "mol/m^3", "smolar": 87.17706239787964, "smolar_units": "J/mol/K"}, "reducing": {"T": 647.096, "T_units": "K", "hmolar": 37556.74658424395, "hmolar_units": "J/mol", "p": 22064000, "p_units": "Pa", "rhomolar": 17873.72799560906, "rhomolar_units": "mol/m^3", "smolar": 79.4039479748897, "smolar_units": "J/mol/K"}, "sat_min_liquid": {"T": 273.16, "T_units": "K", "hmolar": 0.011021389918129065, "hmolar_units": "J/mol", "p": 611.6548008968684, "p_units": "Pa", "rhomolar": 55496.95513999978, "rhomolar_units": "mol/m^3", "smolar": -6.248886779394275e-11, "smolar_units": "J/mol/K"}, "sat_min_vapor": {"T": 273.16, "T_units": "K", "hmolar": 45054.657349954025, "hmolar_units": "J/mol", "p": 611.6548008968684, "p_units": "Pa", "rhomolar": 0.2694716052752858, "rhomolar_units": "mol/m^3", "smolar": 164.93862020846, "smolar_units": "J/mol/K"}}, "T_max": 2000, "T_max_units": "K", "Ttriple": 273.16, "Ttriple_units": "K", "acentric": 0.3442920843, "acentric_units": "-", "alpha0": [{"a1": -8.3204464837497, "a2": 6.6832105275932, "type": "IdealGasHelmholtzLead"}, {"a": 3.00632, "type": "IdealGasHelmholtzLogTau"}, {"n": [0.012436, 0.97315, 1.2795, 0.96956, 0.24873], "t": [1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105], "type": "IdealGasHelmholtzPlanckEinstein"}], "alphar": [{"d": [1, 1, 1, 2, 2, 3, 4, 1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11, 13, 15, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9, 10, 10, 12, 3, 4, 4, 5, 14, 3, 6, 6, 6], "l": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 6, 6, 6, 6], "n": [0.012533547935523, 7.8957634722828, -8.7803203303561, 0.31802509345418, -0.26145533859358, -0.0078199751687981, 0.0088089493102134, -0.66856572307965, 0.20433810950965, -6.6212605039687e-05, -0.19232721156002, -0.25709043003438, 0.16074868486251, -0.040092828925807, 3.9343422603254e-07, -7.5941377088144e-06, 0.00056250979351888, -1.5608652257135e-05, 1.1537996422951e-09, 3.6582165144204e-07, -1.3251180074668e-12, -6.2639586912454e-10, -0.10793600908932, 0.017611491008752, 0.22132295167546, -0.40247669763528, 0.58083399985759, 0.0049969146990806, -0.031358700712549, -0.74315929710341, 0.4780732991548, 0.020527940895948, -0.13636435110343, 0.014180634400617, 0.0083326504880713, -0.029052336009585, 0.038615085574206, -0.020393486513704, -0.0016554050063734, 0.0019955571979541, 0.00015870308324157, -1.638856834253e-05, 0.043613615723811, 0.034994005463765, -0.076788197844621, 0.022446277332006, -6.2689710414685e-05, -5.5711118565645e-10, -0.19905718354408, 0.31777497330738, -0.11841182425981], "t": [-0.5, 0.875, 1, 0.5, 0.75, 0.375, 1, 4, 6, 12, 1, 5, 4, 2, 13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10, 10, 3, 7, 10, 10, 6, 10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23, 23, 10, 50, 44, 46, 50], "type": "ResidualHelmholtzPower"}, {"beta": [150, 150, 250], "d": [3, 3, 3], "epsilon": [1, 1, 1], "eta": [20, 20, 20], "gamma": [1.21, 1.21, 1.25], "n": [-31.306260323435, 31.546140237781, -2521.3154341695], "t": [0, 1, 4], "type": "ResidualHelmholtzGaussian"}, {"A": [0.32, 0.32], "B": [0.2, 0.2], "C": [28, 32], "D": [700, 800], "a": [3.5, 3.5], "b": [0.85, 0.95], "beta": [0.3, 0.3], "n": [-0.14874640856724, 0.31806110878444], "type": "ResidualHelmholtzNonAnalytic"}], "critical_region_splines": {"T_max": 647.096, "T_min": 647.0954167354802, "_note": "Coefficients for the critical cubic spline.  T = c[0]*rho^3 + c[1]*rho^2 + c[2]*rho + c[3] with rho in mol/m^3 and T in K", "cL": [0.0, 0.0, -2.6319631259532804e-06, 647.1430429930077], "cV": [0.0, 0.0, 2.6053577217441603e-06, 647.0494325447503], "rhomolar_max": 18095.336161027783, "rhomolar_min": 17649.856810851063}, "gas_constant": 8.314371357587, "gas_constant_units": "J/mol/K", "molar_mass": 0.018015268, "molar_mass_units": "kg/mol", "p_max": 1000000000, "p_max_units": "Pa", "pseudo_pure": false}], "INFO": {"2DPNG_URL": "http://www.chemspider.com/ImagesHandler.ashx?id=937", "ALIASES": ["water", "WATER", "H2O", "h2o", "R718"], "CAS": "7732-18-5", "CHEMSPIDER_ID": 937, "ENVIRONMENTAL": {"ASHRAE34": "A1", "FH": 0, "GWP100": -1.0, "GWP20": -1.0, "GWP500": -1.0, "HH": 0, "Name": "Water", "ODP": -1.0, "PH": 0}, "FORMULA": "H_{2}O_{1}", "INCHI_KEY": "XLYOFNOQVPJJNP-UHFFFAOYSA-N", "INCHI_STRING": "InChI=1S/H2O/h1H2", "NAME": "Water", "REFPROP_NAME": "WATER", "SMILES": "O"}, "STATES": {"critical": {"T": 647.096, "T_units": "K", "hmolar": 37549.461223920865, "hmolar_units": "J/mol", "p": 22064000.0, "p_units": "Pa", "rhomolar": 17873.72799560906, "rhomolar_units": "mol/m^3", "smolar": 79.3940358438747, "smolar_units": "J/mol/K"}, "triple_liquid": {"T": 273.16, "T_units": "K", "hmolar": 0.011021389918129065, "hmolar_units": "J/mol", "p": 611.6548008968684, "p_units": "Pa", "rhomolar": 55496.95513999978, "rhomolar_units": "mol/m^3", "smolar": -6.248886779394275e-11, "smolar_units": "J/mol/K"}, "triple_vapor": {"T": 273.16, "T_units": "K", "hmolar": 45054.657349954025, "hmolar_units": "J/mol", "p": 611.6548008968684, "p_units": "Pa", "rhomolar": 0.2694716052752858, "rhomolar_units": "mol/m^3", "smolar": 164.93862020846, "smolar_units": "J/mol/K"}}}
        )delim");
        const nlohmann::json ammonia_TillnerRoth = nlohmann::json::parse(R"delim(
            {"ANCILLARIES": {"pS": {"T_r": 405.4, "Tmax": 405.39999999999924, "Tmin": 195.495, "description": "p'' = pc*exp(Tc/T*sum(n_i*theta^t_i))", "max_abserror_percentage": 0.05181632089212851, "n": [0.0016131703802769548, 0.01339903644956835, -6.4517151211338994, -4.349105787320862, 1.8295554336003688, 7.011533373126273], "reducing_value": 11333000.0, "t": [0.108, 0.435, 0.971, 4.023, 4.944, 17.494], "type": "pV", "using_tau_r": true}, "rhoL": {"T_r": 405.4, "Tmax": 405.39999999999924, "Tmin": 195.495, "description": "rho' = rhoc*(1+sum(n_i*theta^t_i))", "max_abserror_percentage": 0.8982235822398099, "n": [0.6217530323464998, 116.65648581323893, -116.06575843470785, 2.722640286484237, -2.6080795358675433, 18.393705416728867], "reducing_value": 13211.777154312385, "t": [0.217, 0.713, 0.724, 1.557, 3.994, 9.339], "type": "rhoLnoexp", "using_tau_r": false}, "rhoV": {"T_r": 405.4, "Tmax": 405.39999999999924, "Tmin": 195.495, "description": "rho'' = rhoc*exp(Tc/T*sum(n_i*theta^t_i))", "max_abserror_percentage": 0.802513430755436, "n": [1.2911157089516188, -1.4201786958021996, -145.4871281248165, 140.71212306607288, -3.147462914747211, -0.9907144617787934], "reducing_value": 13211.777154312385, "t": [0.037, 0.04, 0.584, 0.585, 2.858, 6.099], "type": "rhoV", "using_tau_r": true}, "surface_tension": {"BibTeX": "Mulero-JPCRD-2012", "Tc": 405.4, "a": [0.1028, -0.09453], "description": "sigma = sum(a_i*(1-T/Tc)^n_i)", "n": [1.211, 5.585]}}, "EOS": [{"BibTeX_CP0": "", "BibTeX_EOS": "TillnerRoth-DKV-1993", "STATES": {"hs_anchor": {"T": 445.94, "T_units": "K", "hmolar": 25012.280343289814, "hmolar_units": "J/mol", "p": 18635643.933141705, "p_units": "Pa", "rhomolar": 11890.599438881152, "rhomolar_units": "mol/m^3", "smolar": 75.46262589632495, "smolar_units": "J/mol/K"}, "reducing": {"T": 405.4, "T_units": "K", "hmolar": 21501.16668203028, "hmolar_units": "J/mol", "p": 11333000, "p_units": "Pa", "rhomolar": 13211.77715431239, "rhomolar_units": "mol/m^3", "smolar": 68.56438502935785, "smolar_units": "J/mol/K"}, "sat_min_liquid": {"T": 195.495, "T_units": "K", "hmolar": 0.14111811220522047, "hmolar_units": "J/mol", "p": 6091.2231081315085, "p_units": "Pa", "rhomolar": 43035.33929207322, "rhomolar_units": "mol/m^3", "smolar": -1.9440525067083775e-06, "smolar_units": "J/mol/K"}, "sat_min_vapor": {"T": 195.495, "T_units": "K", "hmolar": 25279.492873914965, "hmolar_units": "J/mol", "p": 6091.223108650368, "p_units": "Pa", "rhomolar": 3.763506027681136, "rhomolar_units": "mol/m^3", "smolar": 129.30945229032756, "smolar_units": "J/mol/K"}}, "T_max": 700, "T_max_units": "K", "Ttriple": 195.495, "Ttriple_units": "K", "acentric": 0.25601, "acentric_units": "-", "alpha0": [{"a1": -15.81502, "a2": 4.255726, "type": "IdealGasHelmholtzLead"}, {"a": -1, "type": "IdealGasHelmholtzLogTau"}, {"n": [11.47434, -1.296211, 0.5706757], "t": [0.3333333333333333, -1.5, -1.75], "type": "IdealGasHelmholtzPower"}, {"a1": -0.965940085369186, "a2": 0.723282863334932, "reference": "OTH", "type": "IdealGasHelmholtzEnthalpyEntropyOffset"}], "alphar": [{"d": [2, 1, 4, 1, 15, 3, 3, 1, 8, 2, 1, 8, 1, 2, 3, 2, 4, 3, 1, 2, 4], "l": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3], "n": [0.04554431, 0.7238548, 0.0122947, -1.858814, 2.141882e-11, -0.0143002, 0.3441324, -0.2873571, 2.352589e-05, -0.03497111, 0.02397852, 0.001831117, -0.04085375, 0.2379275, -0.03548972, -0.1823729, 0.02281556, -0.006663444, -0.008847486, 0.002272635, -0.0005588655], "t": [-0.5, 0.5, 1, 1.5, 3, 0, 3, 4, 4, 5, 3, 5, 6, 8, 8, 10, 10, 5, 7.5, 15, 30], "type": "ResidualHelmholtzPower"}], "gas_constant": 8.314471, "gas_constant_units": "J/mol/K", "molar_mass": 0.01703026, "molar_mass_units": "kg/mol", "p_max": 1000000000, "p_max_units": "Pa", "pseudo_pure": false}], "INFO": {"2DPNG_URL": "http://www.chemspider.com/ImagesHandler.ashx?id=217", "ALIASES": ["NH3", "ammonia", "R717", "AMMONIA"], "CAS": "7664-41-7", "CHEMSPIDER_ID": 217, "ENVIRONMENTAL": {"ASHRAE34": "B2", "FH": 1, "GWP100": -1.0, "GWP20": -1.0, "GWP500": -1.0, "HH": 3, "Name": "Ammonia", "ODP": -1.0, "PH": 0}, "FORMULA": "H_{3}N_{1}", "INCHI_KEY": "QGZKDVFQNNGYKY-UHFFFAOYSA-N", "INCHI_STRING": "InChI=1S/H3N/h1H3", "NAME": "Ammonia", "REFPROP_NAME": "AMMONIA", "SMILES": "N"}, "STATES": {"critical": {"T": 405.56, "T_units": "K", "hmolar": 21501.16668203028, "hmolar_units": "J/mol", "p": 11363400, "p_units": "Pa", "rhomolar": 13696, "rhomolar_units": "mol/m^3", "smolar": 68.56438502935785, "smolar_units": "J/mol/K"}, "triple_liquid": {"T": 195.495, "T_units": "K", "hmolar": 0.14111811220522047, "hmolar_units": "J/mol", "p": 6091.2231081315085, "p_units": "Pa", "rhomolar": 43035.33929207322, "rhomolar_units": "mol/m^3", "smolar": -1.9440525067083775e-06, "smolar_units": "J/mol/K"}, "triple_vapor": {"T": 195.495, "T_units": "K", "hmolar": 25279.492873914965, "hmolar_units": "J/mol", "p": 6091.223108650368, "p_units": "Pa", "rhomolar": 3.763506027681136, "rhomolar_units": "mol/m^3", "smolar": 129.30945229032756, "smolar_units": "J/mol/K"}}}
        )delim");
        const Eigen::ArrayXd a = (Eigen::ArrayXd(15) << 0,-1.855822E-02,5.258010E-02,3.552874E-10,5.451379E-06,-5.998546E-13,-3.687808E-06,0.2586192,-1.368072E-08,1.226146E-02,-7.181443E-02,9.970849E-02,1.0584086E-03,-0.1963687,-0.7777897).finished();
        const Eigen::ArrayXd t = (Eigen::ArrayXd(15) << 0,1.5,0.5,6.5,1.75,15,6,-1,4,3.5,0,-1,8,7.5,4).finished();
        const Eigen::ArrayXd d = (Eigen::ArrayXd(15) << 0,4,5,15,12,12,15,4,15,4,5,6,10,6,2).finished();
        const Eigen::ArrayXd e = (Eigen::ArrayXd(15) << 0,0,1,1,1,1,2,1,1,1,1,2,2,2,2).finished();

        const std::vector<teqp::EOSTerms> pures;

        const double TcNH3 = 405.40, TcH2O = 647.096, k_T = 0.9648407, alpha = 1.125455;
        const double vcNH3 = 0.01703026/225, vcH2O = 0.018015268/322, k_V = 1.2395117, beta = 0.8978069;

        AmmoniaWaterTillnerRoth() : pures(get_EOSs({ ammonia_TillnerRoth, water_Wagner })) {};

        template<typename MoleFracType>
        auto R(const MoleFracType&) const { return 8.314471; }

        template<typename TType, typename RhoType, typename MoleFracType>
        auto alphar_departure(const TType& tau, const RhoType& delta, const MoleFracType& xNH3) const
        {
            std::common_type_t<TType, RhoType, MoleFracType> summer = a[1]*pow(tau, t[1])*pow(delta, d[1]);
            for (auto i=2; i <= 6; ++i){ summer = summer + a[i]*pow(tau, t[i])*pow(delta, d[i])*exp(-pow(delta,e[i]));}
            for (auto i=7; i <= 13; ++i){ summer = summer + xNH3*a[i]*pow(tau, t[i])*pow(delta, d[i])*exp(-pow(delta,e[i]));}
            for (auto i=14; i <= 14; ++i) { summer = summer + xNH3*xNH3 * a[i] * pow(tau, t[i]) * pow(delta, d[i]) * exp(-pow(delta, e[i])); }
            double gamma = 0.5248379;
            // xNH3^gamma is not differentiable at xNH3=0, but limit when multiplied by zero is still zero
            if (getbaseval(xNH3) == 0) {
                return static_cast<decltype(summer)>(0.0);
            }
            return forceeval(xNH3 * (1 - pow(xNH3, gamma)) * summer);
        }
        
        template<typename MoleFracType>
        auto get_Treducing(const MoleFracType& molefrac) const {
            if (molefrac.size() != 2) {
                throw teqp::InvalidArgument("Wrong size of molefrac, should be 2");
            }
            auto xNH3 = molefrac[0];
            if (getbaseval(xNH3) == 0) {
                throw teqp::InvalidArgument("Tillner-Roth model cannot accept mole fraction of zero");
                return static_cast<decltype(xNH3)>(TcH2O);
            }
            auto Tred = forceeval(TcNH3*xNH3*xNH3 + TcH2O*(1-xNH3)*(1-xNH3) + 2.0*xNH3*(1.0-pow(xNH3, alpha))*k_T/2.0*(TcNH3 + TcH2O));
            return Tred;
        }
        template<typename MoleFracType>
        auto get_reducing_temperature(const MoleFracType& molefrac) const {
            return get_Treducing(molefrac);
        }
        
        template<typename MoleFracType>
        auto get_rhoreducing(const MoleFracType& molefrac) const {
            if (molefrac.size() != 2) {
                throw teqp::InvalidArgument("Wrong size of molefrac, should be 2");
            }
            auto xNH3 = molefrac[0];
            if (getbaseval(xNH3) == 0) {
                throw teqp::InvalidArgument("Tillner-Roth model cannot accept mole fraction of zero");
                return static_cast<decltype(xNH3)>(forceeval(1/vcH2O));
            }
            auto vred = forceeval(vcNH3*xNH3*xNH3 + vcH2O*(1-xNH3)*(1-xNH3) + 2.0*xNH3*(1.0-pow(xNH3, beta))*k_V/2.0*(vcNH3 + vcH2O));
            return forceeval(1/vred);
        }
        template<typename MoleFracType>
        auto get_reducing_density(const MoleFracType& molefrac) const {
            return get_rhoreducing(molefrac);
        }

        template<typename TType, typename RhoType, typename MoleFracType>
        auto alphar(const TType& T,
            const RhoType& rho,
            const MoleFracType& molefrac) const
        {
            if (molefrac.size() != 2) {
                throw teqp::InvalidArgument("Wrong size of molefrac, should be 2");
            }
            auto xNH3 = molefrac[0];
            auto Tred = get_Treducing(molefrac);
            auto rhored = get_rhoreducing(molefrac);
            auto delta = forceeval(rho / rhored);
            auto tau = forceeval(Tred / T);
            auto val_CS = pures[0].alphar(tau, delta)*xNH3 + pures[1].alphar(tau, delta)*(1-xNH3);
            auto val_dep = alphar_departure(tau, delta, xNH3);
            return forceeval(val_CS + val_dep);
        }
    };

} /* */