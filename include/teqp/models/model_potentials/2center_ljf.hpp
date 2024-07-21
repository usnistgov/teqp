#pragma once

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include <map>

namespace teqp {

    namespace twocenterljf {
        // Implementation of the 2-center Lennard-Jones model
        // 2 model variants are available:
        // The Model of Mecke et al.:
        // https://link.springer.com/article/10.1007/BF02575128
        // and the revised Model of Lisal et al.:
        // https://link.springer.com/article/10.1023/B:IJOT.0000022332.12319.06

        // Note for dipolar contribution:
        // The original model was developed by Saager et al. (https://www.sciencedirect.com/science/article/abs/pii/0378381292850195)
        // these parameters are used here and the functional published by Kriebel and Winkelmann (https://aip.scitation.org/doi/10.1063/1.472764)
        enum model_types_2CLJF { MECKE = 1, LISAL = 2 };

        const std::map<std::string, model_types_2CLJF> modelmap = { {"2CLJF_Mecke",MECKE},{"2CLJF_Lisal",LISAL} };

        class ParameterContainer {
        public:
            // Parameters for Mecke and Lisal model
            // Density reducing parameters
            const std::map<model_types_2CLJF, std::valarray<double>> p_alpha = {
                   {MECKE, {1.0, 0.5296092, -0.4531784, 0.4421075}},
                   {LISAL, {1.0, 0.5296092, -0.4531784, 0.4421075}}
            };

            auto get_alpha_star_parameter(const std::string& model) {
                return p_alpha.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> p_eta_rho = {
                   {MECKE, {0.5256,3.2088804,-3.1499114,0.43049357}},
                   {LISAL, {0.5256,3.2088804,-3.1499114,0.43049357}}
            };

            auto get_eta_rho_parameter(const std::string& model) {
                return p_eta_rho.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> p_rho = {
              {MECKE, {0.3128,1.11519758,3.48878614,6.10644999}},
              {LISAL, {0.31258137,1.2240569,3.7974509,6.5490937}}
            };

            auto get_rho_parameter(const std::string& model) {
                return p_rho.at(modelmap.at(model));
            };

            // Temperature reducing parameters
            const std::map<model_types_2CLJF, std::valarray<double>> p_t = {
              {MECKE, {34.0122223,17.2324198,0.52922987,12.7653979}},
              {LISAL, {34.037352,17.733741,0.53237307,12.860239}}
            };

            auto get_T_parameter(const std::string& model) {
                return p_t.at(modelmap.at(model));
            };

            // Attractive parameteres
            const std::map<model_types_2CLJF, std::valarray<double>> c = {
              {MECKE, {-0.25359778252E+00,0.94270769752E-02,0.10937076431E-03,-0.45230360227E-05,-0.98945319827E+00,0.77816220730E+01,-0.19338901724E+02,0.16188444167E+02,-0.47837698146E+01,-0.37128104806E-05,0.11481369341E+01,-0.13600256513E+01,-0.34629572236E-05,-0.48388274860E+00,0.92061274747E+00,-0.38763633820E+00,-0.20652959726E+01,0.53102723110E+01,-0.45202666343E+01,0.12858167202E+01,0.31043103969E-03,0.76115392332E-05,-0.15141679018E+01,0.26132719232E+01,-0.88015285297E+00,-0.48730358072E-02,-0.14612399648E-01,-0.19908427778E-03,-0.29960728655E+00,0.25016932001E+00,0.16495699794E-01,0.35210453535E+00,-0.43243419699E+00,-0.31194438133E-01}},
              {LISAL, {-0.64211055047e-1,0.17682583145e-2,-0.62963373291e0,-0.35320115512e0,0.11339264270e2,-0.33311941616e2,0.37022843830e2,-0.18683743554e2,0.34566448842e1,-0.11216048862e-5,0.69315597535e0,-0.95242644353e0,0.13303429920e-1,-0.17518819492e-4,0.30942693727e-5,0.44671277084e-1,-0.84065404026e0,0.12662354443e1,-0.43706789738e0,0.34751432401e-5,-0.52988956334e-6,0.37399304905e-1,-0.32905342462e0,0.63121341882e-1,-0.20913100716e-2,-0.26852824281e-1,0.70733527178e-1,0.58291227149e-1,-0.76337837062e-1,-0.37502524667e-1,0.19201247728e-2,-0.76922623587e-1,0.12939011597e0,-0.37539710780e-1}}
            };

            auto get_c_parameter(const std::string& model) {
                return c.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> m = {
              {MECKE, {-1.5,-1.5,-1.5,-1.5,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-0.5,-0.5,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.0,-2.0,-2.0,-2.0,0.0,0.0,-4.0,-4.0,-4.0,-3.0,-3.0,-3.0}},
              {LISAL, {-1.50,-1.50,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.50,-0.50,-0.50,-0.50,-0.50,0.00,0.00,0.00,0.00,0.00,0.00,-3.00,-2.00,-2.00,-2.00,-1.00,0.00,-4.00,-4.00,-4.00,-4.00,0.00,0.00,0.00}}
            };

            auto get_m_parameter(const std::string& model) {
                return m.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> n = {
             {MECKE, {1.0,3.0,7.0,9.0,1.0,2.0,2.0,2.0,2.0,9.0,1.0,1.0,9.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,5.0,8.0,2.0,2.0,2.0,5.0,5.0,9.0,2.0,2.0,4.0,1.0,1.0,4}},
             {LISAL, {2.0,5.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,10.0,1.0,1.0,3.0,9.0,10.0,1.0,2.0,2.0,2.0,9.0,10.0,1.0,1.0,3.0,6.0,3.0,3.0,1.0,1.0,2.0,6.0,1.0,1.0,1.0}}
            };

            auto get_n_parameter(const std::string& model) {
                return n.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> o = {
                   {MECKE, {0,-1,-2,-3,0,-3,-2,-1,0,-3,-3,-2,-3,-3,-1,0,1,2,3,4,4,-1,-2,-1,0,-1,3,2,-3,-2,-1,-3,-1,4}},
                   {LISAL, {-3,-2,0,1,-3,-2,-1,0,1,-3,-3,-2,-1,-3,-2,0,-3,-2,-1,0,2,-2,0,0,1,2,-1,-3,0,-3,-2,-3,1,3}}
            };

            auto get_o_parameter(const std::string& model) {
                return o.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> p = {
                   {MECKE, {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1}},
                   {LISAL, {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1}}
            };

            auto get_p_parameter(const std::string& model) {
                return p.at(modelmap.at(model));
            };

            const std::map<model_types_2CLJF, std::valarray<double>> q = {
                   {MECKE, {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2}},
                   {LISAL, {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2}}
            };

            auto get_q_parameter(const std::string& model) {
                return q.at(modelmap.at(model));
            };

            // Parameters for the dipolar contribution (Parameters are from Saager et al: https://aip.scitation.org/doi/10.1063/1.472764
            const std::map<model_types_2CLJF, std::valarray<double>> cd = {
                {MECKE, {-0.423652173318e-01,0.204459397242e-01,0.664266837321e-01,-0.324168341478e-01,-0.741263275720e-02,-0.160855507113e-01,0.435623305093e-02,-0.105933370736e-03,-0.132000046519e-05,0.838157718194e-05,0.109144074057e-01,0.257960188278e-01,-0.544140085185e-03,0.349568484468e-02,-0.421407562467e-01,-0.745992658113e-02,0.146102252152e-03,0.566611094911e-03,-0.378643890614e-02,-0.365824539450e-01,0.169287932475e-01, 0.663866480778e-02,0.294409406715e-01,-0.112110434947e-01,-0.182144939032e-05,0.758594753989e-07,-0.216942306418e-04,-0.274025042954e-05}},
                {LISAL, {-0.423652173318e-01,0.204459397242e-01,0.664266837321e-01,-0.324168341478e-01,-0.741263275720e-02,-0.160855507113e-01,0.435623305093e-02,-0.105933370736e-03,-0.132000046519e-05,0.838157718194e-05,0.109144074057e-01,0.257960188278e-01,-0.544140085185e-03,0.349568484468e-02,-0.421407562467e-01,-0.745992658113e-02,0.146102252152e-03,0.566611094911e-03,-0.378643890614e-02,-0.365824539450e-01,0.169287932475e-01, 0.663866480778e-02,0.294409406715e-01,-0.112110434947e-01,-0.182144939032e-05,0.758594753989e-07,-0.216942306418e-04,-0.274025042954e-05 }}
            };

            auto get_dipolar_c_parameter(const std::string& model) {
                return cd.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> nd = {
                 {MECKE, {-5.0,-8.0,-4.0,-3.0,-10.0,-7.0,-10.0,-11.0,-15.0,-10.0,-2.0,-2.0,-1.0,-5.0,-3.0,-1.0,1.0,-9.0,-7.0,-2.0,-1.0,-5.0,-2.0,-1.0,-8.0,-5.0,1.0,-4.0}},
                 {LISAL, {-5.0,-8.0,-4.0,-3.0,-10.0,-7.0,-10.0,-11.0,-15.0,-10.0,-2.0,-2.0,-1.0,-5.0,-3.0,-1.0,1.0,-9.0,-7.0,-2.0,-1.0,-5.0,-2.0,-1.0,-8.0,-5.0,1.0,-4.0}}
            };

            auto get_dipolar_n_parameter(const std::string& model) {
                return nd.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> md = {
                 {MECKE, {2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,2.0,3.0,6.0,2.0,3.0,3.0,6.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,10.0,16.0,4.0,9.0}},
                 {LISAL, {2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,2.0,3.0,6.0,2.0,3.0,3.0,6.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,10.0,16.0,4.0,9.0 }}
            };

            auto get_dipolar_m_parameter(const std::string& model) {
                return md.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> kd = {
                  {MECKE, {5.0,6.0,7.0,7.0,9.0,9.0,11.0,15.0,18.0,18.0,5.0,5.0,5.0,6.0,6.0,6.0,6.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,8.0,10.0}},
                  {LISAL, {5.0,6.0,7.0,7.0,9.0,9.0,11.0,15.0,18.0,18.0,5.0,5.0,5.0,6.0,6.0,6.0,6.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,8.0,10.0}}
            };

            auto get_dipolar_k_parameter(const std::string& model) {
                return kd.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> od = {
                  {MECKE, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }},
                  {LISAL, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }}
            };

            auto get_dipolar_o_parameter(const std::string& model) {
                return od.at(modelmap.at(model));
            };

            // Parameters for the quadrupolar contribution (Parameters are from Saager et al: https://aip.scitation.org/doi/10.1063/1.472764
            const std::map<model_types_2CLJF, std::valarray<double>> cq = {
                 {MECKE, {-0.41215428089610E-2, 0.35578044173610E-2,-0.88809379838910E-3, 0.97379155960910E-4,-0.60423371932610E-7,-0.30447863314610E-4,-0.37893019633710E-3,-0.27538826735210E-1, 0.11830188842010E-1,-0.28345123056210E-2,-0.56770387482810E-4, 0.31470857321210E-2, 0.96378605256910E-3,-0.12759100242410E-2, 0.36374646323810E-3, 0.30106794309610E-4, 0.29177823112810E-6}},
                 {LISAL, {-0.41215428089610E-2, 0.35578044173610E-2,-0.88809379838910E-3, 0.97379155960910E-4,-0.60423371932610E-7,-0.30447863314610E-4,-0.37893019633710E-3,-0.27538826735210E-1, 0.11830188842010E-1,-0.28345123056210E-2,-0.56770387482810E-4, 0.31470857321210E-2, 0.96378605256910E-3,-0.12759100242410E-2, 0.36374646323810E-3, 0.30106794309610E-4, 0.29177823112810E-6}}
            };

            auto get_quadrupolar_c_parameter(const std::string& model) {
                return cq.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> nq = {
                 {MECKE, {-8.0,-6.0,-4.0,-10.0,-20.0,-8.0,-3.0,-3.0,-2.0,0.0,-5.0,-1.0,-3.0,-1.0,0.0,0.0,-10.0}},
                 {LISAL, {-8.0,-6.0,-4.0,-10.0,-20.0,-8.0,-3.0,-3.0,-2.0,0.0,-5.0,-1.0,-3.0,-1.0,0.0,0.0,-10.0}}
            };

            auto get_quadrupolar_n_parameter(const std::string& model) {
                return nq.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> mq = {
                 {MECKE, {2.0,2.0,2.0,2.0,2.0,2.0,8.0,2.0,2.0,2.0,8.0,2.0,5.0,5.0,5.0,8.0,7.0}},
                 {LISAL, {2.0,2.0,2.0,2.0,2.0,2.0,8.0,2.0,2.0,2.0,8.0,2.0,5.0,5.0,5.0,8.0,7.0}}
            };

            auto get_quadrupolar_m_parameter(const std::string& model) {
                return mq.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> kq = {
                  {MECKE, {11.0,12.0,13.0,16.0,19.0,20.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0,10.0,10.0,10.0,10.0,18.0}},
                  {LISAL, {11.0,12.0,13.0,16.0,19.0,20.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0,10.0,10.0,10.0,10.0,18.0}}
            };

            auto get_quadrupolar_k_parameter(const std::string& model) {
                return kq.at(modelmap.at(model));
            };

            const std::map < model_types_2CLJF, std::valarray<double>> oq = {
                  {MECKE, {1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}},
                  {LISAL, {1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}}
            };

            auto get_quadrupolar_o_parameter(const std::string& model) {
                return oq.at(modelmap.at(model));
            };
        };

        // Reducing functions for density and temperature 
        class ReducingDensity {
        public:
            std::valarray<double> p_alpha, p_eta_rho, p_rho;

            // EQ(3)
            auto get_alpha_star(const double& L) const {
                return forceeval(p_alpha[0] + p_alpha[1] * pow(L, 2) + p_alpha[2] * pow(L, 3.5) + p_alpha[3] * pow(L, 4));
            }

            // EQ(6)
            auto get_eta_over_rho(const double& L) const {
                return forceeval(p_eta_rho[0] + p_eta_rho[1] * pow(L, 2) + p_eta_rho[2] * pow(L, 2.5) + p_eta_rho[3] * pow(L, 4));
            }

            // EQ(8)
            auto get_rho_red(const double& L) const {
                return forceeval((p_rho[0] + p_rho[1] * L) / (1.0 + p_rho[2] * L + p_rho[3] * pow(L, 2)));
            }
        };

        class ReducingTemperature {
        public:
            std::valarray<double> p_t;

            // EQ(5)
            auto get_T_red(const double& L) const {
                if (L == 0) {
                    return 0.25 * sqrt((p_t[0] + p_t[1] * L) / (1.0 + p_t[2] * L + p_t[3] * pow(L, 2)));
                }
                else
                {
                    return  sqrt((p_t[0] + p_t[1] * L) / (1.0 + p_t[2] * L + p_t[3] * pow(L, 2)));
                }
            }
        };

        class AttractiveContribution {
        public:
            std::valarray<double> c, m, n, o, p, q;

            // EQ(7)
            template<typename TauType, typename DeltaType>
            auto alphar(const TauType& tau, const DeltaType& delta, const double& alpha) const {
                using result = std::common_type_t<TauType, DeltaType>;

                result r = 0.0;
                for (auto i = 0U; i < c.size(); ++i) {
                    r = r + c[i] * pow(tau, m[i]) * powi(delta, static_cast<int>(n[i])) * pow(alpha, o[i]) * exp(p[i] * powi(delta, static_cast<int>(q[i])));
                }
                return forceeval(r);
            }
        };


        class HardSphereContribution {
        public:
            const double a = 0.67793;
            const double g = 0.3674;

            // EQ(2)
            template<typename TauType, typename DeltaType>
            auto alphar(const TauType& tau, const DeltaType& delta, const double& alpha) const {

                auto eta = forceeval((delta / (a + (1.0 - a) * pow(tau, g))));
                auto r = (pow(alpha, 2) - 1.0) * log(1.0 - eta) + ((pow(alpha, 2) + 3 * alpha) * eta - 3 * alpha * powi(eta, 2)) / (pow(1.0 - eta, 2));
                return forceeval(r);
            }
        };

        class DipolarContribution {
        public:
            std::valarray<double> c, m, n, k, o;
            // EQ(9): https://aip.scitation.org/doi/10.1063/1.472764
            template<typename TauType, typename DeltaType>
            auto alphar(const TauType& tau, const DeltaType& delta, const double& mu_sq) const {
                using result = std::common_type_t<TauType, DeltaType>;

                result r = 0.0;
                for (auto i = 0U; i < c.size(); ++i) {
                    r = r + c[i] * pow(tau, n[i] / 2.0) * pow(delta, m[i] / 2.0) * pow(mu_sq, k[i] / 4.0) * exp(-o[i] * pow(delta, 2.0));
                }
                return forceeval(r);
            }
        };

        class QuadrupolarContribution {
        public:
            std::valarray<double> c, m, n, k, o;
            template<typename TauType, typename DeltaType>
            auto alphar(const TauType& tau, const DeltaType& delta, const double& mu_sq) const {
                using result = std::common_type_t<TauType, DeltaType>;

                result r = 0.0;
                for (auto i = 0U; i < c.size(); ++i) {
                    r = r + c[i] * pow(tau, n[i] / 2.0) * pow(delta, m[i] / 2.0) * pow(mu_sq, k[i] / 4.0) * exp(-o[i] * pow(delta, 2.0));
                }
                return forceeval(r);
            }
        };
        template<typename TypePolarContribution>
        class Twocenterljf {
        public:
            const ReducingDensity redD;
            const ReducingTemperature redT;
            const HardSphereContribution Hard;
            const AttractiveContribution Attr;
            const TypePolarContribution Pole;
            const double L;
            const double mu_sq;

            Twocenterljf(ReducingDensity&& redD, ReducingTemperature&& redT, HardSphereContribution&& Hard, const AttractiveContribution&& Attr, const TypePolarContribution&& Pole, const double L, const double& mu_sq) : redD(redD), redT(redT), Hard(Hard), Attr(Attr), Pole(Pole), L(L), mu_sq(mu_sq) {};

            template<typename TType, typename RhoType, typename MoleFracType>
            auto alphar(const TType& T_star,
                const RhoType& rho_dimer_star,
                const MoleFracType& /*molefrac*/) const
            {
                auto Tred = forceeval(redT.get_T_red(L));
                auto Rred = forceeval(redD.get_rho_red(L));
                auto eta_red = forceeval(redD.get_eta_over_rho(L));
                auto alpha = forceeval(redD.get_alpha_star(L));
                auto delta = forceeval(rho_dimer_star / Rred);
                auto tau = forceeval(T_star / Tred);
                auto delta_eta = forceeval(rho_dimer_star * eta_red);
                auto alphar_1 = Hard.alphar(tau, delta_eta, alpha);
                auto alphar_2 = Attr.alphar(tau, delta, alpha);
                auto val = forceeval(alphar_1 + alphar_2);
                if (mu_sq != 0.0){
                    val += Pole.alphar(tau, delta, mu_sq);
                }
                return forceeval(val);
            }

            template<class VecType>
            auto R(const VecType& /*molefrac*/) const {
                return 1.0;
            }
        };

        inline auto get_density_reducing(const std::string& name) {
            ParameterContainer pContainer;
            ReducingDensity red_rho;
            red_rho.p_alpha = pContainer.get_alpha_star_parameter(name);
            red_rho.p_eta_rho = pContainer.get_eta_rho_parameter(name);
            red_rho.p_rho = pContainer.get_rho_parameter(name);
            return red_rho;
        };

        inline auto get_temperature_reducing(const std::string& name) {
            ParameterContainer pContainer;
            ReducingTemperature red_T;
            red_T.p_t = pContainer.get_T_parameter(name);
            return red_T;
        };

        inline auto get_Attractive_contribution(const std::string& name) {
            ParameterContainer pContainer;
            AttractiveContribution eos;
            eos.c = pContainer.get_c_parameter(name);
            eos.m = pContainer.get_m_parameter(name);
            eos.n = pContainer.get_n_parameter(name);
            eos.o = pContainer.get_o_parameter(name);
            eos.p = pContainer.get_p_parameter(name);
            eos.q = pContainer.get_q_parameter(name);
            return eos;

        }

        inline auto get_HardSphere_contribution() {
            HardSphereContribution eos;
            return eos;
        }

        inline auto get_Dipolar_contribution(const std::string& name) {
            ParameterContainer pContainer;
            DipolarContribution eos;
            eos.c = pContainer.get_dipolar_c_parameter(name);
            eos.n = pContainer.get_dipolar_n_parameter(name);
            eos.m = pContainer.get_dipolar_m_parameter(name);
            eos.k = pContainer.get_dipolar_k_parameter(name);
            eos.o = pContainer.get_dipolar_o_parameter(name);
            return eos;
        }

        inline auto get_Quadrupolar_contribution(const std::string& name) {
            ParameterContainer pContainer;
            DipolarContribution eos;
            eos.c = pContainer.get_quadrupolar_c_parameter(name);
            eos.n = pContainer.get_quadrupolar_n_parameter(name);
            eos.m = pContainer.get_quadrupolar_m_parameter(name);
            eos.k = pContainer.get_quadrupolar_k_parameter(name);
            eos.o = pContainer.get_quadrupolar_o_parameter(name);
            return eos;
        }

        // build the 2-center Lennard-Jones model without dipole
        inline auto build_two_center_model(const std::string& model_version, const double& L = 0.0) {

            // Get reducing for temperature and density
            auto DC_funcs = get_density_reducing(model_version);
            auto TC_func = get_temperature_reducing(model_version);

            //// Get contributions to EOS ( Attractive and regular part)
            auto EOS_hard = get_HardSphere_contribution();
            auto EOS_att = get_Attractive_contribution(model_version);
            auto EOS_dipolar = get_Dipolar_contribution(model_version);
            double mu_sq = 0.0;

            // Build the 2-center Lennard-Jones model
            auto model = Twocenterljf(std::move(DC_funcs), std::move(TC_func), std::move(EOS_hard), std::move(EOS_att), std::move(EOS_dipolar), L, mu_sq);

            return model;
        }
    
        // build the 2-center Lennard-Jones model with dipole
        inline auto build_two_center_model_dipole(const std::string& model_version, const double& L = 0.0, const double& mu_sq = 0.0) {

            // Get reducing for temperature and density
            auto DC_funcs = get_density_reducing(model_version);
            auto TC_func = get_temperature_reducing(model_version);

            //// Get contributions to EOS ( Attractive and regular part)
            auto EOS_hard = get_HardSphere_contribution();
            auto EOS_att = get_Attractive_contribution(model_version);
            auto EOS_dipolar = get_Dipolar_contribution(model_version);

            // Build the 2-center Lennard-Jones model
            auto model = Twocenterljf(std::move(DC_funcs), std::move(TC_func), std::move(EOS_hard), std::move(EOS_att), std::move(EOS_dipolar), L, mu_sq);

            return model;
        }

        // build the 2-center Lennard-Jones model with quadrupole
        inline auto build_two_center_model_quadrupole(const std::string& model_version, const double& L = 0.0, const double& Q_sq = 0.0) {

            // Get reducing for temperature and density
            auto DC_funcs = get_density_reducing(model_version);
            auto TC_func = get_temperature_reducing(model_version);

            // Get contributions to EOS ( Attractive and regular part)
            auto EOS_hard = get_HardSphere_contribution();
            auto EOS_att = get_Attractive_contribution(model_version);

            // Get contributions to EOS ( Attractive and regular part)
            auto EOS_quadrupolar = get_Quadrupolar_contribution(model_version);

            // Build the 2-center Lennard-Jones model
            auto model = Twocenterljf(std::move(DC_funcs), std::move(TC_func), std::move(EOS_hard), std::move(EOS_att), std::move(EOS_quadrupolar), L, Q_sq);

            return model;
        }
    } // namespace twocenterljf
}; // namespace teqp
