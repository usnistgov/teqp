#pragma once
#include <Eigen/Dense>
#include "teqp/exceptions.hpp"


namespace teqp {

    /*!
            Implementation of the polynomial extended corresponding states (pECS) mixture model \n
            The mixture model depends on coefficients for a temperature and density polynomial. \n
            Only valid for binary mixtures. \n
            The pECS reducing function follow the mathematical operations: \n
            \f$F(\bar x, Y, T, \rho ) = \sum_{i=1}^{N} x_{i}^{2} Y_{\mathrm{c},i} + \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} 2 x_{i} x_{j}  Y_{ij} f_{Y}(T,\rho).\f$ \n
            \f$f_{T,v} = \sum_{i}^{n} \sum_{j}^{m-i} c_{ij,T,v} \tau_{\mathrm{ECS}}^{i} \delta_{\mathrm{ECS}}^{j}\f$ \n
            \f$\tau_{\mathrm{ECS}} = \frac{Y_{T,ij}}{T}\f$ \n
            \f$\delta_{\mathrm{ECS}} = \rho Y_{v,ij}\f$ \n
            \f$Y_{T,ij} =  \sqrt{T_{\mathrm{c},i} T_{\mathrm{c},j}}\f$ \n
            \f$Y_{v,ij} =  \left( \frac{1}{\rho_{\mathrm{c},i}^{1/3}} + \frac{1}{\rho_{\mathrm{c},j}^{1/3}}\right)^{3}\f$ \n
            \f$c_{ij,T,v} = a_{ij,1} + a_{ij,2} x_{i} + a_{ij,3} x_{i}^{2}\f$. 
    */
    class Reducing_ECS {

    private:
        /*!
          Matrix containing the coefficients for the temperature polynominal: \f$a_{ij,T}\f$
        */
        Eigen::MatrixXd  tr_coeffs;
        /*!
                  Matrix containing the coefficients for the volume polynominal: \f$a_{ij,v}\f$
        */
        Eigen::MatrixXd dr_coeffs;

    public:
        /*!
          Critical temperature array of corresponding pure fluid contributions.
        */
        Eigen::ArrayXd Tc;
        /*!
          Critical volume array of corresponding pure fluid contributions.
        */
        Eigen::ArrayXd vc;
        template<typename ArrayLike>
        Reducing_ECS(const ArrayLike& Tc, const ArrayLike& vc, const nlohmann::json& jj) : Tc(Tc), vc(vc) {

            if (not jj.contains("tr_coeffs")) {
                throw teqp::InvalidArgument("tr_coeffs not in provided json");
            }

            if (not jj.contains("dr_coeffs")) {
                throw teqp::InvalidArgument("dr_coeffs not in provided json");
            }

            auto json_tr_coeffs = jj.at("tr_coeffs");
            auto json_dr_coeffs = jj.at("dr_coeffs");

            auto rows_tr = json_tr_coeffs.size();
            auto cols_tr = json_tr_coeffs[0].size();
            tr_coeffs.resize(rows_tr, cols_tr);

            auto rows_dr = json_dr_coeffs.size();
            auto cols_dr = json_dr_coeffs[0].size();
            dr_coeffs.resize(rows_dr, cols_dr);


            for (auto i = 0; i < rows_tr; ++i) {
                for (auto j = 0; j < cols_tr; ++j) {
                    tr_coeffs(i, j) = json_tr_coeffs[i][j];
                }
            }

            for (auto i = 0; i < rows_dr; ++i) {
                for (auto j = 0; j < cols_dr; ++j) {
                    dr_coeffs(i, j) = json_dr_coeffs[i][j];
                }
            }

        }

        /*!
            Reducing function for temperature
        */
        template <typename TTYPE, typename RHOTYPE, typename MoleFractions>
        auto get_tr(const TTYPE& temperature, const RHOTYPE& density, const MoleFractions& molefraction) const {
            
            auto p00 = tr_coeffs(0, 0) + molefraction[0] * tr_coeffs(0, 1) + molefraction[0] * molefraction[0] * tr_coeffs(0, 2);
            auto p10 = tr_coeffs(1, 0) + molefraction[0] * tr_coeffs(1, 1) + molefraction[0] * molefraction[0] * tr_coeffs(1, 2);
            auto p01 = tr_coeffs(2, 0) + molefraction[0] * tr_coeffs(2, 1) + molefraction[0] * molefraction[0] * tr_coeffs(2, 2);
            auto p20 = tr_coeffs(3, 0) + molefraction[0] * tr_coeffs(3, 1) + molefraction[0] * molefraction[0] * tr_coeffs(3, 2);
            auto p11 = tr_coeffs(4, 0) + molefraction[0] * tr_coeffs(4, 1) + molefraction[0] * molefraction[0] * tr_coeffs(4, 2);
            auto p02 = tr_coeffs(5, 0) + molefraction[0] * tr_coeffs(5, 1) + molefraction[0] * molefraction[0] * tr_coeffs(5, 2);
            auto dc_scale = 1.0/(0.125* pow( pow(vc[0],1.0/3.0)  + pow(vc[1],1.0/3.0),3.0));
            auto tc_scale = sqrt(Tc[0] * Tc[1]);
            auto tau = tc_scale / temperature;
            auto delta = density / dc_scale;

            auto tc_func = pow(molefraction[0], 2.0) * Tc[0] + pow(molefraction[1], 2.0) * Tc[1] + 2.0 * molefraction[0] * molefraction[1] * \
                (p00 + p10 * delta + p01 * tau + p20 * delta * delta + p02 * tau * tau + p11 * delta * tau) * tc_scale;
            return forceeval(tc_func);
        }


        /*!
            Reducing function for density
        */
        template <typename TTYPE, typename RHOTYPE, typename MoleFractions>
        auto get_dr(const TTYPE& temperature, const RHOTYPE& density, const MoleFractions& molefraction) const {

            auto p00 = dr_coeffs(0, 0) + molefraction[0] * dr_coeffs(0, 1) + molefraction[0] * molefraction[0] * dr_coeffs(0, 2);
            auto p10 = dr_coeffs(1, 0) + molefraction[0] * dr_coeffs(1, 1) + molefraction[0] * molefraction[0] * dr_coeffs(1, 2);
            auto p01 = dr_coeffs(2, 0) + molefraction[0] * dr_coeffs(2, 1) + molefraction[0] * molefraction[0] * dr_coeffs(2, 2);
            auto p20 = dr_coeffs(3, 0) + molefraction[0] * dr_coeffs(3, 1) + molefraction[0] * molefraction[0] * dr_coeffs(3, 2);
            auto p11 = dr_coeffs(4, 0) + molefraction[0] * dr_coeffs(4, 1) + molefraction[0] * molefraction[0] * dr_coeffs(4, 2);
            auto p02 = dr_coeffs(5, 0) + molefraction[0] * dr_coeffs(5, 1) + molefraction[0] * molefraction[0] * dr_coeffs(5, 2);
            auto dc_scale = 1.0/(0.125* pow( pow(vc[0],1.0/3.0)  + pow(vc[1],1.0/3.0),3.0));
            auto vc_scale = 1.0/dc_scale;
            auto tc_scale = sqrt(Tc[0] * Tc[1]);
            auto tau = tc_scale / temperature;
            auto delta = density / dc_scale;

            auto vc_ = pow(molefraction[0], 2.0) * vc[0] + pow(molefraction[1], 2.0) * vc[1] + 2.0 * molefraction[0] * molefraction[1] * \
                (p00 + p10 * delta + p01 * tau + p20 * delta * delta + p02 * tau * tau + p11 * delta * tau) * vc_scale;
            auto dc_func = (1.0 / vc_);
            return forceeval(dc_func);
        }

    };

    template<typename BaseClass>
    class MultiFluidAdapter_Ecs {

    private:
        std::string meta = "";

    public:
        const BaseClass& base;
        const Reducing_ECS redfunc;

        template<class VecType>
        auto R(const VecType& molefrac) const { return base.R(molefrac); }

        MultiFluidAdapter_Ecs(const BaseClass& base, Reducing_ECS&& redfunc) : base(base), redfunc(redfunc) {};

        /// Store some sort of metadata in string form (perhaps a JSON representation of the model?)
        void set_meta(const std::string& m) { meta = m; }
        
        /// Get the metadata stored in string form
        auto get_meta() const { return meta; }

        template<typename TType, typename RhoType, typename MoleFracType>
        auto alphar(const TType& T,
            const RhoType& rho,
            const MoleFracType& molefrac) const
        {
            if (static_cast<std::size_t>(molefrac.size()) != 2){
                 throw teqp::InvalidArgument("Wrong size of mole fractions - ECS mutant is only valid for a binary mixture");
            }
            auto Tred = forceeval(redfunc.get_tr(T, rho, molefrac));
            auto rhored = forceeval(redfunc.get_dr(T, rho, molefrac));
            auto delta = forceeval(rho / rhored);
            auto tau = forceeval(Tred / T);
            auto val = base.corr.alphar(tau, delta, molefrac);
            return forceeval(val);
        }
    };

    template<class Model>
    auto build_multifluid_ecs_mutant(const Model& model, const nlohmann::json& jj) {
        auto N = model.redfunc.Tc.size();
        auto red = model.redfunc;
        auto Tc = red.Tc, vc = red.vc;
        auto newred = Reducing_ECS(Tc, vc, jj);
        auto mfa = MultiFluidAdapter_Ecs(model, std::move(newred));
        mfa.set_meta(jj.dump());
        return mfa;
    }

}

