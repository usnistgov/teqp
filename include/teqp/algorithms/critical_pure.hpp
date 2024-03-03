#pragma once

#include "nlohmann/json.hpp"

#include <Eigen/Dense>
#include "teqp/derivs.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include <optional>

using namespace teqp::cppinterface;

namespace teqp {
    
    /**
    * Calculate the criticality conditions for a pure fluid and its Jacobian w.r.t. the temperature and density
    * for additional fine tuning with multi-variate rootfinding
    *
    */
    inline auto get_pure_critical_conditions_Jacobian(const AbstractModel& model, const double T, const double rho,
        const std::optional<std::size_t>& alternative_pure_index = std::nullopt, const std::optional<std::size_t>& alternative_length = std::nullopt) {

        Eigen::ArrayXd z;
        if (!alternative_pure_index) {
            z = (Eigen::ArrayXd(1) << 1.0).finished();
        }
        else {
            z = Eigen::ArrayXd(alternative_length.value()); z.setZero();
            auto index = alternative_pure_index.value();
            if (index >= 0 && index < static_cast<std::size_t>(z.size())){
                z(index) = 1.0;
            }
            else{
                throw teqp::InvalidArgument("The provided alternative index of " + std::to_string(index) + " is out of range");
            }
        }
        auto R = model.get_R(z);

        auto ders = model.get_Ar04n(T, rho, z);

        auto dpdrho = R * T * (1 + 2 * ders[1] + ders[2]); // Should be zero at critical point
        auto d2pdrho2 = R * T / rho * (2 * ders[1] + 4 * ders[2] + ders[3]); // Should be zero at critical point

        auto resids = (Eigen::ArrayXd(2) << dpdrho, d2pdrho2).finished();

        /*  Sympy code for derivatives:
        import sympy as sy
        rho, R, Trecip,T = sy.symbols('rho,R,(1/T),T')
        alphar = sy.symbols('alphar', cls=sy.Function)(Trecip, rho)
        p = rho*R/Trecip*(1 + rho*sy.diff(alphar,rho))
        dTrecip_dT = -1/T**2

        sy.simplify(sy.diff(p,rho,3).replace(Trecip,1/T))

        sy.simplify(sy.diff(sy.diff(p,rho,1),Trecip)*dTrecip_dT)

        sy.simplify(sy.diff(sy.diff(p,rho,2),Trecip)*dTrecip_dT)
        */

        // Note: these derivatives are expressed in terms of 1/T and rho as independent variables
        auto Ar11 = model.get_Ar11(T, rho, z);
        auto Ar12 = model.get_Ar12(T, rho, z);
        auto Ar13 = model.get_Ar13(T, rho, z);

        auto d3pdrho3 = R * T / (rho * rho) * (6 * ders[2] + 6 * ders[3] + ders[4]);
        auto d_dpdrho_dT = R * (-(Ar12 + 2 * Ar11) + ders[2] + 2 * ders[1] + 1);
        auto d_d2pdrho2_dT = R / rho * (-(Ar13 + 4 * Ar12 + 2 * Ar11) + ders[3] + 4 * ders[2] + 2 * ders[1]);

        // Jacobian of resid terms w.r.t. T and rho
        Eigen::MatrixXd J(2, 2);
        J(0, 0) = d_dpdrho_dT; // d(dpdrho)/dT
        J(0, 1) = d2pdrho2; // d2pdrho2
        J(1, 0) = d_d2pdrho2_dT; // d(d2pdrho2)/dT
        J(1, 1) = d3pdrho3; // d3pdrho3

        return std::make_tuple(resids, J);
    }

    template <typename Model, typename Scalar, ADBackends backend = ADBackends::autodiff,
              typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, Model>::value>::type>
    auto get_pure_critical_conditions_Jacobian(const Model& model, const Scalar T, const Scalar rho,
        const std::optional<std::size_t>& alternative_pure_index = std::nullopt, const std::optional<std::size_t>& alternative_length = std::nullopt) {
        using namespace teqp::cppinterface::adapter;
        auto view_ = std::unique_ptr<AbstractModel>(view(model));
//        static_assert(std::is_base_of<teqp::cppinterface::AbstractModel, std::decay_t<decltype(*view_)> >::value);
        return get_pure_critical_conditions_Jacobian(*(view_), T, rho, alternative_pure_index, alternative_length);
    }

    template<typename Model, typename Scalar, ADBackends backend = ADBackends::autodiff, typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, Model>::value>::type>
    auto solve_pure_critical(const Model& model, const Scalar T0, const Scalar rho0, const std::optional<nlohmann::json>& flags = std::nullopt) {
        auto x = (Eigen::ArrayXd(2) << T0, rho0).finished();
        int maxsteps = 10;
        std::optional<std::size_t> alternative_pure_index;
        std::optional<std::size_t> alternative_length;
        if (flags){
            if (flags.value().contains("maxsteps")){
                maxsteps = flags.value().at("maxsteps");
            }
            if (flags.value().contains("alternative_pure_index")){
                auto i = flags.value().at("alternative_pure_index").get<int>();
                if (i < 0){ throw teqp::InvalidArgument("alternative_pure_index cannot be less than 0"); }
                alternative_pure_index = i;
            }
            if (flags.value().contains("alternative_length")){
                auto i = flags.value().at("alternative_length").get<int>();
                if (i < 2){ throw teqp::InvalidArgument("alternative_length cannot be less than 2"); }
                alternative_length = i;
            }
        }
        // A convenience method to make linear system solving more concise with Eigen datatypes
        // All arguments are converted to matrices, the solve is done, and an array is returned
        auto linsolve = [](const auto& a, const auto& b) {
            return a.matrix().colPivHouseholderQr().solve(b.matrix()).array().eval();
        };
        for (auto counter = 0; counter < maxsteps; ++counter) {
            auto [resids, Jacobian] = get_pure_critical_conditions_Jacobian(model, x[0], x[1], alternative_pure_index, alternative_length);
            auto v = linsolve(Jacobian, -resids);
            x += v;
        }
        return std::make_tuple(x[0], x[1]);
    }

    inline auto solve_pure_critical(const AbstractModel& model, const double T0, const double rho0, const std::optional<nlohmann::json>& flags = std::nullopt) {
        auto x = (Eigen::ArrayXd(2) << T0, rho0).finished();
        int maxsteps = 10;
        std::optional<std::size_t> alternative_pure_index;
        std::optional<std::size_t> alternative_length;
        if (flags){
            if (flags.value().contains("maxsteps")){
                maxsteps = flags.value().at("maxsteps");
            }
            if (flags.value().contains("alternative_pure_index")){
                auto i = flags.value().at("alternative_pure_index").get<int>();
                if (i < 0){ throw teqp::InvalidArgument("alternative_pure_index cannot be less than 0"); }
                alternative_pure_index = i;
            }
            if (flags.value().contains("alternative_length")){
                auto i = flags.value().at("alternative_length").get<int>();
                if (i < 2){ throw teqp::InvalidArgument("alternative_length cannot be less than 2"); }
                alternative_length = i;
            }
        }
        // A convenience method to make linear system solving more concise with Eigen datatypes
        // All arguments are converted to matrices, the solve is done, and an array is returned
        auto linsolve = [](const auto& a, const auto& b) {
            return a.matrix().colPivHouseholderQr().solve(b.matrix()).array().eval();
        };
        for (auto counter = 0; counter < maxsteps; ++counter) {
            auto [resids, Jacobian] = get_pure_critical_conditions_Jacobian(model, x[0], x[1], alternative_pure_index, alternative_length);
            auto v = linsolve(Jacobian, -resids);
            x += v;
        }
        return std::make_tuple(x[0], x[1]);
    }

    template <typename Model, typename Scalar, typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, Model>::value>::type>
    Scalar get_Brho_critical_extrap(const Model& model, const Scalar& Tc, const Scalar& rhoc, const std::optional<Eigen::ArrayXd>& z = std::nullopt) {

        using tdx = TDXDerivatives<Model, Scalar>;
        auto z_ = (Eigen::ArrayXd(1) << 1.0).finished();
        if (z){
            z_ = z.value();
        }
        auto R = model.R(z_);
        auto ders = tdx::template get_Ar0n<4>(model, Tc, rhoc, z_);
        //auto dpdrho = R*Tc*(1 + 2 * ders[1] + ders[2]); // Should be zero
        //auto d2pdrho2 = R*Tc/rhoc*(2 * ders[1] + 4 * ders[2] + ders[3]); // Should be zero
        auto d3pdrho3 = R * Tc / (rhoc * rhoc) * (6 * ders[2] + 6 * ders[3] + ders[4]);
        auto Ar11 = tdx::template get_Ar11(model, Tc, rhoc, z_);
        auto Ar12 = tdx::template get_Ar12(model, Tc, rhoc, z_);
        auto d2pdrhodT = R * (1 + 2 * ders[1] + ders[2] - 2 * Ar11 - Ar12);
        auto Brho = sqrt(6 * d2pdrhodT * Tc / d3pdrho3);
        return Brho;
    }


    template <typename Model, typename Scalar, typename = typename std::enable_if<not std::is_base_of<teqp::cppinterface::AbstractModel, Model>::value>::type>
    Eigen::Array<double, 2, 1> extrapolate_from_critical(const Model& model, const Scalar& Tc, const Scalar& rhoc, const Scalar& T, const std::optional<Eigen::ArrayXd>& z = std::nullopt)  {
        auto Brho = get_Brho_critical_extrap(model, Tc, rhoc, z);

        auto drhohat_dT = Brho / Tc;
        auto dT = T - Tc;

        auto drhohat = dT * drhohat_dT;
        auto rholiq = -drhohat / sqrt(1 - T / Tc) + rhoc;
        auto rhovap = drhohat / sqrt(1 - T / Tc) + rhoc;
        return (Eigen::ArrayXd(2) << rholiq, rhovap).finished();
    }

    

    inline double get_Brho_critical_extrap(const AbstractModel& model, const double& Tc, const double& rhoc, const std::optional<Eigen::ArrayXd>& z = std::nullopt) {
        
        auto z_ = (Eigen::ArrayXd(1) << 1.0).finished();
        if (z){
            z_ = z.value();
        }
        auto R = model.get_R(z_);
        auto ders = model.get_Ar04n(Tc, rhoc, z_);
        //auto dpdrho = R*Tc*(1 + 2 * ders[1] + ders[2]); // Should be zero
        //auto d2pdrho2 = R*Tc/rhoc*(2 * ders[1] + 4 * ders[2] + ders[3]); // Should be zero
        auto d3pdrho3 = R * Tc / (rhoc * rhoc) * (6 * ders[2] + 6 * ders[3] + ders[4]);
        auto Ar11 = model.get_Ar11(Tc, rhoc, z_);
        auto Ar12 = model.get_Ar12(Tc, rhoc, z_);
        auto d2pdrhodT = R * (1 + 2 * ders[1] + ders[2] - 2 * Ar11 - Ar12);
        auto Brho = sqrt(6 * d2pdrhodT * Tc / d3pdrho3);
        return Brho;
    }

    inline auto extrapolate_from_critical(const AbstractModel& model, const double& Tc, const double& rhoc, const double& T, const std::optional<Eigen::ArrayXd>& z = std::nullopt) {

        auto Brho = get_Brho_critical_extrap(model, Tc, rhoc, z);

        auto drhohat_dT = Brho / Tc;
        auto dT = T - Tc;

        auto drhohat = dT * drhohat_dT;
        auto rholiq = -drhohat / sqrt(1 - T / Tc) + rhoc;
        auto rhovap = drhohat / sqrt(1 - T / Tc) + rhoc;
        return (Eigen::ArrayXd(2) << rholiq, rhovap).finished();
    }
};
