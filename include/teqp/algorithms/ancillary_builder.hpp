#pragma once

#include <iostream>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"

namespace teqp{
namespace ancillaries{

using namespace teqp;
using namespace teqp::cppinterface;

auto build_ancillaries(const AbstractModel& model, double Tcritguess, double rhocritguess, double Tmin, std::optional<nlohmann::json> flags_ = std::nullopt)
{
    nlohmann::json flags = flags_.value_or(nlohmann::json::object());
    
    bool ii = flags.is_object();
    bool verify = flags.value("verify", true);
    int Npts = flags.value("Npts", 1000);
    double Theta_nearcrit = flags.value("Theta_nearcrit", 0.01);
    
    auto [Tcrittrue, rhocrittrue] = model.solve_pure_critical(Tcritguess, rhocritguess);
    double Tclosec = (1-Theta_nearcrit)*Tcrittrue;
    auto rhovec = model.extrapolate_from_critical(Tcrittrue, rhocrittrue, Tclosec);
    double rhoL = rhovec[0], rhoV = rhovec[1];
    auto molefrac = (Eigen::ArrayXd(1) << 1.0).finished();
    double R = model.get_R(molefrac);
    double pcrittrue = rhocrittrue*R*Tcrittrue*(1+model.get_Ar01(Tcrittrue, rhocrittrue, molefrac));
    
    double dT = (Tcrittrue-Tmin)/(Npts-1);
    
    // Convenience function to get the density derivatives
    auto getdrhodTs = [&R, &molefrac](const auto& model, double T, double rhoL, double rhoV){
        double dpsatdT = model.dpsatdT_pure(T, rhoL, rhoV);

        auto get_drhodT = [&](double T, double rho){
            double dpdrho = R*T*(1 + 2*model.get_Ar01(T, rho, molefrac) + model.get_Ar02(T, rho, molefrac));
            double dpdT = R*rho*(1 + model.get_Ar01(T, rho, molefrac) - model.get_Ar11(T, rho, molefrac));
            return -dpdT/dpdrho + dpsatdT/dpdrho;
        };

        return std::make_tuple(get_drhodT(T, rhoL), get_drhodT(T, rhoV));
    };
    
    std::vector<double> Thetas_, rhoLs_, rhoVs_, pLs_;
    for (int i = 0; i < Npts; ++i){
        auto T = Tclosec - dT*i;
        auto rhovec = model.pure_VLE_T(T, rhoL, rhoV, 10);
        rhoL = rhovec[0]; rhoV = rhovec[1];
        auto [drhodTL, drhodTV] = getdrhodTs(model, T, rhoL, rhoV);
        rhoL += drhodTL*dT;
        rhoV += drhodTV*dT;
        T += dT;
        auto Theta = (Tcrittrue-T)/Tcrittrue;
        double pL = rhoL*R*T*(1+model.get_Ar01(T, rhoL, molefrac));
        
        Thetas_.push_back(Theta);
        rhoLs_.push_back(rhoL);
        rhoVs_.push_back(rhoV);
        pLs_.push_back(pL);
    }
    auto N = Thetas_.size();
    auto pLs = Eigen::Map<Eigen::ArrayXd>(&(pLs_[0]), N);
    auto rhoLs = Eigen::Map<Eigen::ArrayXd>(&(rhoLs_[0]), N);
    auto rhoVs = Eigen::Map<Eigen::ArrayXd>(&(rhoVs_[0]), N);
    auto Thetass = Eigen::Map<Eigen::ArrayXd>(&(Thetas_[0]), N);
    
    // Solve the least-squares problem for the polynomial coefficients
    Eigen::ArrayXd exponents = Eigen::ArrayXd::LinSpaced(10, 0, 4.5);
    Eigen::MatrixXd A(N,exponents.size());
    Eigen::VectorXd bL(N), bV(N), bpL(N);
    for (auto i = 0; i < exponents.size(); ++i){
        auto view = (Eigen::Map<Eigen::ArrayXd>(&(Thetass[0]), N));
        A.col(i) = view.pow(exponents[i]);
    }
    
    bL = (rhoLs/rhocrittrue)-1;
    auto TTr = 1.0-Thetass;
    bV = (rhoVs/rhocrittrue).log()*TTr;
    bpL = (pLs/pcrittrue).log()*TTr;
    
    Eigen::ArrayXd cLarray = A.colPivHouseholderQr().solve(bL).array();
    Eigen::ArrayXd cVarray = A.colPivHouseholderQr().solve(bV).array();
    Eigen::ArrayXd cpLarray = A.colPivHouseholderQr().solve(bpL).array();
    
    auto toj = [](const Eigen::ArrayXd& a){
        std::vector<double> o(a.size());
        Eigen::Map<Eigen::ArrayXd>(&(o[0]), o.size()) = a;
        return o;
    };
    
    nlohmann::json jrhoL = {
        {"T_r", Tcrittrue},
        {"Tmax", Tcrittrue},
        {"Tmin", Tmin},
        {"type", "rhoLnoexp"},
        {"description", "I'm a description"},
        {"n", toj(cLarray)},
        {"t", toj(exponents)},
        {"reducing_value", rhocrittrue},
        {"using_tau_r", false},
    };
    nlohmann::json jrhoV = {
        {"T_r", Tcrittrue},
        {"Tmax", Tcrittrue},
        {"Tmin", Tmin},
        {"type", "type"},
        {"description", "I'm a description"},
        {"n", toj(cVarray)},
        {"t", toj(exponents)},
        {"reducing_value", rhocrittrue},
        {"using_tau_r", true},
    };
    nlohmann::json jpsat = {
        {"T_r", Tcrittrue},
        {"Tmax", Tcrittrue},
        {"Tmin", Tmin},
        {"type", "type"},
        {"description", "I'm a description"},
        {"n", toj(cpLarray)},
        {"t", toj(exponents)},
        {"reducing_value", pcrittrue},
        {"using_tau_r", true},
    };
    nlohmann::json j{
        {"rhoL", jrhoL},
        {"rhoV", jrhoV},
        {"pS", jpsat}
    };
    auto anc = MultiFluidVLEAncillaries(j);
    
    if (verify){
        // Verify
        for (auto i = 0U; i < Thetas_.size(); ++i){
            double T = (1-Thetas_[i])*Tcrittrue;
            
            double rhoLanc = anc.rhoL(T);
            double rhoVanc = anc.rhoV(T);
//            double panc = anc.pL(T);
            if (std::abs(rhoLanc/rhoLs_[i]-1) > 1e-2){
                std::cout << T << " " << rhoLs_[i] << " " << rhoLanc << std::endl;
            }
            if (std::abs(rhoVanc/rhoVs_[i]-1) > 1e-2){
                std::cout << T << " " << rhoVs_[i] << " " << rhoVanc << std::endl;
            }
        }
    }
    
    return anc;
}


}
}
