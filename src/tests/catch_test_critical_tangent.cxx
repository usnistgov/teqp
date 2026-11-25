#include <catch2/catch_test_macros.hpp>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/constants.hpp"

#include "test_common.in"

using namespace teqp;
using teqp::cppinterface::AbstractModel;
using ct = CriticalTracing<AbstractModel, double, Eigen::ArrayXd>;

namespace {

struct LocusCase {
    std::string label;
    std::unique_ptr<AbstractModel> model;
    std::vector<std::tuple<double, Eigen::ArrayXd>> pure_critical_points; // one start per pure fluid
};

std::vector<LocusCase> locus_cases() {
    std::vector<LocusCase> cases;
    {
        // Argon + xenon, as in the vdW tracing test of catch_tests.cxx
        std::vector<double> Tc = { 150.687, 289.733 }, pc = { 4863000.0, 5842000.0 };
        nlohmann::json j = { {"kind", "vdW"}, {"model", {{"Tcrit / K", Tc}, {"pcrit / Pa", pc}}} };
        LocusCase c{ "vdW argon + xenon", teqp::cppinterface::make_model(j), {} };
        for (auto i = 0; i < 2; ++i) {
            Eigen::ArrayXd rhovec0 = Eigen::ArrayXd::Zero(2);
            rhovec0[i] = pc[i] / (constants::R_CODATA2017 * Tc[i]) / (3.0 / 8.0);
            c.pure_critical_points.emplace_back(Tc[i], rhovec0);
        }
        cases.push_back(std::move(c));
    }
    {
        std::vector<std::string> names = { "Methane", "Ethane" };
        const auto mf = build_multifluid_model(names, FLUIDDATAPATH);
        nlohmann::json j = { {"kind", "multifluid"}, {"model", {
            {"components", names}, {"root", FLUIDDATAPATH},
            {"BIP", FLUIDDATAPATH + "/dev/mixtures/mixture_binary_pairs.json"},
            {"departure", FLUIDDATAPATH + "/dev/mixtures/mixture_departure_functions.json"}}} };
        LocusCase c{ "multifluid methane + ethane", teqp::cppinterface::make_model(j), {} };
        for (auto i = 0; i < 2; ++i) {
            Eigen::ArrayXd rhovec0 = Eigen::ArrayXd::Zero(2);
            rhovec0[i] = 1.0 / mf.redfunc.vc[i];
            c.pure_critical_points.emplace_back(mf.redfunc.Tc[i], rhovec0);
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

/// The points of a default trace whose mole fractions are away from the pure-fluid ends
std::vector<std::tuple<double, Eigen::ArrayXd>> interior_trace_points(const AbstractModel& model, double T0, const Eigen::ArrayXd& rhovec0) {
    std::vector<std::tuple<double, Eigen::ArrayXd>> out;
    for (const auto& pt : ct::trace_critical_arclength_binary(model, T0, rhovec0)) {
        Eigen::ArrayXd rhovec(2);
        rhovec << pt.at("rho0 / mol/m^3").get<double>(), pt.at("rho1 / mol/m^3").get<double>();
        double z0 = rhovec[0] / rhovec.sum();
        if (z0 > 0.05 && z0 < 0.95) {
            out.emplace_back(pt.at("T / K").get<double>(), rhovec);
        }
    }
    return out;
}

double scaled_criticality_residual(const AbstractModel& model, double T, const Eigen::ArrayXd& rhovec) {
    auto c = ct::get_criticality_conditions(model, T, rhovec);
    double rho = rhovec.sum(), RT = model.R(rhovec / rho) * T;
    return std::max(std::abs(c[0]) / (RT / rho), std::abs(c[1]) / (RT / (rho * rho)));
}

/// The critical point at fixed T, by Newton steps on both criticality conditions with a centered-
/// difference Jacobian. The second condition is odd in the eigenvector v0, so every evaluation aligns
/// v0 with the one at the guess; otherwise a sign flip of v0 between Jacobian columns stalls Newton.
Eigen::ArrayXd exact_critical_point(const AbstractModel& model, double T, const Eigen::ArrayXd& rhovec_guess) {
    Eigen::ArrayXd v0 = ct::eigen_problem(model, T, rhovec_guess).v0;
    auto resid = [&](const Eigen::ArrayXd& rhovec) -> Eigen::Vector2d {
        Eigen::ArrayXd d = ct::get_derivs(model, T, rhovec, v0).tot;
        double rho = rhovec.sum(), RT = model.R(rhovec / rho) * T;
        return Eigen::Vector2d(d[2] / (RT / rho), d[3] / (RT / (rho * rho)));
    };
    Eigen::ArrayXd rhovec = rhovec_guess;
    for (int iter = 0; iter < 50; ++iter) {
        Eigen::Matrix2d J;
        for (int j = 0; j < 2; ++j) {
            double hj = 1e-6 * rhovec[j];
            Eigen::ArrayXd plus = rhovec, minus = rhovec;
            plus[j] += hj; minus[j] -= hj;
            J.col(j) = (resid(plus) - resid(minus)) / (2 * hj);
        }
        Eigen::ArrayXd step = J.colPivHouseholderQr().solve(-resid(rhovec)).array();
        rhovec += step;
        if ((step / rhovec).abs().maxCoeff() < 1e-14) {
            break;
        }
    }
    return rhovec;
}

}

TEST_CASE("Critical-locus tangent equals the temperature derivative of exact critical points", "[crit]")
{
    // At an exact critical point (T, rhovec), the tangent drhovec/dT along the critical locus must
    // match a centered difference of exact critical points at T +- h. Each exact point solves both
    // criticality conditions at fixed T (exact_critical_point), independently of the tangent.
    for (auto& c : locus_cases()) {
        for (auto& [T0, rhovec0] : c.pure_critical_points) {
            auto pts = interior_trace_points(*c.model, T0, rhovec0);
            REQUIRE(pts.size() > 5);
            double worst = 0;
            for (auto& [T, rhovec_traced] : pts) {
                Eigen::ArrayXd rhovec = exact_critical_point(*c.model, T, rhovec_traced);
                REQUIRE(scaled_criticality_residual(*c.model, T, rhovec) < 1e-10);

                Eigen::ArrayXd tangent = ct::get_drhovec_dT_crit(*c.model, T, rhovec);
                double h = 1e-5 * T;
                Eigen::ArrayXd rhovec_plus = exact_critical_point(*c.model, T + h, rhovec + h * tangent);
                Eigen::ArrayXd rhovec_minus = exact_critical_point(*c.model, T - h, rhovec - h * tangent);
                Eigen::ArrayXd fd = (rhovec_plus - rhovec_minus) / (2 * h);
                worst = std::max(worst, (tangent - fd).matrix().norm() / fd.matrix().norm());
            }
            CAPTURE(c.label, T0, worst);
            // worst is the truncation of this centered difference: it falls as h^2 from h = 1e-4 T to 1e-5 T, and at
            // h = 1e-5 T it is 5.9e-10 and 5.8e-10 for van der Waals and 3.1e-9 and 1.6e-8 for methane + ethane (from
            // methane and from ethane). At h = 3e-6 T the rounding of the solved points shows (up to 8.7e-9).
            // The bound is 12 times the largest value. Without the eigenvector-rotation terms in the Jacobian of the
            // criticality conditions, worst is 0.20-0.22.
            CHECK(worst < 2e-7);
        }
    }
}

TEST_CASE("A critical-locus trace with default options stays on the critical locus", "[crit]")
{
    // Every traced point must satisfy the criticality conditions; measured as the distance in molar
    // concentrations to the exact critical point at the traced temperature.
    for (auto& c : locus_cases()) {
        for (auto& [T0, rhovec0] : c.pure_critical_points) {
            auto pts = interior_trace_points(*c.model, T0, rhovec0);
            REQUIRE(pts.size() > 5);
            double worst = 0;
            for (auto& [T, rhovec_traced] : pts) {
                Eigen::ArrayXd rhovec = exact_critical_point(*c.model, T, rhovec_traced);
                REQUIRE(scaled_criticality_residual(*c.model, T, rhovec) < 1e-10);
                worst = std::max(worst, (rhovec_traced / rhovec - 1).abs().maxCoeff());
            }
            CAPTURE(c.label, T0, worst);
            // Without the eigenvector-rotation term in get_drhovec_dT_crit, worst is 0.09-0.12 here
            CHECK(worst < 1e-4);
        }
    }
}
