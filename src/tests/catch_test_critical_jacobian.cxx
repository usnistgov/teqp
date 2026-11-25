#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/constants.hpp"

#include "test_common.in"

using namespace teqp;
using teqp::cppinterface::AbstractModel;
using ct = CriticalTracing<AbstractModel, double, Eigen::ArrayXd>;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

/// A critical point at a fixed mole fraction x0 of the first component and the tangent d(rhovec)/dT there.
///
/// How the numbers were produced (a Python script outside teqp, with the fluid data of teqp/fluiddata at commit
/// aeb8423, last changed in f273f6c): the partial derivatives of Psi^r = alphar R(x) T rho with respect to T,
/// rho_0 and rho_1 up to order 4 in the concentrations were taken from alphar of the teqp model with multicomplex
/// numbers (mcx), and the ideal-gas part R(x) T sum_i rho_i (ln rho_i - 1) was added in closed form. The critical
/// point is the root of the two criticality conditions at fixed x0, by Newton iterations. The Jacobian of the
/// conditions was built from these partial derivatives with the formulas of the section "Tangent of the critical
/// curve" of doc/source/algorithms/critical_curves.ipynb, and the tangent is its null vector. The pins therefore
/// test the derivatives and the arithmetic of get_criticality_conditions_Jacobian, with the same formulas. The test case "The tangent
/// at solved critical points equals the difference of neighbouring solved critical points" tests the formulas,
/// with no number from outside teqp.
struct Pin {
    double T, rho0, rho1, drho0dT, drho1dT;
    Eigen::ArrayXd rhovec() const { return (Eigen::ArrayXd(2) << rho0, rho1).finished(); }
};

struct JacCase {
    std::string label;
    std::unique_ptr<AbstractModel> model;
    std::vector<Pin> pins;                                            // x0 = 0.3 and x0 = 0.7 (carbon dioxide + ethane: 0.1 and 0.3)
    bool R_differs;                                                   // the two pure-fluid equations use different gas constants
    double pin_tol;                                                   // relative agreement of the tangent with the pinned one
    std::vector<std::tuple<double, Eigen::ArrayXd>> pure_critical;    // the pure-fluid ends at which the Jacobian is the limit of the mixture Jacobian
};

std::unique_ptr<AbstractModel> make_multifluid(const std::vector<std::string>& names) {
    nlohmann::json j = { {"kind", "multifluid"}, {"model", {
        {"components", names}, {"root", FLUIDDATAPATH},
        {"BIP", FLUIDDATAPATH + "/dev/mixtures/mixture_binary_pairs.json"},
        {"departure", FLUIDDATAPATH + "/dev/mixtures/mixture_departure_functions.json"}}} };
    return teqp::cppinterface::make_model(j);
}

std::vector<JacCase> build_jac_cases() {
    std::vector<JacCase> cases;
    {
        std::vector<double> Tc = { 150.687, 289.733 }, pc = { 4863000.0, 5842000.0 };
        nlohmann::json j = { {"kind", "vdW"}, {"model", {{"Tcrit / K", Tc}, {"pcrit / Pa", pc}}} };
        JacCase c{ "vdW argon + xenon", teqp::cppinterface::make_model(j), {
            {261.90565787593994, 2296.7586013211953, 5359.10340308279, -84.01169245465654, 41.36410810030948},
            {208.4072224648659, 6869.869782376157, 2944.2299067326394, -84.59533496927358, 49.44288931776534}}, false, 1e-10, {} };
        for (auto i = 0; i < 2; ++i) {
            Eigen::ArrayXd rhovec0 = Eigen::ArrayXd::Zero(2);
            rhovec0[i] = pc[i] / (constants::R_CODATA2017 * Tc[i]) / (3.0 / 8.0);
            c.pure_critical.emplace_back(Tc[i], rhovec0);
        }
        cases.push_back(std::move(c));
    }
    auto add_multifluid = [&cases](const std::string& label, const std::vector<std::string>& names, std::vector<Pin> pins,
                                   bool R_differs, const std::vector<int>& pure_ends = { 0, 1 }, double pin_tol = 1e-10) {
        const auto mf = build_multifluid_model(names, FLUIDDATAPATH);
        JacCase c{ label, make_multifluid(names), std::move(pins), R_differs, pin_tol, {} };
        for (auto i : pure_ends) {
            Eigen::ArrayXd rhovec0 = Eigen::ArrayXd::Zero(2);
            rhovec0[i] = 1.0 / mf.redfunc.vc[i];
            c.pure_critical.emplace_back(mf.redfunc.Tc[i], rhovec0);
        }
        cases.push_back(std::move(c));
    };
    add_multifluid("multifluid methane + ethane", { "Methane", "Ethane" }, {
        {281.437127800818, 2456.0922298808205, 5730.881869721915, -106.617000835477, 57.78163623308096},
        {239.4955611119929, 6960.065526647727, 2982.8852257061694, -101.5818150330838, 65.66426371328674}}, true);
    add_multifluid("multifluid nitrogen + methane", { "Nitrogen", "Methane" }, {
        {173.8759393945451, 3215.08575593449, 7501.8667638471425, -187.90552232928516, 150.29652885204345},
        {147.47018467803593, 7933.601389326271, 3400.114881139831, -159.16053183675726, 164.22996347568792}}, false);
    add_multifluid("multifluid argon + methane", { "Argon", "Methane" }, {
        {181.67887334346744, 3326.165263744751, 7761.052282071085, -328.14047573082746, 235.0255811725758},
        {164.83289203266168, 8633.50592902753, 3700.0739695832276, -313.241019064116, 252.80748328837157}}, false);
    add_multifluid("multifluid methane + n-butane", { "Methane", "n-Butane" }, {
        {396.08087510184623, 1653.1626423842888, 3857.379498896674, -59.68824292004313, 12.108490654422924},
        {322.02858596172644, 6608.713231393839, 2832.3056705973604, -67.12490181168555, 11.962569430888564}}, true);
    // Carbon dioxide + ethane. The equation of state of carbon dioxide (Span and Wagner) has nonanalytic terms that
    // are singular at delta = tau = 1. In the multifluid model both pure-fluid ends evaluate them at that point, and
    // the critical curve passes next to it (at x(CO2) = 0.7: tau - 1 = -3.8e-5). The pins are at x(CO2) = 0.1 and 0.3
    // (tau - 1 = -7.1e-3 and -9.9e-3), where teqp's tangent and the multicomplex tangent agree to 4.3e-13.
    // The ethane end is a limit case: carbon dioxide is absent there, its terms enter the
    // derivatives with respect to its concentration through their value and their density derivatives at
    // delta = tau = 1, and the unit tangents at pure ethane and at x(CO2) = 1e-12 differ by 7.0e-11. The carbon
    // dioxide end is not a limit case: the Jacobian there contains mixed temperature and density derivatives of the
    // nonanalytic terms, which do not exist at delta = tau = 1, and the tangent there depends on the last digits of
    // the state. The unit tangents at pure carbon dioxide and at x(C2H6) = 1e-12 differ by 0.48 at T_c and 1/v_c of
    // the fluid file and by 3.6e-5 at the solved critical point, whose density differs from 1/v_c by 8e-8 (relative).
    add_multifluid("multifluid carbon dioxide + ethane", { "CarbonDioxide", "Ethane" }, {
        {302.45204161462107, 698.7132357477437, 6288.419121729693, -220.19912594247333, 154.9282760692765},
        {295.8288617599254, 2237.895620478676, 5221.756447783577, -264.3547240434911, 183.07185988538544}}, true, { 1 });
    return cases;
}

/// The models and pins of the test cases, built once for all of them
const std::vector<JacCase>& jac_cases() {
    static const std::vector<JacCase> cases = build_jac_cases();
    return cases;
}

}

TEST_CASE("Mixed derivatives of Psi^r from nested dual numbers", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            const auto& m = *c.model;
            double T = pin.T;
            Eigen::ArrayXd rhovec = pin.rhovec(), v = (Eigen::ArrayXd(2) << 0.6, -0.8).finished();
            double rho = rhovec.sum();
            Eigen::ArrayXd z = rhovec / rho;
            double R = m.R(z);
            CAPTURE(c.label, T);

            // One direction given four times: entries 1, 3, 7, 15 are the derivatives of order 1 to 4 along it
            Eigen::ArrayXd s = m.get_Psir_sigma_derivs(T, rhovec, v);
            Eigen::ArrayXXd V4(4, 2); V4 << v[0], v[1], v[0], v[1], v[0], v[1], v[0], v[1];
            Eigen::ArrayXd d4 = m.get_Psir_mixed_dir_derivs(T, rhovec, Eigen::ArrayXd::Zero(4), V4);
            REQUIRE(d4.size() == 16);
            CHECK_THAT(d4[0], WithinRel(s[0], 1e-13));
            CHECK_THAT(d4[1], WithinRel(s[1], 1e-12));
            CHECK_THAT(d4[3], WithinRel(s[2], 1e-11));
            CHECK_THAT(d4[7], WithinRel(s[3], 1e-10));
            CHECK_THAT(d4[15], WithinRel(s[4], 1e-9));
            // every entry with the same number of bits is the same derivative, reached through a different order of
            // operations; the tolerance per order is the one used against get_Psir_sigma_derivs above, because a
            // derivative of order 3 can be small against the terms it is summed from (AppleClang on arm64 contracts
            // to fused multiply-adds and differs from GCC beyond 1e-13 there)
            CHECK_THAT(d4[2], WithinRel(d4[1], 1e-12)); CHECK_THAT(d4[5], WithinRel(d4[3], 1e-11));
            CHECK_THAT(d4[11], WithinRel(d4[7], 1e-10)); CHECK_THAT(d4[14], WithinRel(d4[7], 1e-10));

            // Directions e0, e1 and temperature: the Hessian of Psi^r, d2Psi^r/dT drho_i and dPsi^r/dT = rho R (Ar00 - Ar10)
            Eigen::ArrayXXd V3(3, 2); V3 << 1, 0, 0, 1, 0, 0;
            Eigen::ArrayXd Tdir3 = (Eigen::ArrayXd(3) << 0, 0, 1).finished();
            Eigen::ArrayXd d3 = m.get_Psir_mixed_dir_derivs(T, rhovec, Tdir3, V3);
            REQUIRE(d3.size() == 8);
            Eigen::MatrixXd H = m.build_Psir_Hessian_autodiff(T, rhovec);
            Eigen::ArrayXd grad = m.build_Psir_gradient_autodiff(T, rhovec), gradT = m.build_d2PsirdTdrhoi_autodiff(T, rhovec);
            CHECK_THAT(d3[1], WithinRel(grad[0], 1e-12)); CHECK_THAT(d3[2], WithinRel(grad[1], 1e-12));
            CHECK_THAT(d3[3], WithinRel(H(0, 1), 1e-11));
            CHECK_THAT(d3[4], WithinRel(rho * R * (m.get_Ar00(T, rho, z) - m.get_Ar10(T, rho, z)), 1e-11));
            CHECK_THAT(d3[5], WithinRel(gradT[0], 1e-11)); CHECK_THAT(d3[6], WithinRel(gradT[1], 1e-11));

            // Temperature twice: d2Psi^r/dT2 = rho R Ar20 / T (absolute part in the tolerance: zero for van der Waals)
            Eigen::ArrayXXd V3T(3, 2); V3T << 1, 0, 0, 0, 0, 0;
            Eigen::ArrayXd TdirTT = (Eigen::ArrayXd(3) << 0, 1, 1).finished();
            Eigen::ArrayXd dd = m.get_Psir_mixed_dir_derivs(T, rhovec, TdirTT, V3T);
            double Ar20 = m.get_Ar20(T, rho, z);
            CHECK_THAT(dd[6] * T / (rho * R), WithinAbs(Ar20, 1e-10 * (1.0 + std::abs(Ar20))));
            CHECK_THAT(dd[1], WithinRel(grad[0], 1e-12));

            // Two or five directions are not offered
            CHECK_THROWS(m.get_Psir_mixed_dir_derivs(T, rhovec, Eigen::ArrayXd::Zero(2), Eigen::ArrayXXd::Zero(2, 2)));
            CHECK_THROWS(m.get_Psir_mixed_dir_derivs(T, rhovec, Eigen::ArrayXd::Zero(4), Eigen::ArrayXXd::Zero(3, 2)));
        }
    }
}

TEST_CASE("The gradient of R(x) with respect to the molar concentrations", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            const auto& m = *c.model;
            Eigen::ArrayXd rhovec = pin.rhovec();
            double rho = rhovec.sum();
            CAPTURE(c.label, pin.T);
            Eigen::ArrayXd gradR = m.build_R_gradient_autodiff(rhovec);
            REQUIRE(gradR.size() == 2);
            // R is constant or the mole-fraction-weighted mean of the pure-fluid values: dR/drho_j = (R_j - R(x))/rho
            for (auto j = 0; j < 2; ++j) {
                Eigen::ArrayXd ej = Eigen::ArrayXd::Zero(2); ej[j] = 1.0;
                CHECK_THAT(gradR[j], WithinAbs((m.R(ej) - m.R(rhovec / rho)) / rho, 1e-13 * m.R(rhovec / rho) / rho));
            }
            // Where the two equations of state use different gas constants (8.31451 for methane, argon and carbon
            // dioxide, 8.314472 for ethane and n-butane) the gradient is not zero; elsewhere it is zero to rounding
            const double scale = m.R(rhovec / rho) / rho;
            Eigen::ArrayXd e0 = Eigen::ArrayXd::Zero(2), e1 = Eigen::ArrayXd::Zero(2); e0[0] = 1.0; e1[1] = 1.0;
            CHECK((m.R(e0) != m.R(e1)) == c.R_differs);
            if (c.R_differs) { CHECK(gradR.abs().maxCoeff() > 1e-7 * scale); }
            else { CHECK(gradR.abs().maxCoeff() <= 1e-13 * scale); }
        }
    }
}

namespace {

double scaled_residual(const AbstractModel& model, double T, const Eigen::ArrayXd& rhovec, const Eigen::Array2d& C) {
    double rho = rhovec.sum(), RT = model.R(rhovec / rho) * T;
    return std::max(std::abs(C[0]) / (RT / rho), std::abs(C[1]) / (RT / (rho * rho)));
}

/// The unit tangent of the critical curve in the variables (T/T_s, rho_0/rho_s, rho_1/rho_s)
Eigen::Vector3d unit_tangent(const Eigen::Matrix<double, 2, 3>& J, double T_s, double rho_s) {
    Eigen::Vector3d r0 = J.row(0).transpose(), r1 = J.row(1).transpose();
    Eigen::Vector3d t = r0.cross(r1);
    t[0] /= T_s; t[1] /= rho_s; t[2] /= rho_s;
    return t / t.norm();
}

}

TEST_CASE("get_criticality_conditions_Jacobian reproduces the criticality conditions at the pinned critical points", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            auto cj = ct::get_criticality_conditions_Jacobian(*c.model, pin.T, pin.rhovec());
            CHECK(scaled_residual(*c.model, pin.T, pin.rhovec(), cj.C) < 1e-9);
            auto C = ct::get_criticality_conditions(*c.model, pin.T, pin.rhovec());
            const double rho = pin.rhovec().sum(), RT = c.model->R(pin.rhovec() / rho) * pin.T;
            CHECK_THAT(cj.C[0], WithinAbs(C[0], 1e-12 * RT / rho));
            CHECK_THAT(cj.C[1], WithinAbs(C[1], 1e-12 * RT / (rho * rho)));
            CHECK(!cj.at_infinite_dilution);
        }
    }
}

TEST_CASE("get_criticality_conditions_Jacobian passes a Taylor test", "[crit][critjac]")
{
    // |C(u + eps d) - C(u) - eps J d| must fall as eps^2: the ratio to eps |J||d| falls 10x per decade
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            auto base = ct::get_criticality_conditions_Jacobian(*c.model, pin.T, pin.rhovec());
            Eigen::Vector3d u(pin.T, pin.rho0, pin.rho1), d(0.37 * pin.T, -0.61 * pin.rho0, 0.83 * pin.rho1);
            std::vector<double> ratio;
            for (double eps : { 1e-3, 1e-4, 1e-5, 1e-6 }) {
                Eigen::Vector3d w = u + eps * d;
                Eigen::ArrayXd rv = (Eigen::ArrayXd(2) << w[1], w[2]).finished();
                auto pert = ct::get_criticality_conditions_Jacobian(*c.model, w[0], rv, base.ei.v0);
                double worst = 0;
                for (auto i = 0; i < 2; ++i) {
                    double lin = eps * base.J.row(i).dot(d), scale = eps * (base.J.row(i).transpose().array() * d.array()).abs().sum();
                    worst = std::max(worst, std::abs(pert.C[i] - base.C[i] - lin) / scale);
                }
                ratio.push_back(worst);
            }
            CAPTURE(ratio);
            for (auto i = 0; i + 1 < static_cast<int>(ratio.size()); ++i) {
                CHECK(ratio[i] / ratio[i + 1] >= 9.0);
            }
        }
    }
}

TEST_CASE("The tangent from get_criticality_conditions_Jacobian equals the pinned multicomplex tangent", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            auto cj = ct::get_criticality_conditions_Jacobian(*c.model, pin.T, pin.rhovec());
            Eigen::Vector3d r0 = cj.J.row(0).transpose(), r1 = cj.J.row(1).transpose();
            Eigen::Vector3d t = r0.cross(r1);
            CHECK_THAT(t[1] / t[0], WithinRel(pin.drho0dT, c.pin_tol));
            CHECK_THAT(t[2] / t[0], WithinRel(pin.drho1dT, c.pin_tol));
        }
    }
}

TEST_CASE("The tangent at solved critical points equals the difference of neighbouring solved critical points", "[crit][critjac]")
{
    // Nothing in this test case comes from outside teqp. The pin is the starting point only: the critical point is
    // solved at the mole fraction x0 of the pin and at x0 +- dx (critical_polish_fixedmolefrac), and the centred
    // difference of the two neighbours is compared with get_drhovec_dT_crit at the solved point at x0. After a change
    // of the fluid data the test cases with pinned numbers fail and this one passes; if this one fails, the
    // Jacobian is wrong.
    //
    // The largest relative deviation over the 12 critical points is 5.2e-6, 4.7e-7, 5.2e-8, 5.0e-9 and 2.6e-9 at
    // dx = 1e-3, 3e-4, 1e-4, 3e-5 and 1e-5 (carbon dioxide + ethane at x(CO2) = 0.1 and 0.3). It falls as dx^2 down
    // to 3e-5, so it is the truncation of the difference; at 1e-5 the rounding of the solved points shows. The
    // tolerance is 20 times the deviation at dx = 3e-5.
    const double dx = 3e-5;
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            const double x0 = pin.rho0 / (pin.rho0 + pin.rho1);
            auto [T0, rhovec0] = ct::critical_polish_fixedmolefrac(*c.model, pin.T, pin.rhovec(), x0);
            auto [Tp, rhovecp] = ct::critical_polish_fixedmolefrac(*c.model, T0, rhovec0, x0 + dx);
            auto [Tm, rhovecm] = ct::critical_polish_fixedmolefrac(*c.model, T0, rhovec0, x0 - dx);
            Eigen::ArrayXd difference = (rhovecp - rhovecm) / (Tp - Tm);
            Eigen::ArrayXd t = ct::get_drhovec_dT_crit(*c.model, T0, rhovec0).array();
            CHECK_THAT(t[0], WithinRel(difference[0], 1e-7));
            CHECK_THAT(t[1], WithinRel(difference[1], 1e-7));
        }
    }
}

TEST_CASE("The tangent from get_criticality_conditions_Jacobian changes by rounding for a relative change of 1e-13 in T", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            double rho_s = pin.rho0 + pin.rho1;
            auto t1 = unit_tangent(ct::get_criticality_conditions_Jacobian(*c.model, pin.T, pin.rhovec()).J, pin.T, rho_s);
            auto t2 = unit_tangent(ct::get_criticality_conditions_Jacobian(*c.model, pin.T * (1 + 1e-13), pin.rhovec()).J, pin.T, rho_s);
            CHECK((t1 - t2).norm() < 1e-9);
        }
    }
}

TEST_CASE("get_criticality_conditions_Jacobian at a pure fluid is the limit of the mixture Jacobian", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& [Tc, rhovec_pure] : c.pure_critical) {
            CAPTURE(c.label, Tc);
            double rho = rhovec_pure.sum();
            auto pure = ct::get_criticality_conditions_Jacobian(*c.model, Tc, rhovec_pure);
            CHECK(pure.at_infinite_dilution);
            CHECK(pure.J.allFinite());

            // The same temperature and density with a mole fraction 1e-12 of the other component
            Eigen::ArrayXd rhovec_dilute = rhovec_pure;
            for (auto i = 0; i < 2; ++i) { rhovec_dilute[i] = (rhovec_pure[i] == 0) ? 1e-12 * rho : (1 - 1e-12) * rho; }
            auto dilute = ct::get_criticality_conditions_Jacobian(*c.model, Tc, rhovec_dilute, pure.ei.v0);
            CHECK(!dilute.at_infinite_dilution);

            auto tp = unit_tangent(pure.J, Tc, rho), td = unit_tangent(dilute.J, Tc, rho);
            CHECK(std::min((tp - td).norm(), (tp + td).norm()) < 1e-8);

            // The second condition is odd in v0: with v0 oriented the other way, the second condition and the second
            // row of the Jacobian change sign, and the first condition and the first row do not
            const Eigen::ArrayXd minus_v0 = -pure.ei.v0;
            auto flipped = ct::get_criticality_conditions_Jacobian(*c.model, Tc, rhovec_pure, minus_v0);
            CHECK(flipped.at_infinite_dilution);
            CHECK((flipped.ei.v0 == minus_v0).all());
            CHECK(flipped.C[0] == pure.C[0]);
            CHECK(flipped.C[1] == -pure.C[1]);
            CHECK((flipped.J.row(0).array() == pure.J.row(0).array()).all());
            CHECK((flipped.J.row(1).array() == -pure.J.row(1).array()).all());
        }
    }
}

TEST_CASE("get_criticality_conditions_Jacobian is continuous from a small concentration of one component to zero", "[crit][critjac]")
{
    // At a mole fraction x of the minor component q, the component of v0 along q is of the order of x, and the Jacobian
    // contains it divided by rho_q. An iterative eigenvalue solver returns that component as zero below x of about
    // 1e-16; 3e-17 to 1e-15 are on both sides of that mole fraction. The unit tangent at x differs from the one of
    // the closed-form limit by a term linear in x (0.4 x to 700 x for these mixtures) and by rounding; the largest
    // difference over these mole fractions is 3.9e-11 (argon + methane at the methane end, x = 1e-14).
    //
    // Without an alignment vector, eigen_problem orients the eigenvectors by a rule on the vectors of the iterative
    // solver. Where that solver returns the minor component of v0 as zero, v0 has the sign it has at x = 0; this is
    // checked up to x = 3e-16. At larger x the rule can give the other sign: -e_p at the n-butane end of methane +
    // n-butane for x from 1e-14 to 1e-2.
    for (auto& c : jac_cases()) {
        for (auto& [Tc, rhovec_pure] : c.pure_critical) {
            double rho = rhovec_pure.sum();
            auto pure = ct::get_criticality_conditions_Jacobian(*c.model, Tc, rhovec_pure);
            auto tp = unit_tangent(pure.J, Tc, rho);
            for (double x : { 1e-100, 1e-20, 1e-17, 3e-17, 1e-16, 3e-16, 1e-15, 1e-14 }) {
                CAPTURE(c.label, Tc, x);
                Eigen::ArrayXd rhovec_dilute = rhovec_pure;
                for (auto i = 0; i < 2; ++i) { rhovec_dilute[i] = (rhovec_pure[i] == 0) ? x * rho : (1 - x) * rho; }
                auto dilute = ct::get_criticality_conditions_Jacobian(*c.model, Tc, rhovec_dilute, pure.ei.v0);
                CHECK(!dilute.at_infinite_dilution);
                CHECK(dilute.J.allFinite());
                auto td = unit_tangent(dilute.J, Tc, rho);
                CHECK(std::min((tp - td).norm(), (tp + td).norm()) < 1e-9);
                // The first condition is the smaller eigenvalue of a Hessian whose other eigenvalue is RT/rho_q
                CHECK_THAT(dilute.C[0], WithinAbs(pure.C[0], 1e-11 * c.model->R(rhovec_dilute / rho) * Tc / rho));
                if (x <= 3e-16) {
                    CHECK(ct::eigen_problem(*c.model, Tc, rhovec_dilute).v0.matrix().dot(pure.ei.v0.matrix()) > 0);
                }
            }
        }
    }
}

TEST_CASE("get_drhovec_dT_crit is the null vector of get_criticality_conditions_Jacobian", "[crit][critjac]")
{
    for (auto& c : jac_cases()) {
        for (auto& pin : c.pins) {
            CAPTURE(c.label, pin.T);
            Eigen::ArrayXd t = ct::get_drhovec_dT_crit(*c.model, pin.T, pin.rhovec()).array();
            REQUIRE(t.size() == 2);
            CHECK_THAT(t[0], WithinRel(pin.drho0dT, c.pin_tol));
            CHECK_THAT(t[1], WithinRel(pin.drho1dT, c.pin_tol));
        }
        for (auto& [Tc, rhovec_pure] : c.pure_critical) {
            CAPTURE(c.label, Tc);
            Eigen::ArrayXd t = ct::get_drhovec_dT_crit(*c.model, Tc, rhovec_pure).array();
            CHECK(t.allFinite());
        }
    }
}

TEST_CASE("get_drhovec_dT_crit throws for a state without a finite tangent, get_criticality_conditions_Jacobian for a negative concentration", "[crit][critjac]")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (auto& c : jac_cases()) {
        const auto& pin = c.pins.front();
        CAPTURE(c.label);
        Eigen::ArrayXd negative = pin.rhovec(), notanumber = pin.rhovec();
        negative[0] = -negative[0];
        notanumber[1] = nan;
        CHECK_THROWS_AS(ct::get_criticality_conditions_Jacobian(*c.model, pin.T, negative), std::invalid_argument);
        CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*c.model, pin.T, negative), std::invalid_argument);
        CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*c.model, nan, pin.rhovec()), std::invalid_argument);
        CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*c.model, pin.T, notanumber), std::invalid_argument);
        // The message names the state
        CHECK_THROWS_WITH(ct::get_drhovec_dT_crit(*c.model, nan, pin.rhovec()), ContainsSubstring("T = nan K, rhovec = ["));
        CHECK_THROWS_WITH(ct::get_criticality_conditions_Jacobian(*c.model, pin.T, negative), ContainsSubstring("rhovec = [-"));
        // get_criticality_conditions_Jacobian does not throw for a temperature or a concentration that is not finite;
        // its result is not finite
        CHECK_FALSE(ct::get_criticality_conditions_Jacobian(*c.model, nan, pin.rhovec()).J.allFinite());
        CHECK_FALSE(ct::get_criticality_conditions_Jacobian(*c.model, pin.T, notanumber).J.allFinite());
    }
    // A finite state without a finite tangent: methane + ethane at the concentrations of a pin and T = 2e-4 K, a
    // temperature that a trace which has left the critical curve reaches (test case "A trace that has left the
    // critical curve ends with finite points")
    const auto& c = jac_cases()[1];
    REQUIRE(c.label == "multifluid methane + ethane");
    CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*c.model, 2e-4, c.pins.front().rhovec()), std::invalid_argument);
    CHECK_THROWS_WITH(ct::get_drhovec_dT_crit(*c.model, 2e-4, c.pins.front().rhovec()), ContainsSubstring("not finite"));
}

TEST_CASE("get_criticality_conditions_Jacobian and get_drhovec_dT_crit with both concentrations zero or with three components", "[crit][critjac]")
{
    // Both concentrations zero: eigen_problem throws, and the two functions throw its exception
    for (const auto& c : jac_cases()) {
        CAPTURE(c.label);
        const auto& pin = c.pins.front();
        const Eigen::ArrayXd zero = Eigen::ArrayXd::Zero(2);
        CHECK_THROWS_AS(ct::eigen_problem(*c.model, pin.T, zero), std::invalid_argument);
        CHECK_THROWS_AS(ct::get_criticality_conditions_Jacobian(*c.model, pin.T, zero), std::invalid_argument);
        CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*c.model, pin.T, zero), std::invalid_argument);
    }
    // Three components: get_criticality_conditions_Jacobian is for binary mixtures, and get_drhovec_dT_crit throws its exception
    std::vector<double> Tc = { 150.687, 289.733, 190.564 }, pc = { 4863000.0, 5842000.0, 4599200.0 };
    nlohmann::json j = { {"kind", "vdW"}, {"model", {{"Tcrit / K", Tc}, {"pcrit / Pa", pc}}} };
    const auto model = teqp::cppinterface::make_model(j);
    const Eigen::ArrayXd rhovec = (Eigen::ArrayXd(3) << 2000.0, 3000.0, 1000.0).finished();
    CHECK_THROWS_WITH(ct::get_criticality_conditions_Jacobian(*model, 250.0, rhovec), ContainsSubstring("binary"));
    CHECK_THROWS_WITH(ct::get_drhovec_dT_crit(*model, 250.0, rhovec), ContainsSubstring("binary"));
    CHECK_THROWS_AS(ct::get_drhovec_dT_crit(*model, 250.0, rhovec), std::invalid_argument);
}

TEST_CASE("symmetric_eigen_2x2 returns ascending eigenvalues and orthonormal eigenvectors", "[crit][critjac]")
{
    auto check = [](double a, double b, double d) {
        auto [values, vectors] = ct::symmetric_eigen_2x2(a, b, d);
        Eigen::Matrix2d H; H << a, b, b, d;
        const double scale = std::max(std::abs(a), std::abs(d)) + std::abs(b);
        CHECK(values[0] <= values[1]);
        CHECK((H * vectors - vectors * values.matrix().asDiagonal()).cwiseAbs().maxCoeff() <= 1e-15 * scale);
        CHECK((vectors.transpose() * vectors - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff() <= 1e-15);
        return std::make_pair(values, vectors);
    };
    SECTION("b = 0: no rotation") {
        auto [values, vectors] = check(2.0, 0.0, 1.0);
        CHECK(values[0] == 1.0); CHECK(values[1] == 2.0);
        CHECK(vectors(0, 0) == 0.0); CHECK(std::abs(vectors(1, 0)) == 1.0);
    }
    SECTION("a = d with b != 0: a rotation by 45 degrees") {
        auto [values, vectors] = check(1.0, 0.5, 1.0);
        CHECK_THAT(values[0], WithinRel(0.5, 1e-15)); CHECK_THAT(values[1], WithinRel(1.5, 1e-15));
        CHECK_THAT(std::abs(vectors(0, 0)), WithinRel(std::sqrt(0.5), 1e-15));
        CHECK_THAT(std::abs(vectors(1, 0)), WithinRel(std::sqrt(0.5), 1e-15));
    }
    SECTION("a small off-diagonal entry beside a large diagonal entry") {
        // The small component of the eigenvector of the smaller eigenvalue, -b/(d - a) to first order, has the relative
        // accuracy of the entries; the order of a and d does not matter
        for (auto [a, d] : { std::make_pair(1.0, 1e10), std::make_pair(1e10, 1.0) }) {
            CAPTURE(a, d);
            auto [values, vectors] = check(a, 1e-20, d);
            const auto small = (a < d) ? 1 : 0;
            CHECK_THAT(vectors(small, 0) / vectors(1 - small, 0), WithinRel(-1e-20 / std::abs(d - a), 1e-15));
        }
    }
    SECTION("an entry that is NaN gives eigenvalues and eigenvectors that are NaN, and no exception") {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (auto [a, b, d] : { std::make_tuple(nan, 1.0, 2.0), std::make_tuple(1.0, nan, 2.0), std::make_tuple(1.0, 1.0, nan) }) {
            CAPTURE(a, b, d);
            auto [values, vectors] = ct::symmetric_eigen_2x2(a, b, d);
            CHECK(values.hasNaN());
            CHECK(vectors.hasNaN());
        }
    }
}

TEST_CASE("A trace that has left the critical curve ends with finite points", "[crit][critjac]")
{
    // Carbon dioxide + nitrogen from pure nitrogen with default options. The trace without polishing leaves the critical
    // curve and arrives at T = 2e-4 K, where the tangent is not finite. get_drhovec_dT_crit throws there, the tracer
    // catches the exception inside the step, and the trace ends after 155 points and not at max_step_count.
    const std::vector<std::string> names = { "CarbonDioxide", "Nitrogen" };
    const auto model = make_multifluid(names);
    const auto mf = build_multifluid_model(names, FLUIDDATAPATH);
    Eigen::ArrayXd rhovec0 = Eigen::ArrayXd::Zero(2);
    rhovec0[1] = 1.0 / mf.redfunc.vc[1];
    const TCABOptions opt;
    const auto pts = ct::trace_critical_arclength_binary(*model, mf.redfunc.Tc[1], rhovec0, std::nullopt, opt);
    REQUIRE(pts.size() > 1);
    CHECK(static_cast<int>(pts.size()) < opt.max_step_count);
    for (const auto& pt : pts) {
        CHECK(std::isfinite(pt.at("T / K").get<double>()));
        CHECK(std::isfinite(pt.at("rho0 / mol/m^3").get<double>()));
        CHECK(std::isfinite(pt.at("rho1 / mol/m^3").get<double>()));
    }
}

TEST_CASE("get_criticality_conditions_Jacobian next to the nonanalytic point of carbon dioxide", "[crit][critjac]")
{
    // The critical point of carbon dioxide + ethane at x(CO2) = 0.7 has tau - 1 = -3.8e-5 and delta - 1 = -0.018.
    // The fourth derivatives of the nonanalytic terms vary there on the scale of |tau - 1|.
    const auto& c = jac_cases().back();
    REQUIRE(c.label == "multifluid carbon dioxide + ethane");
    const Pin pin{ 291.17744225091707, 6222.574542036436, 2666.8176608727586, 805.7385902376561, -298.75980665264854 };
    auto base = ct::get_criticality_conditions_Jacobian(*c.model, pin.T, pin.rhovec());
    CHECK(scaled_residual(*c.model, pin.T, pin.rhovec(), base.C) < 1e-9);

    // The Taylor test with steps much shorter than |tau - 1|
    Eigen::Vector3d u(pin.T, pin.rho0, pin.rho1), d(0.37 * pin.T, -0.61 * pin.rho0, 0.83 * pin.rho1);
    std::vector<double> ratio;
    for (double eps : { 1e-5, 1e-6, 1e-7 }) {
        Eigen::Vector3d w = u + eps * d;
        Eigen::ArrayXd rv = (Eigen::ArrayXd(2) << w[1], w[2]).finished();
        auto pert = ct::get_criticality_conditions_Jacobian(*c.model, w[0], rv, base.ei.v0);
        double worst = 0;
        for (auto i = 0; i < 2; ++i) {
            double lin = eps * base.J.row(i).dot(d), scale = eps * (base.J.row(i).transpose().array() * d.array()).abs().sum();
            worst = std::max(worst, std::abs(pert.C[i] - base.C[i] - lin) / scale);
        }
        ratio.push_back(worst);
    }
    CAPTURE(ratio);
    for (auto i = 0; i + 1 < static_cast<int>(ratio.size()); ++i) {
        CHECK(ratio[i] / ratio[i + 1] >= 9.0);
    }

    // The tangent changes by up to 1.3e-9 (relative) for relative changes of T of up to 8e-13 at this state, and it
    // agrees with the multicomplex tangent to 9.1e-11. The tolerance is 100 times that.
    Eigen::ArrayXd t = ct::get_drhovec_dT_crit(*c.model, pin.T, pin.rhovec()).array();
    CHECK_THAT(t[0], WithinRel(pin.drho0dT, 1e-8));
    CHECK_THAT(t[1], WithinRel(pin.drho1dT, 1e-8));
}

TEST_CASE("get_criticality_conditions_Jacobian at the critical point of pure carbon dioxide returns the conditions of the pure fluid", "[crit][critjac]")
{
    // The conditions need derivatives of Psi^r of order 3 at most in the density of the pure fluid, and they vanish at
    // its critical point. The Jacobian there is not tested: with the nonanalytic terms its column for the concentration
    // of ethane is not the limit of that column at a small concentration of ethane.
    const auto& c = jac_cases().back();
    REQUIRE(c.label == "multifluid carbon dioxide + ethane");
    const auto mf = build_multifluid_model({ "CarbonDioxide", "Ethane" }, FLUIDDATAPATH);
    Eigen::ArrayXd rhovec = Eigen::ArrayXd::Zero(2);
    rhovec[0] = 1.0 / mf.redfunc.vc[0];
    auto cj = ct::get_criticality_conditions_Jacobian(*c.model, mf.redfunc.Tc[0], rhovec);
    CHECK(cj.at_infinite_dilution);
    CHECK(scaled_residual(*c.model, mf.redfunc.Tc[0], rhovec, cj.C) < 1e-9);
}
