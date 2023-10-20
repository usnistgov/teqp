#pragma once

namespace teqp{

struct TVLEOptions {
    double init_dt = 1e-5, abs_err = 1e-8, rel_err = 1e-8, max_dt = 100000, init_c = 1.0, p_termination = 1e15, crit_termination = 1e-12;
    int max_steps = 1000, integration_order = 5, revision = 1, verbosity = 0;
    bool polish = true, polish_exception_on_fail = false;
    double polish_reltol_rho = 0.05;
    bool calc_criticality = false;
    bool terminate_unstable = false;
};

struct PVLEOptions {
    double init_dt = 1e-5, abs_err = 1e-8, rel_err = 1e-8, max_dt = 100000, init_c = 1.0, crit_termination = 1e-12;
    int max_steps = 1000, integration_order = 5, verbosity = 0;
    bool polish = true, polish_exception_on_fail = false;
    double polish_reltol_rho = 0.05;
    bool calc_criticality = false;
    bool terminate_unstable = false;
};

struct MixVLEpxFlags {
    double atol = 1e-10,
    reltol = 1e-10,
    axtol = 1e-10,
    relxtol = 1e-10;
    int maxiter = 10;
};

struct MixVLETpFlags {
    double atol = 1e-10,
    reltol = 1e-10,
    axtol = 1e-10,
    relxtol = 1e-10,
    relaxation = 1.0;
    int maxiter = 10;
};

enum class VLE_return_code { unset, xtol_satisfied, functol_satisfied, maxfev_met, maxiter_met, notfinite_step };

struct MixVLEReturn {
    bool success = false;
    std::string message = "";
    Eigen::ArrayXd rhovecL, rhovecV;
    VLE_return_code return_code;
    int num_iter=-1, num_fev=-1;
    double T=-1;
    Eigen::ArrayXd r, initial_r;
};

}
