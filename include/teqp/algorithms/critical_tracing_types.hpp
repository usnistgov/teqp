# pragma once

namespace teqp {

struct TCABOptions {
    double abs_err = 1.0e-6,
    rel_err = 1.0e-6,
    init_dt = 10, ///< The initial step size
    max_dt = 10000000000,
    T_tol = 1e-6, ///< The tolerance on temperature to indicate that it is converged
    init_c = 1.0; ///< The c parameter which controls the initial search direction for the first step. Choices are 1 or -1
    int small_T_count = 5; ///< How many small temperature steps indicates convergence
    int integration_order = 5; ///< The order of integration, either 1 for simple Euler or 5 for adaptive RK45
    int max_step_count = 1000; ///< Maximum number of steps allowed
    int skip_dircheck_count = 1; ///< Only start checking the direction dot product after this many steps
    bool polish = false; ///< If true, polish the solution at every step
    double polish_reltol_T = 0.01; ///< The maximum allowed change in temperature when polishing
    double polish_reltol_rho = 0.05; ///< The maximum allowed change in any molar concentration when polishing
    bool terminate_negative_density = true; ///< Stop the tracing if the density is negative
    bool calc_stability = false; ///< Calculate the local stability with the method of Deiters and Bell
    double stability_rel_drho = 0.001; ///< The relative size of the step (relative to the sum of the molar concentration vector) to be used when taking the step in the direction of \f$\sigma_1\f$ when assessing local stability
    int verbosity = 0; ///< The greater the verbosity, the more output you will get, especially about polishing failures
    bool polish_exception_on_fail = false; ///< If true, when polishing fails, throw an exception, otherwise, terminate tracing
    bool pure_endpoint_polish = false; ///< If true, if the last step crossed into negative concentrations, try to interpolate to find the pure fluid endpoint hiding in the data
};

struct EigenData {
    Eigen::ArrayXd v0, v1, eigenvalues;
    Eigen::MatrixXd eigenvectorscols;
};

}
