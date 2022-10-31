#include "pybind11_wrapper.hpp"

#include "teqpversion.hpp"
#include "teqp/ideal_eosterms.hpp"

namespace py = pybind11;

// The implementation of each prototype are in separate files to move the compilation into 
// multiple compilation units so that multiple processors can be used
// at the same time to carry out the compilation
// 
// This speeds up things a lot on linux, but not much in MSVC
void add_vdW(py::module &m);
void add_PCSAFT(py::module& m);
void add_CPA(py::module& m);
void add_multifluid(py::module& m);
void add_multifluid_mutant(py::module& m);
void add_cubics(py::module& m);
void add_model_potentials(py::module& m);

template<typename Model, int iT, int iD, typename Class>
void add_ig_deriv_impl(Class& cls) {
    using idx = TDXDerivatives<Model>;
    using RAX = Eigen::Ref<const Eigen::ArrayXd>;
    const std::string fname = "get_Aig" + std::to_string(iT) + std::to_string(iD);
    cls.def(fname.c_str(),
        [](const Model& m, const double T, const double rho, const RAX molefrac) { return idx::template get_Aigxy<iT, iD, ADBackends::autodiff>(m, T, rho, molefrac); },
        py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert()
    );
}

template<typename Model, typename Class>
void add_ig_derivatives(py::module& m, Class& cls) {
    add_ig_deriv_impl<Model, 0, 0>(cls); add_ig_deriv_impl<Model, 0, 1>(cls); add_ig_deriv_impl<Model, 0, 2>(cls); add_ig_deriv_impl<Model, 0, 3>(cls); add_ig_deriv_impl<Model, 0, 4>(cls);
    add_ig_deriv_impl<Model, 1, 0>(cls); add_ig_deriv_impl<Model, 1, 1>(cls); add_ig_deriv_impl<Model, 1, 2>(cls); add_ig_deriv_impl<Model, 1, 3>(cls); add_ig_deriv_impl<Model, 1, 4>(cls);
    add_ig_deriv_impl<Model, 2, 0>(cls); add_ig_deriv_impl<Model, 2, 1>(cls); add_ig_deriv_impl<Model, 2, 2>(cls); add_ig_deriv_impl<Model, 2, 3>(cls); add_ig_deriv_impl<Model, 2, 4>(cls);
}

/** 
A redirector factory function. It is needed to at runtime select the desired attribute (method)
that is available on the model and call it. Runtime overload resolution turns out
to be much faster than overload resolution from a free function w/ pybind11. 

First the function selects the desired attribute at runtime and forwards all 
positional and keyword arguments to the method of interest.  The name of the method is captured by value
inside the lambda
*/
inline auto call_method_factory(py::module &m, const std::string& attribute) {
    
    auto f = [attribute](const py::object& model, const py::args& args, const py::kwargs& kwargs) {
        std::string warning_string = ("Calling the top-level function " + attribute + " is deprecated" +
            " and much slower than calling the same-named method of the model instance");
        PyErr_WarnEx(PyExc_DeprecationWarning, warning_string.c_str(), 1);
        return model.attr(attribute.c_str())(*args, **kwargs);
    };
    m.def(attribute.c_str(), f);
}

/// Instantiate "instances" of models (really wrapped Python versions of the models), and then attach all derivative methods
void init_teqp(py::module& m) {

    // The options class for critical tracer, not tied to a particular model
    py::class_<TCABOptions>(m, "TCABOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &TCABOptions::abs_err)
        .def_readwrite("rel_err", &TCABOptions::rel_err)
        .def_readwrite("init_dt", &TCABOptions::init_dt)
        .def_readwrite("init_c", &TCABOptions::init_c)
        .def_readwrite("max_dt", &TCABOptions::max_dt)
        .def_readwrite("T_tol", &TCABOptions::T_tol)
        .def_readwrite("small_T_count", &TCABOptions::small_T_count)
        .def_readwrite("max_step_count", &TCABOptions::max_step_count)
        .def_readwrite("skip_dircheck_count", &TCABOptions::skip_dircheck_count)
        .def_readwrite("integration_order", &TCABOptions::integration_order)
        .def_readwrite("calc_stability", &TCABOptions::calc_stability)
        .def_readwrite("stability_rel_drho", &TCABOptions::stability_rel_drho)
        .def_readwrite("verbosity", &TCABOptions::verbosity)
        .def_readwrite("polish", &TCABOptions::polish)
        .def_readwrite("polish_reltol_rho", &TCABOptions::polish_reltol_rho)
        .def_readwrite("polish_reltol_T", &TCABOptions::polish_reltol_T)
        .def_readwrite("pure_endpoint_polish", &TCABOptions::pure_endpoint_polish)
        .def_readwrite("polish_exception_on_fail", &TCABOptions::polish_exception_on_fail)
        ;

    // The options class for isotherm tracer, not tied to a particular model
    py::class_<TVLEOptions>(m, "TVLEOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &TVLEOptions::abs_err)
        .def_readwrite("rel_err", &TVLEOptions::rel_err)
        .def_readwrite("init_dt", &TVLEOptions::init_dt)
        .def_readwrite("init_c", &TVLEOptions::init_c)
        .def_readwrite("max_dt", &TVLEOptions::max_dt)
        .def_readwrite("max_steps", &TVLEOptions::max_steps)
        .def_readwrite("integration_order", &TVLEOptions::integration_order)
        .def_readwrite("polish", &TVLEOptions::polish)
        .def_readwrite("calc_criticality", &TVLEOptions::calc_criticality)
        .def_readwrite("terminate_unstable", &TVLEOptions::terminate_unstable)
        ;

    // The options class for isobar tracer, not tied to a particular model
    py::class_<PVLEOptions>(m, "PVLEOptions")
        .def(py::init<>())
        .def_readwrite("abs_err", &PVLEOptions::abs_err)
        .def_readwrite("rel_err", &PVLEOptions::rel_err)
        .def_readwrite("init_dt", &PVLEOptions::init_dt)
        .def_readwrite("init_c", &PVLEOptions::init_c)
        .def_readwrite("max_dt", &PVLEOptions::max_dt)
        .def_readwrite("max_steps", &PVLEOptions::max_steps)
        .def_readwrite("integration_order", &PVLEOptions::integration_order)
        .def_readwrite("polish", &PVLEOptions::polish)
        .def_readwrite("calc_criticality", &PVLEOptions::calc_criticality)
        .def_readwrite("terminate_unstable", &PVLEOptions::terminate_unstable)
        ;

    // The options class for the finder of VLLE solutions from VLE tracing, not tied to a particular model
    py::class_<VLLEFinderOptions>(m, "VLLEFinderOptions")
        .def(py::init<>())
        .def_readwrite("max_steps", &VLLEFinderOptions::max_steps)
        .def_readwrite("rho_trivial_threshold", &VLLEFinderOptions::rho_trivial_threshold)
        ;

    py::class_<MixVLETpFlags>(m, "MixVLETpFlags")
        .def(py::init<>())
        .def_readwrite("atol", &MixVLETpFlags::atol)
        .def_readwrite("reltol", &MixVLETpFlags::reltol)
        .def_readwrite("axtol", &MixVLETpFlags::axtol)
        .def_readwrite("relxtol", &MixVLETpFlags::relxtol)
        .def_readwrite("maxiter", &MixVLETpFlags::maxiter)
        ;

    py::class_<MixVLEpxFlags>(m, "MixVLEpxFlags")
        .def(py::init<>())
        .def_readwrite("atol", &MixVLEpxFlags::atol)
        .def_readwrite("reltol", &MixVLEpxFlags::reltol)
        .def_readwrite("axtol", &MixVLEpxFlags::axtol)
        .def_readwrite("relxtol", &MixVLEpxFlags::relxtol)
        .def_readwrite("maxiter", &MixVLEpxFlags::maxiter)
        ;
    
    // The Jacobian and value matrices for Newton-Raphson
    py::class_<IterationMatrices>(m, "IterationMatrices")
        .def(py::init<>())
        .def_readonly("J", &IterationMatrices::J)
        .def_readonly("v", &IterationMatrices::v)
        .def_readonly("vars", &IterationMatrices::vars)
        ;

    py::enum_<VLE_return_code>(m, "VLE_return_code")
        .value("unset", VLE_return_code::unset)
        .value("xtol_satisfied", VLE_return_code::xtol_satisfied)
        .value("functol_satisfied", VLE_return_code::functol_satisfied)
        .value("maxiter_met", VLE_return_code::maxiter_met)
        .value("maxfev_met", VLE_return_code::maxfev_met)
        .value("notfinite_step", VLE_return_code::notfinite_step)
        ;

    py::class_<MixVLEReturn>(m, "MixVLEReturn")
        .def(py::init<>())
        .def_readonly("success", &MixVLEReturn::success)
        .def_readonly("message", &MixVLEReturn::message)
        .def_readonly("rhovecL", &MixVLEReturn::rhovecL)
        .def_readonly("rhovecV", &MixVLEReturn::rhovecV)
        .def_readonly("return_code", &MixVLEReturn::return_code)
        .def_readonly("num_iter", &MixVLEReturn::num_iter)
        .def_readonly("T", &MixVLEReturn::T)
        .def_readonly("num_fev", &MixVLEReturn::num_fev)
        .def_readonly("r", &MixVLEReturn::r)
        .def_readonly("initial_r", &MixVLEReturn::initial_r)
        ;

    // The ideal gas Helmholtz energy class
    auto alphaig = py::class_<IdealHelmholtz>(m, "IdealHelmholtz").def(py::init<const nlohmann::json&>());
    alphaig.def_static("convert_CoolProp_format", [](const std::string &path, int index){return convert_CoolProp_idealgas(path, index);});
    add_ig_derivatives<IdealHelmholtz>(m, alphaig);
    alphaig.def("get_deriv_mat2", [](const IdealHelmholtz &ig, double T, double rho, const Eigen::ArrayXd& z){return DerivativeHolderSquare<2, AlphaWrapperOption::idealgas>(ig, T, rho, z).derivs;});

    add_vdW(m);
    add_PCSAFT(m);
    add_CPA(m);
    add_multifluid(m);
    add_multifluid_mutant(m);
    add_cubics(m);
    add_model_potentials(m);

    call_method_factory(m, "get_Ar00iso");
    call_method_factory(m, "get_Ar10iso");
    call_method_factory(m, "get_Psiriso"),

    call_method_factory(m, "get_splus");
    call_method_factory(m, "get_pr");
    call_method_factory(m, "get_B2vir");
    call_method_factory(m, "get_B12vir");
    
    call_method_factory(m, "pure_VLE_T");
    call_method_factory(m, "extrapolate_from_critical");

    call_method_factory(m, "build_Psir_Hessian_autodiff");
    call_method_factory(m, "build_Psi_Hessian_autodiff");
    call_method_factory(m, "build_Psir_gradient_autodiff");
    call_method_factory(m, "build_d2PsirdTdrhoi_autodiff");
    call_method_factory(m, "get_chempotVLE_autodiff");
    call_method_factory(m, "get_dchempotdT_autodiff");
    call_method_factory(m, "get_fugacity_coefficients");
    call_method_factory(m, "get_partial_molar_volumes");

    call_method_factory(m, "trace_critical_arclength_binary");
    call_method_factory(m, "get_criticality_conditions");
    call_method_factory(m, "eigen_problem");
    call_method_factory(m, "get_minimum_eigenvalue_Psi_Hessian");
    call_method_factory(m, "get_drhovec_dT_crit");

    call_method_factory(m, "get_pure_critical_conditions_Jacobian");
    call_method_factory(m, "solve_pure_critical");
    call_method_factory(m, "mix_VLE_Tx");
    call_method_factory(m, "mixture_VLE_px");

    call_method_factory(m, "get_drhovecdp_Tsat");
    call_method_factory(m, "trace_VLE_isotherm_binary");
    call_method_factory(m, "get_drhovecdT_psat");
    call_method_factory(m, "trace_VLE_isobar_binary");
    call_method_factory(m, "get_dpsat_dTsat_isopleth");

    call_method_factory(m, "mix_VLLE_T");
    call_method_factory(m, "find_VLLE_T_binary");

    // Some functions for timing overhead of interface
    m.def("___mysummer", [](const double &c, const Eigen::ArrayXd &x) { return c*x.sum(); });
    using RAX = Eigen::Ref<const Eigen::ArrayXd>;
    using namespace pybind11::literals; // for "arg"_a
    m.def("___mysummerref", [](const double& c, const RAX x) { return c * x.sum(); }, "c"_a, "x"_a.noconvert());
    m.def("___myadder", [](const double& c, const double& d) { return c+d; });
}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    m.attr("__version__") = TEQPVERSION;
    init_teqp(m);
}
