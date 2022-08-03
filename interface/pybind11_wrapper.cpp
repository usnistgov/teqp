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

template<typename Model, int iT, int iD, typename Class>
void add_ig_deriv_impl(Class& cls) {
    using idx = TDXDerivatives<Model>;
    using RAX = Eigen::Ref<Eigen::ArrayXd>;
    if constexpr (iT == 0 && iD == 0){
        cls.def("get_Aig00", 
            [](const Model& m, const double T, const double rho, const RAX molefrac) { return AlphaCallWrapper<1, decltype(m)>(m).alpha(T, rho, molefrac); }, 
            py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert()
        );
    }
    else{
        const std::string fname = "get_Aig" + std::to_string(iT) + std::to_string(iD);
        cls.def(fname.c_str(), 
            [](const Model& m, const double T, const double rho, const RAX molefrac) { return idx::template get_Aigxy<iT, iD, ADBackends::autodiff>(m, T, rho, molefrac); }, 
            py::arg("T"), py::arg("rho"), py::arg("molefrac").noconvert()
        );
    }
}

template<typename Model, typename Class>
void add_ig_derivatives(py::module& m, Class& cls) {
    add_ig_deriv_impl<Model, 0, 0>(cls); add_ig_deriv_impl<Model, 0, 1>(cls); add_ig_deriv_impl<Model, 0, 2>(cls); add_ig_deriv_impl<Model, 0, 3>(cls); add_ig_deriv_impl<Model, 0, 4>(cls);
    add_ig_deriv_impl<Model, 1, 0>(cls); add_ig_deriv_impl<Model, 1, 1>(cls); add_ig_deriv_impl<Model, 1, 2>(cls); add_ig_deriv_impl<Model, 1, 3>(cls); add_ig_deriv_impl<Model, 1, 4>(cls);
    add_ig_deriv_impl<Model, 2, 0>(cls); add_ig_deriv_impl<Model, 2, 1>(cls); add_ig_deriv_impl<Model, 2, 2>(cls); add_ig_deriv_impl<Model, 2, 3>(cls); add_ig_deriv_impl<Model, 2, 4>(cls);
}

/// Instantiate "instances" of models (really wrapped Python versions of the models), and then attach all derivative methods
void init_teqp(py::module& m) {
    add_vdW(m);
    add_PCSAFT(m);
    add_CPA(m);
    add_multifluid(m);
    add_multifluid_mutant(m);
    add_cubics(m);

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
        ;

    // The options class for the finder of VLLE solutions from VLE tracing, not tied to a particular model
    py::class_<VLLEFinderOptions>(m, "VLLEFinderOptions")
        .def(py::init<>())
        .def_readwrite("max_steps", &VLLEFinderOptions::max_steps)
        .def_readwrite("rho_trivial_threshold", &VLLEFinderOptions::rho_trivial_threshold)
        ;

    py::enum_<VLE_return_code>(m, "VLE_return_code")
        .value("unset", VLE_return_code::unset)
        .value("xtol_satisfied", VLE_return_code::xtol_satisfied)
        .value("functol_satisfied", VLE_return_code::functol_satisfied)
        .value("maxiter_met", VLE_return_code::maxiter_met)
        .value("notfinite_step", VLE_return_code::notfinite_step)
        ;

    // The ideal gas Helmholtz energy class
    auto alphaig = py::class_<IdealHelmholtz>(m, "IdealHelmholtz").def(py::init<const nlohmann::json&>());
    add_ig_derivatives<IdealHelmholtz>(m, alphaig);

    // Some functions for timing overhead of interface
    m.def("___mysummer", [](const double &c, const Eigen::ArrayXd &x) { return c*x.sum(); });
    using RAX = Eigen::Ref<Eigen::ArrayXd>;
    using namespace pybind11::literals; // for "arg"_a
    m.def("___mysummerref", [](const double& c, const RAX x) { return c * x.sum(); }, "c"_a, "x"_a.noconvert());
    m.def("___myadder", [](const double& c, const double& d) { return c+d; });
}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    m.attr("__version__") = TEQPVERSION;
    init_teqp(m);
}