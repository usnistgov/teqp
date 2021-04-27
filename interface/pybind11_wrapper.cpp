#define USE_AUTODIFF

#include "pybind11_json/pybind11_json.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "teqp/core.hpp"

#include "teqp/algorithms/critical_tracing.hpp"
#include "teqp/models/pcsaft.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/algorithms/VLE.hpp"

namespace py = pybind11;

//template<typename Model>
//void add_TDx_derivatives(py::module& m) {
//    using id = TDXDerivatives<Model, double, Eigen::Array<double, Eigen::Dynamic, 1> >;
//    //m.def("get_Ar00", &id::get_Ar00, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_Ar10", &id::get_Ar10<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_Ar01", &id::get_Ar01<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_Ar11", &id::get_Ar11<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_Ar02", &id::get_Ar02<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_Ar20", &id::get_Ar20<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac"));
//    m.def("get_neff", &id::get_neff<ADBackends::autodiff>, py::arg("model"), py::arg("T"), py::arg("rho"), py::arg("molefrac")); 
//}

template<typename Model, typename Wrapper>
void add_derivatives(py::module &m, Wrapper &cls) {
    using id = IsochoricDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1> >;
    m.def("get_Ar00", &id::get_Ar00, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Ar10", &id::get_Ar10, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_Psir", &id::get_Psir, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("get_pr", &id::get_pr, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("get_splus", &id::get_splus, py::arg("model"), py::arg("T"), py::arg("rho"));

    m.def("build_Psir_Hessian_autodiff", &id::build_Psir_Hessian_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));
    m.def("build_Psir_gradient_autodiff", &id::build_Psir_gradient_autodiff, py::arg("model"), py::arg("T"), py::arg("rho"));

    using vd = VirialDerivatives<Model, double, Eigen::Array<double,Eigen::Dynamic,1>>;
    m.def("get_B2vir", &vd::get_B2vir, py::arg("model"), py::arg("T"), py::arg("molefrac"));
    m.def("get_B12vir", &vd::get_B12vir, py::arg("model"), py::arg("T"), py::arg("molefrac"));

    //add_TDx_derivatives<Model>(m);

    using ct = CriticalTracing<Model, double, Eigen::Array<double, Eigen::Dynamic, 1>>;
    m.def("trace_critical_arclength_binary", &ct::trace_critical_arclength_binary);

    m.def("extrapolate_from_critical", &extrapolate_from_critical<Model, double>);
    m.def("pure_VLE_T", &pure_VLE_T<Model, double>);

    //cls.def("get_Ar01", [](const Model& m, const double T, const Eigen::ArrayXd& rhovec) { return id::get_Ar01(m, T, rhovec); });
    //cls.def("get_Ar10", [](const Model& m, const double T, const Eigen::ArrayXd& rhovec) { return id::get_Ar10(m, T, rhovec); });
    using tdx = TDXDerivatives<Model, double, Eigen::Array<double, Eigen::Dynamic, 1> >;
    cls.def("get_Ar00", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::get_Ar00(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar01", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar01<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar10", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar10<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar11", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar11<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar12", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar12<ADBackends::autodiff>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));

    cls.def("get_Ar01n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<1>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar02n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<2>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar03n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<3>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar04n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<4>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar05n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<5>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_Ar06n", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::template get_Ar0n<6>(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
    cls.def("get_neff", [](const Model& m, const double T, const double rho, const Eigen::ArrayXd& molefrac) { return tdx::get_neff(m, T, rho, molefrac); }, py::arg("T"), py::arg("rho"), py::arg("molefrac"));
}

/// Instantiate "instances" of models (really wrapped Python versions of the models), and then attach all derivative methods
void init_teqp(py::module& m) {

    using vdWEOSd = vdWEOS<double>;
    auto wvdW = py::class_<vdWEOSd>(m, "vdWEOS")
        .def(py::init<const std::valarray<double>&, const std::valarray<double>&>(),py::arg("Tcrit"), py::arg("pcrit"))
        ;
    add_derivatives<vdWEOSd>(m, wvdW);
    
    auto wvdW1 = py::class_<vdWEOS1>(m, "vdWEOS1")
        .def(py::init<const double&, const double&>(), py::arg("a"), py::arg("b"))    
        ;
    add_derivatives<vdWEOS1>(m, wvdW1);

    py::class_<SAFTCoeffs>(m, "SAFTCoeffs")
        .def(py::init<>())
        .def_readwrite("name", &SAFTCoeffs::name)
        .def_readwrite("m", &SAFTCoeffs::m)
        .def_readwrite("sigma_Angstrom", &SAFTCoeffs::sigma_Angstrom)
        .def_readwrite("epsilon_over_k", &SAFTCoeffs::epsilon_over_k)
        .def_readwrite("BibTeXKey", &SAFTCoeffs::BibTeXKey)
        ;

    auto wPCSAFT = py::class_<PCSAFTMixture>(m, "PCSAFTEOS")
        .def(py::init<const std::vector<std::string> &>(), py::arg("names"))
        .def(py::init<const std::vector<SAFTCoeffs> &>(), py::arg("coeffs"))
        .def("print_info", &PCSAFTMixture::print_info)
        .def("max_rhoN", &PCSAFTMixture::max_rhoN<Eigen::ArrayXd>)
        .def("get_m", &PCSAFTMixture::get_m)
        .def("get_sigma_Angstrom", &PCSAFTMixture::get_sigma_Angstrom)
        .def("get_epsilon_over_k_K", &PCSAFTMixture::get_epsilon_over_k_K)
        ;
    add_derivatives<PCSAFTMixture>(m, wPCSAFT);

    // Multifluid model
    m.def("build_multifluid_model", &build_multifluid_model);
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"",""},"",""));
    using idMF = IsochoricDerivatives<MultiFluid, double, Eigen::Array<double, Eigen::Dynamic, 1> >;
    auto wMF = py::class_<MultiFluid>(m, "MultiFluid")
        .def("get_Tcvec", [](const MultiFluid& c) { return c.redfunc.Tc; })
        .def("get_vcvec", [](const MultiFluid& c) { return c.redfunc.vc; })
        ;
    add_derivatives<MultiFluid>(m, wMF);

    // for timing testing
    m.def("mysummer", [](const double &c, const Eigen::ArrayXd &x) { return c*x.sum(); });
    m.def("myadder", [](const double& c, const double& d) { return c+d; });

}

PYBIND11_MODULE(teqp, m) {
    m.doc() = "TEQP: Templated Equation of State Package";
    init_teqp(m);
}