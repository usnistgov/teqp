#include "pybind11_wrapper.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_mutant.hpp"
#include "teqp/derivs.hpp"

#include <type_traits>

#include "multifluid_shared.hpp"

void add_multifluid_mutant(py::module& m) {

    // A typedef for the base model
    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));
    
    // Wrap the function for generating a multifluid mutant
    m.def("build_multifluid_mutant", &build_multifluid_mutant<MultiFluid>);

    // The reducing function type and the departure function types are the same
    // as the base model
    using DepType = std::decay_t<decltype(MultiFluid::dep)>;

    // Typedef for modified mutant type
    using BIPmod = std::decay_t<MultiFluidAdapter<DepType, MultiFluid>>;

    // Define python wrapper of the mutant class
    auto wMFBIP = py::class_<BIPmod>(m, "MultiFluidMutant");
    
    add_derivatives<BIPmod>(m, wMFBIP);
    add_multifluid_methods<BIPmod>(wMFBIP);
    wMFBIP.def("get_alpharij", [](const BIPmod& c, const int i, const int j, const double &tau, const double &delta) { return c.depfunc.get_alpharij(i, j, tau, delta); });

    // Wrap the deprecated function for generating a multifluid mutant called an invariant
    m.def("build_multifluid_mutant_invariant", &build_multifluid_mutant<MultiFluid>);
}