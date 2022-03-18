#include "pybind11_wrapper.hpp"

#include "teqp/models/multifluid.hpp"
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
    using RedType = std::decay_t<decltype(MultiFluid::redfunc)>;
    using DepType = std::decay_t<decltype(MultiFluid::dep)>;

    // Typedef for modified mutant type
    using BIPmod = std::decay_t<MultiFluidAdapter<RedType, DepType, MultiFluid>>;

    // Define python wrapper of the mutant class
    auto wMFBIP = py::class_<BIPmod>(m, "MultiFluidMutant");
    
    add_derivatives<BIPmod>(m, wMFBIP);
    add_multifluid_methods<BIPmod>(wMFBIP);
}

void add_multifluid_mutant_invariant(py::module& m) {

    // A typedef for the base model from which we steal the pure fluids
    using MultiFluid = std::decay_t<decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""))>;

    // Wrap the function for generating a multifluid mutant
    m.def("build_multifluid_mutant_invariant", &build_multifluid_mutant_invariant<MultiFluid>);

    // Typedef for mutant with the invariant reducing function
    using Mutant = std::invoke_result_t<decltype(build_multifluid_mutant_invariant<MultiFluid>), MultiFluid&, nlohmann::json&>;

    // Define python wrapper of the mutant class
    auto wMutant = py::class_<Mutant>(m, "MultiFluidMutantInvariant");

    add_derivatives<Mutant>(m, wMutant);
    add_multifluid_methods<Mutant>(wMutant);
}