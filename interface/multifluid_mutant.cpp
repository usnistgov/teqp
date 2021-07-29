#include "pybind11_wrapper.hpp"

#include "teqp/models/multifluid.hpp"
#include "teqp/derivs.hpp"

void add_multifluid_mutant(py::module& m) {

    using MultiFluid = decltype(build_multifluid_model(std::vector<std::string>{"", ""}, "", ""));
    {
        m.def("build_multifluid_mutant", &build_multifluid_mutant<MultiFluid>);
        using RedType = std::decay_t<decltype(MultiFluid::redfunc)>;
        using DepType = std::decay_t<decltype(MultiFluid::dep)>;
        using BIPmod = MultiFluidAdapter<RedType, DepType, MultiFluid>;
        auto wMFBIP = py::class_<BIPmod>(m, "MultiFluidMutant");
        add_derivatives<BIPmod>(m, wMFBIP);
    }
}