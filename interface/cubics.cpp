#include "pybind11_wrapper.hpp"

#include "teqp/models/cubics.hpp"

void add_cubics(py::module& m) {

    using va = std::valarray<double>;
    
    using PR = decltype(canonical_PR(va{}, va{}, va{}));
    auto wPR = py::class_<PR>(m, "PengRobinson");
    add_derivatives<PR>(m, wPR);

    using SRK = decltype(canonical_SRK(va{}, va{}, va{}));
    auto wSRK = py::class_<SRK>(m, "SoaveRedlichKwong");
    add_derivatives<SRK>(m, wSRK);
}