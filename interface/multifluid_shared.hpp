#pragma once 

#include "teqp/exceptions.hpp"

template<typename Model, typename Wrapper>
void add_multifluid_methods(Wrapper &wMF){
    wMF.def("get_Tcvec", [](const Model& c) { return c.redfunc.Tc; })
       .def("get_vcvec", [](const Model& c) { return c.redfunc.vc; })
       .def("get_Tr", [](const Model& c, const Eigen::ArrayXd &molefrac) { return c.redfunc.get_Tr(molefrac); })
       .def("get_rhor", [](const Model& c, const Eigen::ArrayXd &molefrac) { return c.redfunc.get_rhor(molefrac); })
       .def("set_meta", [](Model& c, const std::string &s) { return c.set_meta(s); })
       .def("get_meta", [](const Model& c) { return c.get_meta(); })
       .def("build_ancillaries", [](const Model& c) { 
           if (c.redfunc.Tc.size() != 1) {
               throw teqp::InvalidArgument("Can only build ancillaries for pure fluids");
           }
           auto jancillaries = nlohmann::json::parse(c.get_meta()).at("pures")[0].at("ANCILLARIES");
           return teqp::MultiFluidVLEAncillaries(jancillaries);
           })
       ;
}