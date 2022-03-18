#pragma once 

template<typename Model, typename Wrapper>
void add_multifluid_methods(Wrapper &wMF){
    wMF.def("get_Tcvec", [](const Model& c) { return c.redfunc.Tc; })
       .def("get_vcvec", [](const Model& c) { return c.redfunc.vc; })
       .def("get_Tr", [](const Model& c, const Eigen::ArrayXd &molefrac) { return c.redfunc.get_Tr(molefrac); })
       .def("get_rhor", [](const Model& c, const Eigen::ArrayXd &molefrac) { return c.redfunc.get_rhor(molefrac); })
       .def("set_meta", [](Model& c, const std::string &s) { return c.set_meta(s); })
       .def("get_meta", [](const Model& c) { return c.get_meta(); });
}