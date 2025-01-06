#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_GENERICSAFT
#include "teqp/models/saft/genericsaft.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_GENERICSAFT
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_genericSAFT(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(saft::genericsaft::GenericSAFT(spec));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_genericSAFT(const nlohmann::json &){
            throw teqp::NotImplementedError("The generic SAFT model has been disabled");
        }
#endif
    }
}
