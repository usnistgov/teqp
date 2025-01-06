#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_SOFTSAFT
#include "teqp/models/saft/softsaft.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_SOFTSAFT
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SOFTSAFT(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(saft::softsaft::SoftSAFT(spec));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SOFTSAFT(const nlohmann::json &){
            throw teqp::NotImplementedError("The soft SAFT model has been disabled");
        }
#endif
    }
}
