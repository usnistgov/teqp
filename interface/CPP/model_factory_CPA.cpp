#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_CPA
#include "teqp/models/CPA.hpp"
#endif

namespace teqp{
    namespace cppinterface{
        using teqp::cppinterface::adapter::make_owned;
#ifndef DISABLE_CPA
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_CPA(const nlohmann::json &spec){
            return make_owned(CPA::CPAfactory(spec));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_CPA(const nlohmann::json &){
            throw teqp::NotImplementedError("The CPA model has been disabled");
        }
#endif
    }
}
