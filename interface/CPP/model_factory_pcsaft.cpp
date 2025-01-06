#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_PCSAFT
#include "teqp/models/pcsaft.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#ifndef DISABLE_PCSAFT
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFT(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(PCSAFT::PCSAFTfactory(spec));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFTPureGrossSadowski2001(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(PCSAFT::PCSAFTPureGrossSadowski2001(spec));
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFT(const nlohmann::json &){
            throw teqp::NotImplementedError("The PC-SAFT model has been disabled");
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFTPureGrossSadowski2001(const nlohmann::json &){
            throw teqp::NotImplementedError("The PC-SAFT model for pure fluids has been disabled");
        }
#endif
    }
}
