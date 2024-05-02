#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"
#include "teqp/models/pcsaft.hpp"

namespace teqp{
    namespace cppinterface{
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFT(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(PCSAFT::PCSAFTfactory(spec));
        }
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_PCSAFTPureGrossSadowski2001(const nlohmann::json &spec){
            return teqp::cppinterface::adapter::make_owned(PCSAFT::PCSAFTPureGrossSadowski2001(spec));
        }
    }
}
