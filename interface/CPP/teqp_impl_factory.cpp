#include "teqpcpp.cpp"

using namespace teqp;
using namespace teqp::cppinterface;

std::shared_ptr<AbstractModel> teqp::cppinterface::make_model(const nlohmann::json& j) {
    return std::make_shared<ModelImplementer>(build_model(j));
}
std::shared_ptr<AbstractModel> teqp::cppinterface::make_multifluid_model(const std::vector<std::string>& components, const std::string& coolprop_root, const std::string& BIPcollectionpath, const nlohmann::json& flags, const std::string& departurepath) {
    return std::make_shared<ModelImplementer>(build_multifluid_model(components, coolprop_root, BIPcollectionpath, flags, departurepath));
}

std::shared_ptr<AbstractModel> teqp::cppinterface::emplace_model(AllowedModels&& model){
    return std::make_shared<ModelImplementer>(std::move(model));
}
