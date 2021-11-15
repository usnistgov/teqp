#pragma once

//template <typename T>
//concept has_meta = requires(T const& m) {
//    { m.get_meta() };
//};
namespace teqp { 
template<typename... Args>
class ModelContainer {
public:
    using mid = std::size_t;
    using varModels = std::variant<Args...>;
private:
    mid last_id = 0;
    std::map<mid, varModels> modcoll;
public:
    template <typename T> const auto& get_ref(mid id) { return std::get<T>(get_model(id)); }

    auto size() const { return modcoll.size(); }
    auto new_id() {
        last_id++;
        return last_id;
    }
    template<typename Instance>
    auto add_model(Instance&& instance) {
        auto uid = new_id();
        modcoll.emplace(uid, std::move(instance));
        return uid;
    }

    const varModels& get_model(mid id) const {
        return modcoll.at(id);
    }

    template <typename Function>
    auto caller(const mid& mid, const Function &f) const {
        return std::visit([&](auto& model) { return f(model); }, get_model(mid));
    }

    //nlohmann::json get_meta(const mid& mid) const {
    //    const auto& modvar = get_model(mid);
    //    nlohmann::json result;
    //    std::visit([&](auto&& model) {
    //        if constexpr (has_meta<decltype(model)>) { result = model.get_meta(); }
    //    }, modvar);
    //    return result;
    //}
};
}; // namespace teqp