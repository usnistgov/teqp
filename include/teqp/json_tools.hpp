#include "nlohmann/json.hpp"

namespace teqp{
    inline auto all_same_length(const nlohmann::json& j, const std::vector<std::string>& ks) {
        std::set<decltype(j[0].size())> lengths;
        for (auto k : ks) { lengths.insert(j.at(k).size()); }
        return lengths.size() == 1;
    }
}