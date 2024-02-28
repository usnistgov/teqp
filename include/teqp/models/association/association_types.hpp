#pragma once

namespace teqp {
namespace association{

enum class association_classes {not_set, a1A, a2B, a3B, a4C, not_associating};

inline auto get_association_classes(const std::string& s) {
    if (s == "1A") { return association_classes::a1A; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "3B") { return association_classes::a3B; }
    else if (s == "4C") { return association_classes::a4C; }
    else {
        throw std::invalid_argument("bad association flag:" + s);
    }
}

enum class radial_dist { CS, KG };

inline auto get_radial_dist(const std::string& s) {
    if (s == "CS") { return radial_dist::CS; }
    else if (s == "KG") { return radial_dist::KG; }
    else {
        throw std::invalid_argument("bad association flag:" + s);
    }
}

}
}
