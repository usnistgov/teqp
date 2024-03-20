#include <catch2/catch_test_macros.hpp>

#include "teqp/json_tools.hpp"

#include "stbrumme-hashing/sha256.h"
#include "stbrumme-hashing/sha256.cpp"

TEST_CASE("Test sha256", "[sha256]"){
    // Test values were generated from INCHI strings of all compounds in TDE, and hashlib.sha256(row['inchi_key'].encode('ascii')).hexdigest() in python
    SHA256 sha256;
    for (auto& el : teqp::load_a_JSON_file("../src/tests/resources/testvalues_sha256.json")){
        CHECK(sha256(el["input"]) == el["sha256"].get<std::string>());
    }
}
