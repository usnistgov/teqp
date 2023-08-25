#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include "teqp/json_tools.hpp"

static nlohmann::json person_schema = R"(
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "A person",
    "properties": {
        "name": {
            "description": "Name",
            "type": "string"
        },
        "age": {
            "description": "Age of the person",
            "type": "number",
            "minimum": 2,
            "maximum": 200
        }
    },
    "required": [
                 "name",
                 "age"
                 ],
    "type": "object"
}
)"_json;

static json bad_person = {{"age", 42}};
static json good_person = {{"name", "Albert"}, {"age", 42}};

TEST_CASE("Test JSON validation with json-schema-validator", "[JSON]")
{
    json_validator validator; // create validator
    validator.set_root_schema(person_schema); // insert root-schema
    
    CHECK_THROWS(validator.validate(bad_person));
    CHECK_NOTHROW(validator.validate(good_person));
    
    teqp::JSONValidator jv(person_schema);
    CHECK(!jv.is_valid(bad_person));
    CHECK(jv.is_valid(good_person));
    
//    for (auto err : jv.get_validation_errors(bad_person)){
//        std::cout << err << std::endl;
//    }
}
