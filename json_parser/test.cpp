#include <iostream>
#include "json_parser.hpp"

int main() {
    std::string str = "example.json";
    JSON obj(str);
    obj.add("key", std::vector<JSONvalues_for_vector>{
        std::string("hello"),
        42,
        true,
        JSON("{\"nestedKey\":\"nestedValue\"}", true)
    });
    obj.print(2);
    try{
        std::cout<<"Value at \"key\" is: "<<obj["key"]<<std::endl;
        obj.add("key", "string");
        std::cout<<"Value at \"key\" after change is: "<<obj["key"]<<std::endl;
    }
    catch(const std::exception& ex){
        std::cout<<ex.what()<<std::endl;
    }
    return 0;
}
