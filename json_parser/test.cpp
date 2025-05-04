#include <iostream>
#include "json_parser.hpp"

int main() {
    JSON json("example.json");
    try{
        json.print_rtl();
    }
    catch(const std::exception& ex){
        std::cout<<ex.what()<<std::endl;
    }
    return 0;
}
