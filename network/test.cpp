#include <iostream>
#include "network.hpp"

int main(){
    network m_network(3, {10, 15, 8});
    std::cout<<m_network.get_layers()<<std::endl;
    m_network.get_neurons();
    return 0;
}