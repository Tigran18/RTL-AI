#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "/home/tigranda1809/RTL AI/network/network.hpp"
#include "/home/tigranda1809/RTL AI/json_parser/json_parser.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ai_module, m) {
    py::class_<network>(m, "Network")
    .def(py::init<
        std::vector<size_t>,
        std::vector<network::ActivationType>,
        double, size_t, size_t, double>())
        .def("train", &network::train)
        .def("predict", &network::predict)
        .def("save_model", &network::save_model) 
        .def("load_model", &network::load_model) 
        .def("display_outputs", &network::display_outputs);

    py::class_<JSON>(m, "JSON")
        .def(py::init<const std::string&, bool>(), py::arg("input"), py::arg("isRawString") = true)
        .def("print", &JSON::print)
        .def("print_rtl", &JSON::print_rtl);
    py::enum_<network::ActivationType>(m, "ActivationType")
        .value("Sigmoid", network::ActivationType::Sigmoid)
        .value("ReLU", network::ActivationType::ReLU)
        .value("Tanh", network::ActivationType::Tanh);

}
