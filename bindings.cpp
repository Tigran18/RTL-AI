#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "/home/tigranda1809/RTL AI/network/network.hpp"
#include "/home/tigranda1809/RTL AI/json_parser/json_parser.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ai_module, m) {
    py::class_<network>(m, "Network")
        .def(py::init<std::vector<size_t>, double, size_t>())
        .def("train", &network::train)
        .def("predict", &network::predict)
        .def("save_model", &network::save_model) 
        .def("load_model", &network::load_model) 
        .def("display_outputs", &network::display_outputs);

    py::class_<JSON>(m, "JSON")
        .def(py::init<const std::string&, bool>(), py::arg("input"), py::arg("isRawString") = true)
        .def("print", &JSON::print)
        .def("print_rtl", &JSON::print_rtl);
}
