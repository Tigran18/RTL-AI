#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "network.hpp"

namespace py = pybind11;

// Expose add_bias function
void add_bias(py::array_t<float> input, float bias) {
    auto buf = input.mutable_unchecked<2>();
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        for (ssize_t j = 0; j < buf.shape(1); ++j) {
            buf(i,j) += bias;
        }
    }
}

PYBIND11_MODULE(py_network, m) {
    m.doc() = "GPU-based neural network module";

    // Bind add_bias
    m.def("add_bias", &add_bias, py::arg("input"), py::arg("bias"),
          "Add bias to each element of a 2D numpy array.");

    // Bind network::ActivationType enum
    py::enum_<network::ActivationType>(m, "ActivationType")
        .value("Sigmoid", network::ActivationType::Sigmoid)
        .value("ReLU", network::ActivationType::ReLU)
        .value("Tanh", network::ActivationType::Tanh)
        .export_values();

    // Bind network class as Network
    py::class_<network>(m, "Network")
        .def(py::init<std::vector<size_t>, std::vector<network::ActivationType>,
                      float, size_t, size_t, float, bool>(),
             py::arg("layer_sizes"),
             py::arg("activations"),
             py::arg("learning_rate") = 0.01f,
             py::arg("epochs") = 1000,
             py::arg("batch_size") = 32,
             py::arg("momentum") = 0.9f,
             py::arg("use_batch_norm") = false)
        .def("train", &network::train,
             py::arg("inputs"), py::arg("targets"),
             py::arg("val_inputs") = std::vector<std::vector<float>>(),
             py::arg("val_targets") = std::vector<std::vector<float>>())
        .def("evaluate", &network::evaluate,
             py::arg("inputs"), py::arg("targets"))
        .def("predict", &network::predict,
             py::arg("input"))
        .def("save_model", &network::save_model,
             py::arg("filename"))
        .def("load_model", &network::load_model,
             py::arg("filename"))
        .def("get_layer_sizes", &network::get_layer_sizes)
        .def("display_outputs", &network::display_outputs);
}
