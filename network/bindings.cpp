#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "network.hpp"

namespace py = pybind11;

PYBIND11_MODULE(network_lib, m) {
    m.doc() = "Python bindings for CUDA-based neural network library";

    py::enum_<network::ActivationType>(m, "ActivationType", py::arithmetic(),
                                      "Enum for activation functions")
        .value("Sigmoid", network::ActivationType::Sigmoid, "Sigmoid activation")
        .value("ReLU", network::ActivationType::ReLU, "ReLU activation")
        .value("Tanh", network::ActivationType::Tanh, "Tanh activation")
        .export_values();

    py::class_<network>(m, "Network", "Neural network with CUDA acceleration")
        .def(py::init<std::vector<size_t>,
                      std::vector<network::ActivationType>,
                      float, size_t, size_t, float, bool>(),
             py::arg("layer_sizes"),
             py::arg("activations"),
             py::arg("learning_rate") = 0.01f,
             py::arg("epochs") = 1000,
             py::arg("batch_size") = 32,
             py::arg("momentum") = 0.9f,
             py::arg("use_batch_norm") = false,
             "Initialize a neural network with specified layer sizes and parameters")
        .def("train", &network::train,
             py::arg("inputs"),
             py::arg("targets"),
             py::arg("val_inputs") = std::vector<std::vector<float>>{},
             py::arg("val_targets") = std::vector<std::vector<float>>{},
             "Train the network with input and target data")
        .def("evaluate", &network::evaluate,
             py::arg("inputs"),
             py::arg("targets"),
             "Evaluate the network on input and target data, returning MSE")
        .def("predict", &network::predict,
             py::arg("input"),
             "Predict output for a single input vector")
        .def("save_model", &network::save_model,
             py::arg("filename"),
             "Save the trained model to a file")
        .def("load_model", &network::load_model,
             py::arg("filename"),
             "Load a model from a file")
        .def("get_layer_sizes", &network::get_layer_sizes,
             py::return_value_policy::reference_internal,
             "Get the number of neurons in each layer")
        .def("display_outputs", &network::display_outputs,
             "Display the outputs of each layer (for debugging)");
}