// python_bindings/py_network.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "network.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstddef> // optional: for ptrdiff_t if needed

namespace py = pybind11;

// CPU implementation: add bias to every element of a float32 NumPy array (in-place)
void add_bias(py::array_t<float, py::array::c_style | py::array::forcecast> arr, float bias) {
    py::buffer_info info = arr.request();
    if (info.ptr == nullptr) return;

    // Compute total number of elements (flatten)
    size_t N = 1;
    for (int d = 0; d < info.ndim; ++d) {
        N *= static_cast<size_t>(info.shape[d]);
    }

    float* ptr = static_cast<float*>(info.ptr);
    for (size_t i = 0; i < N; ++i) {
        ptr[i] += bias;
    }
}

// Check CUDA availability
bool cuda_available() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) return false;
    return deviceCount > 0;
}

PYBIND11_MODULE(py_network, m) {
    m.doc() = "pybind11 bindings for GPU-based neural network";

    // Utilities
    m.def("add_bias", &add_bias, py::arg("array"), py::arg("bias"),
          "Add a scalar bias to every element of a NumPy float32 array (in-place).");
    m.def("cuda_available", &cuda_available, "Return true if CUDA device(s) are available.");

    // Activation enum
    py::enum_<network::ActivationType>(m, "ActivationType")
        .value("Sigmoid", network::ActivationType::Sigmoid)
        .value("ReLU",    network::ActivationType::ReLU)
        .value("Tanh",    network::ActivationType::Tanh)
        .export_values();

    // Network class binding
    py::class_<network>(m, "Network")
        .def(py::init<
              std::vector<size_t>,
              std::vector<network::ActivationType>,
              float, size_t, size_t, float, bool>(),
             py::arg("number_of_neurons_per_layer"),
             py::arg("activations"),
             py::arg("learning_rate") = 0.01f,
             py::arg("epochs") = 1000,
             py::arg("batch_size") = 32,
             py::arg("momentum") = 0.9f,
             py::arg("use_batch_norm") = false,
             "Construct a Network")
        .def("train", &network::train,
             py::arg("inputs"), py::arg("targets"),
             py::arg("val_inputs") = std::vector<std::vector<float>>{},
             py::arg("val_targets") = std::vector<std::vector<float>>{},
             "Train the network.")
        .def("evaluate", &network::evaluate,
             py::arg("inputs"), py::arg("targets"),
             "Compute MSE on dataset.")
        .def("predict", &network::predict,
             py::arg("input"),
             "Predict single input (returns Python list).")
        .def("display_outputs", &network::display_outputs,
             "Print stored layer outputs (debug).")
        .def("save_model", &network::save_model, py::arg("filename"))
        .def("load_model", &network::load_model, py::arg("filename"))
        .def("get_layer_sizes", &network::get_layer_sizes,
             "Return layer sizes as a vector<size_t>.");
}
