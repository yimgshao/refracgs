#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <WaterTracer.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<WaterTracer>(m, "WaterTracer")
        .def(pybind11::init<std::string, std::string>())
        .def("trace", &WaterTracer::trace)
        .def("build_bvh", &WaterTracer::buildBVH);
}
