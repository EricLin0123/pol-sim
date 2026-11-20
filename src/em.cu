#include <cuda_runtime.h>
#include <iostream>
#include "module_geometry.hpp"

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--model")) {
        std::cerr << "Usage: " << argv[0] << " [--model path/to/module.geom]\n";
        return 1;
    }
    if (argc == 3) {
        try {
            const auto model = load_module_geometry(argv[2]);
            std::cout << "Loaded " << model.boxes.size() << " boxes and "
                      << model.pads.size() << " pad surfaces and " << model.capsules.size()
                      << " wire capsules (mm)." << std::endl;
            if (model.illustrative) std::cout << "ILLUSTRATIVE internal geometry; not device-validated.\n";
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::cerr << "CUDA discovery failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cerr << "No CUDA devices detected.\n";
        return 1;
    }
    status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(nullptr); // Initialize runtime context.
    cudaDeviceProp device{};
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&device, 0);
    if (status != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    std::cout << "em scaffold: GPU 0: " << device.name
              << " (compute " << device.major << '.' << device.minor << ")\n";
    std::cout << "No simulation implemented.\n";
    return 0;
}
