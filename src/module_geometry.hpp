#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host-side input only; future CUDA solvers can upload numeric arrays to device memory.
// All lengths are mm. Convert to SI at the solver boundary.
struct ModuleBox {
    std::string name;
    std::array<double, 6> bounds; // xmin ymin zmin xmax ymax zmax
};
struct ModulePad {
    std::string name, net, status;
    std::array<double, 6> geometry; // x y width height corner_radius z; zero thickness
};
struct ModuleGeometry {
    std::vector<ModuleBox> boxes;
    std::vector<ModulePad> pads;
};

inline ModuleGeometry load_module_geometry(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line) || line != "POL_SIM_GEOMETRY_V1 mm")
        throw std::runtime_error("Missing or invalid module geometry: " + path);
    ModuleGeometry model;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        std::string kind, trailing;
        row >> kind;
        if (kind == "BOX") {
            ModuleBox box;
            row >> std::quoted(box.name);
            for (auto& value : box.bounds) row >> value;
            if (!row) throw std::runtime_error("Invalid BOX record");
            for (double value : box.bounds)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite BOX coordinate");
            for (int i = 0; i < 3; ++i)
                if (box.bounds[i] >= box.bounds[i+3]) throw std::runtime_error("Invalid BOX bounds");
            model.boxes.push_back(box);
        } else if (kind == "PAD") {
            ModulePad pad;
            row >> std::quoted(pad.name) >> std::quoted(pad.net) >> std::quoted(pad.status);
            for (auto& value : pad.geometry) row >> value;
            if (!row) throw std::runtime_error("Invalid PAD record");
            for (double value : pad.geometry)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite PAD coordinate");
            if (pad.geometry[2] <= 0 || pad.geometry[3] <= 0 || pad.geometry[4] < 0)
                throw std::runtime_error("Invalid PAD dimensions");
            model.pads.push_back(pad);
        } else {
            throw std::runtime_error("Unknown module geometry record");
        }
        if (row >> trailing) throw std::runtime_error("Trailing geometry fields");
    }
    if (model.boxes.empty()) throw std::runtime_error("Empty module geometry");
    return model;
}
