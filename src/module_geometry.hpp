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
    std::string material = "unassigned", role = "envelope";
    std::array<double, 6> bounds; // xmin ymin zmin xmax ymax zmax
};
struct ModulePad {
    std::string name, net, status;
    std::array<double, 6> geometry; // x y width height corner_radius z; zero thickness
};
struct ModuleCapsule {
    std::string name, material;
    std::array<double, 7> geometry; // ax ay az bx by bz radius (mm)
};
struct ModuleGeometry {
    std::vector<ModuleBox> boxes;
    std::vector<ModulePad> pads;
    std::vector<ModuleCapsule> capsules;
    std::string composite_envelope;
    double wire_clearance_mm = 0;
    bool illustrative = false;
    // Material priority: copper capsule union, terminal boxes minus copper,
    // then composite envelope minus terminals and clearance-expanded capsules.
    // The clearance sleeve has no assigned material. Envelopes are not solids.
};

inline ModuleGeometry load_module_geometry(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line) ||
        (line != "POL_SIM_GEOMETRY_V1 mm" && line != "POL_SIM_GEOMETRY_V2 mm"))
        throw std::runtime_error("Missing or invalid module geometry: " + path);
    const bool v2 = line == "POL_SIM_GEOMETRY_V2 mm";
    ModuleGeometry model;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        std::string kind, trailing;
        row >> kind;
        if (kind == "BOX") {
            ModuleBox box;
            row >> std::quoted(box.name);
            if (v2) row >> std::quoted(box.material) >> std::quoted(box.role);
            for (auto& value : box.bounds) row >> value;
            if (!row) throw std::runtime_error("Invalid BOX record");
            for (double value : box.bounds)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite BOX coordinate");
            for (int i = 0; i < 3; ++i)
                if (box.bounds[i] >= box.bounds[i+3]) throw std::runtime_error("Invalid BOX bounds");
            model.boxes.push_back(box);
        } else if (kind == "CAPSULE" && v2) {
            ModuleCapsule capsule;
            row >> std::quoted(capsule.name) >> std::quoted(capsule.material);
            for (auto& value : capsule.geometry) row >> value;
            if (!row) throw std::runtime_error("Invalid CAPSULE record");
            for (double value : capsule.geometry)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite CAPSULE coordinate");
            if (capsule.geometry[6] <= 0) throw std::runtime_error("Invalid CAPSULE radius");
            double length2 = 0;
            for (int i = 0; i < 3; ++i) {
                double delta = capsule.geometry[i+3] - capsule.geometry[i];
                length2 += delta*delta;
            }
            if (length2 == 0) throw std::runtime_error("Zero-length CAPSULE");
            model.capsules.push_back(capsule);
        } else if (kind == "COMPOSITE" && v2) {
            if (!model.composite_envelope.empty()) throw std::runtime_error("Duplicate COMPOSITE");
            row >> std::quoted(model.composite_envelope) >> model.wire_clearance_mm;
            if (!row || !std::isfinite(model.wire_clearance_mm) || model.wire_clearance_mm < 0)
                throw std::runtime_error("Invalid COMPOSITE clearance");
        } else if (kind == "ILLUSTRATIVE" && v2) {
            int flag = -1;
            row >> flag;
            if (!row || (flag != 0 && flag != 1)) throw std::runtime_error("Invalid ILLUSTRATIVE flag");
            model.illustrative = flag == 1;
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
    if (!model.composite_envelope.empty()) {
        bool found = false;
        for (const auto& box : model.boxes)
            if (box.name == model.composite_envelope && box.role == "envelope") found = true;
        if (!found || model.capsules.empty()) throw std::runtime_error("Invalid composite region references");
    }
    return model;
}
