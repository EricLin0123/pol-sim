#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
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
    std::array<double, 6> geometry; // x y width height corner_radius z; contact metadata, not an extra solid
};
struct ModuleCapsule {
    std::string name, material;
    std::array<double, 7> geometry; // ax ay az bx by bz radius (mm)
};
struct ModulePinVolume {
    std::string name, net, status, material;
    std::array<double, 7> geometry; // x y width height corner_radius zmin zmax (mm)
};
struct ModuleHeatSource {
    std::string name, host, surface, geometry_status;
    double depth_mm;
    std::optional<double> active_area_mm2, power_w;
    std::optional<std::array<double, 4>> footprint_xy_mm; // xmin ymin xmax ymax
    std::optional<std::array<double, 6>> bounds_mm; // resolved source subvolume, if placed
    // An annotation within host silicon, not an additional material body.
    // surface == unresolved means no source z placement may be inferred.
};
struct ModuleGeometry {
    std::vector<ModuleBox> boxes;
    std::vector<ModulePad> pads;
    std::vector<ModulePinVolume> pin_volumes;
    std::vector<ModuleCapsule> capsules;
    std::vector<ModuleHeatSource> heat_sources;
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
        } else if (kind == "HEAT_SOURCE" && v2) {
            ModuleHeatSource source;
            std::string area, power;
            int has_xy = -1;
            std::array<double, 4> xy;
            row >> std::quoted(source.name) >> std::quoted(source.host)
                >> std::quoted(source.surface) >> std::quoted(source.geometry_status)
                >> source.depth_mm >> area >> power >> has_xy;
            for (auto& value : xy) row >> value;
            if (!row || (has_xy != 0 && has_xy != 1) || !std::isfinite(source.depth_mm) || source.depth_mm <= 0)
                throw std::runtime_error("Invalid HEAT_SOURCE record");
            auto optional_value = [](const std::string& text) -> std::optional<double> {
                if (text == "unset") return std::nullopt;
                std::size_t consumed = 0;
                double value = std::stod(text, &consumed);
                if (consumed != text.size() || !std::isfinite(value) || value < 0)
                    throw std::runtime_error("Invalid HEAT_SOURCE value");
                return value;
            };
            source.active_area_mm2 = optional_value(area);
            source.power_w = optional_value(power);
            if (source.active_area_mm2 && *source.active_area_mm2 <= 0)
                throw std::runtime_error("Invalid HEAT_SOURCE area");
            if (source.surface != "unresolved" && source.surface != "top" && source.surface != "bottom")
                throw std::runtime_error("Invalid HEAT_SOURCE surface");
            for (double value : xy)
                if (!std::isfinite(value)) throw std::runtime_error("Invalid HEAT_SOURCE footprint");
            if (has_xy) {
                if (xy[0] >= xy[2] || xy[1] >= xy[3]) throw std::runtime_error("Invalid HEAT_SOURCE footprint");
                source.footprint_xy_mm = xy;
            }
            model.heat_sources.push_back(source);
        } else if (kind == "PIN_VOLUME" && v2) {
            ModulePinVolume pin;
            row >> std::quoted(pin.name) >> std::quoted(pin.net)
                >> std::quoted(pin.status) >> std::quoted(pin.material);
            for (auto& value : pin.geometry) row >> value;
            if (!row) throw std::runtime_error("Invalid PIN_VOLUME record");
            for (double value : pin.geometry)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite PIN_VOLUME coordinate");
            const auto& g = pin.geometry;
            if (g[2] <= 0 || g[3] <= 0 || g[4] < 0 || g[4] > g[2]/2 || g[4] > g[3]/2 || g[5] >= g[6])
                throw std::runtime_error("Invalid PIN_VOLUME dimensions");
            model.pin_volumes.push_back(pin);
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
    for (auto& source : model.heat_sources) {
        const ModuleBox* host = nullptr;
        for (const auto& box : model.boxes) if (box.name == source.host) host = &box;
        if (!host || source.depth_mm > host->bounds[5]-host->bounds[2]+1e-9)
            throw std::runtime_error("Invalid HEAT_SOURCE host or depth");
        if (source.footprint_xy_mm) {
            const auto& xy = *source.footprint_xy_mm;
            for (int i = 0; i < 2; ++i)
                if (xy[i] < host->bounds[i]-1e-9 || xy[i+2] > host->bounds[i+3]+1e-9)
                    throw std::runtime_error("HEAT_SOURCE lies outside host");
            if (source.surface != "unresolved") {
                double z0 = source.surface == "top" ? host->bounds[5]-source.depth_mm : host->bounds[2];
                source.bounds_mm = std::array<double, 6>{xy[0],xy[1],z0,xy[2],xy[3],z0+source.depth_mm};
            }
        }
    }
    if (!model.composite_envelope.empty()) {
        bool found = false;
        for (const auto& box : model.boxes)
            if (box.name == model.composite_envelope && box.role == "envelope") found = true;
        if (!found || model.capsules.empty()) throw std::runtime_error("Invalid composite region references");
    }
    return model;
}
