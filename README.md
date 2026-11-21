# CUDA Power Module Simulation

Research foundation for **CUDA-Accelerated Thermal and Electromagnetic Simulation
of Integrated-Inductor DC-DC Power Modules**, including top liquid cooling.
All numerical solvers will use CUDA C++17; Python handles geometry, processing,
plots, animation, and interactive viewing. No physics models, CPU solvers,
simulation meshes, or loss estimates are implemented yet.

Requirements: NVIDIA CUDA Toolkit, a compatible NVIDIA GPU and driver for execution,
CMake 3.24+, a CUDA-compatible C++ compiler, Python 3.11+, and `uv`.
PyVista interactive viewing needs a graphical display; offscreen rendering needs
a working VTK EGL/OSMesa/display backend. Gmsh may require system OpenGL libraries.

The Qt backend is included in the default dependencies. To view the full assembly:

```sh
uv sync
uv run python/geometry.py
uv run python/visualize.py
```

`visualize.py` rebuilds the full assembly from current local inputs on every run.
It includes illustrative winding details when actual winding dimensions are unset,
with that assumption labeled in the viewer. No model-selection or demo arguments
are needed. Sidebar checkboxes independently toggle the package envelope,
composite boundary, silicon die, winding, leads, terminals, FET annotations, and
pads/pins across the 3D and plan views. All start visible. Uncheck envelopes or
pads to inspect interior parts. Viewer exports are stored in
`output/visualization/`, keeping canonical simulation inputs separate.
`geometry.py` remains the explicit geometry generator. `visualize.py` uses PyVista
instead when given a VTK result file. A working desktop display is required. The viewer tries
Qt when installed, then Tk otherwise. If no interactive backend can initialize,
it saves `output/module.png` and reports its path. Use `--screenshot output/module.png`
for intentional headless rendering, or `--backend QtAgg` to require Qt and report
initialization errors.

From the repository root:

```sh
git init
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv sync
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 2
./build/thermal
./build/em
```

For compilation without a working GPU driver, replace `native` with the compute
architecture of the intended GPU (for example `75` for compute capability 7.5).
This is a build target, not a device specification. Both executables initialize
CUDA and print GPU 0's name and compute capability, or exit nonzero on failure.

The initialized workspace contains a confidential, ignored `config.yaml`. In a
fresh clone, run `cp config.example.yaml config.yaml` and fill its null entries
using authorized local inputs. The template intentionally contains no device
values. Missing physical parameters stay null until established.

```sh
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/geometry.py --config config.yaml
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/geometry.py --screenshot output/geometry.png
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/visualize.py
# Once a CUDA exporter exists:
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/visualize.py output/result.vtu --scalar temperature
```

The local geometry configuration now defines a centered IC package and inductor
stack, with the package bottom at z=0 and no additional interface thickness.
These are assumed outer envelopes, not internal die, winding, or core geometry.
Material properties are kept in local configuration. No solver mesh is generated.

`data/footprint.json` is the shared source for pin geometry and net definitions.
The generator preserves its original coordinates, provenance, warnings, and
unconfirmed assignments. `rect_mm` is interpreted as `[x_min, y_min, width, height]`;
source orientation still needs verification before manufacturing or simulation.
Pin lands use configurable thickness and bottom z, extruded as rounded copper
prisms. Original footprint data remains unchanged. The viewer shows a
rotatable 3D assembly and a labeled pad-plane view, using matplotlib.

```sh
# Generate data/module.json and data/module.geom, then open matplotlib:
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/geometry.py
# Export without a display:
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/geometry.py --no-view
# View a saved model, including from another working directory:
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv run python python/geometry.py --model data/module.json
# CUDA programs read the shared geometry before initializing the GPU:
./build/thermal --model data/module.geom
./build/em --model data/module.geom
```

Default input/output paths are anchored to the repository, not the current working
directory. Python tasks can read `data/module.json` directly with `json.load`.
CUDA C++ tasks include `src/module_geometry.hpp` and call
`load_module_geometry(path)`; both scaffold executables already support it.
The reader loads host-side arrays for later GPU upload; it does not solve physics.

`module.json` contains geometry plus the complete footprint metadata.
`module.geom` is the dependency-free C++ representation, regenerated alongside it.
The V2 format begins with `POL_SIM_GEOMETRY_V2 mm` and uses these records:

```text
BOX "name" "material" "role" xmin ymin zmin xmax ymax zmax
CAPSULE "name" "material" ax ay az bx by bz radius
COMPOSITE "envelope_box_name" wire_clearance
ILLUSTRATIVE 0_or_1
PAD "name" "net" "status" x y width height corner_radius z
PIN_VOLUME "name" "net" "status" "material" x y width height effective_corner_radius zmin zmax
HEAT_SOURCE "name" "host_box" "surface" "status" depth area_or_unset power_or_unset has_xy xmin ymin xmax ymax
```

The C++ reader also accepts the earlier V1 envelope format.
Lengths are mm; future SI solvers must convert at their input boundary. The pad
radius is retained as supplied; the preview clamps it to half the shorter side.
Regenerate both files after editing config or footprint; do not edit exports.

An optional local `die` proposal adds a silicon solid and heat-source annotations
to both exports. Die/FET x/y coordinates are relative to the IC package origin;
die bottom z is relative to the package bottom. Heat-source areas and powers are
retained separately; an unset power is not zero. Control logic stays unplaced
unless its geometry is supplied. FET annotations remain silicon, not extra solids.
`heat_source_surface` may be `top`, `bottom`, or null. A null surface preserves the
specified depth but leaves source z bounds unresolved. When a surface is specified, the viewer draws finite-depth FET subvolumes;
otherwise it shows labeled projections only. A separate floorplan remains visible. The package remains an
envelope because its internal leadframe, die attach, and molding fill are unknown.
In `HEAT_SOURCE` records, `has_xy=0` means the four following coordinates are
ignored; `surface="unresolved"` forbids inferring source z. CUDA readers retain
unknown powers/areas as `std::nullopt`, not zero. Geometry status is preserved.

`geometry.pin_thickness_mm` and `pin_bottom_z_mm` control copper pad volumes;
bottom z is relative to the package bottom. `PIN_VOLUME` records carry rounded
prism geometry into CUDA C++; legacy `PAD` records describe contact faces only
and must not be counted as additional solids. Pin corner radii are clamped to
half the smaller pad dimension, matching the existing footprint preview.
FET sources occupy their configured depth **within** the silicon die. A mesher
must fragment silicon into source regions and remaining silicon; adding overlapping
source solids to an intact die would double-count volume. C++ heat-source
`bounds_mm` stores the resolved subvolume when its surface and footprint exist.

`module.json` includes `meshing_status`, with overlapping pin bounding-box
candidates, unplaced heat sources, and remaining geometry omissions. This is
an input audit, not a guarantee of a watertight mesh. The package is still an
envelope, and the composite is a difference-region description. A conformal
Boolean partition, material assignment to every remaining region, and actual
volume meshing are still required. Preview triangles/tubes are display assets,
not a closed, stitched simulation mesh.

Parametric choke internals are implemented in `python/inductor.py`. Configure
`inductor.winding_geometry` in the local YAML for a round-wire helix with radial
leads and two rectangular terminals. Parameters control turns, wire diameter,
radius, winding height and bottom offset, insulation clearance, terminal size
and radial location, and tessellation. Winding bottom is relative to the
inductor bottom. An integer turn count places terminals at the same angle and
is rejected if they overlap; this topology does not represent every choke.
The material of the terminals, insulation properties, and composite properties
remain unspecified. Inductance is not computed or calibrated by this generator.

Unknown device internals remain null. To inspect explicitly illustrative
parameters without changing the device configuration or canonical exports:

```sh
uv run python/geometry.py --demo-winding
uv run python/geometry.py --demo-winding --screenshot output/inductor-demo.png
uv run python/visualize.py --model output/inductor-demo/module.json
./build/thermal --model output/inductor-demo/module.geom
./build/em --model output/inductor-demo/module.geom
```

Demo files go to `output/inductor-demo/` and carry an illustrative flag in JSON
and CUDA input. The viewer shows the composite's transparent outer boundary;
its interior is defined by constructive geometry in the exports, not a volume
mesh. Material priority is the copper capsule union, terminal boxes minus
copper, then composite envelope minus terminals and clearance-expanded capsules.
The intervening clearance sleeve has no assigned material. Envelope boxes are
not additional physical solids. These rules prevent material double counting
when downstream solvers implement region classification. Geometric checks
cover containment, winding pitch, and terminal overlap, not all manufacturing
constraints or self-intersections. The demo is not matched to a reference image.

All files under `data/` except `.gitkeep` are ignored because package geometry and
pin definitions are confidential. Restore authorized local data after cloning.
The VTK visualization script remains available for future simulation results.

This repository publishes only the generic research scaffold. Confidential source
documents belong in `confidential/` or `inputs/`; proprietary settings belong in ignored
`config.yaml`, `config.*.yaml`, or `private/`. Generated data belongs in `output/`.
Ignore rules cannot enforce remote visibility or prevent explicitly forced adds;
review staged files before committing. Do not commit confidential values into
source code, the public-safe template, or documentation.
