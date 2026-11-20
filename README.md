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

For interactive matplotlib windows, install the optional Qt backend:

```sh
uv sync --extra gui
uv run --extra gui python/geometry.py
uv run --extra gui python/visualize.py
```

Both commands open the module model; `visualize.py` uses PyVista instead when
given a VTK result file. A working desktop display is required. The viewer tries
Qt when installed, then Tk otherwise. If no interactive backend can initialize,
it saves `output/module.png` and reports its path. Use `--screenshot output/module.png`
for intentional headless rendering, or `--backend QtAgg` to require Qt and report
initialization errors. The `--extra gui` flag keeps Qt installed during `uv run`.

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
Materials and pad thickness remain unspecified. No solver mesh is generated.

`data/footprint.json` is the shared source for pin geometry and net definitions.
The generator preserves its original coordinates, provenance, warnings, and
unconfirmed assignments. `rect_mm` is interpreted as `[x_min, y_min, width, height]`;
source orientation still needs verification before manufacturing or simulation.
Pads are zero-thickness surfaces at the package bottom. The viewer shows a
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

`module.json` contains envelopes plus the complete footprint metadata.
`module.geom` is the dependency-free C++ representation, regenerated alongside it:
first line `POL_SIM_GEOMETRY_V1 mm`, followed by `BOX "name" xmin ymin zmin xmax
ymax zmax` and `PAD "name" "net" "status" x y width height corner_radius z` records.
Lengths are mm; future SI solvers must convert at their input boundary. The pad
radius is retained as supplied; the preview clamps it to half the shorter side.
Regenerate both files after editing config or footprint; do not edit exports.

All files under `data/` except `.gitkeep` are ignored because package geometry and
pin definitions are confidential. Restore authorized local data after cloning.
The VTK visualization script remains available for future simulation results.

Keep this repository private. No remote is configured or publication performed.
If hosted later, select private visibility. Confidential source documents belong
in `confidential/` or `inputs/`; proprietary settings belong in ignored
`config.yaml`, `config.*.yaml`, or `private/`. Generated data belongs in `output/`.
Ignore rules cannot enforce remote visibility or prevent explicitly forced adds;
review staged files before committing. Do not commit confidential values into
source code, the public-safe template, or documentation.
