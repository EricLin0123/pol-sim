# CUDA Power Module Simulation

Research foundation for **CUDA-Accelerated Thermal and Electromagnetic Simulation
of Integrated-Inductor DC-DC Power Modules**, including top liquid cooling.
All numerical solvers will use CUDA C++17; Python handles geometry, processing,
plots, animation, interactive viewing, and Gmsh volume meshing. Steady-state
thermal assembly and solving use native CUDA C++; EM remains a scaffold.

## Steady-state thermal simulation (CUDA only)

With the existing mesh and populated local `config.yaml`:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 2
./build/thermal --self-test
uv run python/thermal.py
uv run python/visualize.py output/thermal/temperature.vtu
uv run python/visualize.py output/thermal/components/silicon_die.vtu
```

On a build machine without a usable GPU, specify the intended compute
architecture instead of `native`. A compatible GPU/driver with compute capability
6.0+ is required to execute the double-precision CUDA solver. If device discovery
fails, execution stops; no CPU fallback or fabricated temperature field is used.
`uv run python/thermal.py --prepare-only` validates/prepares inputs without a GPU;
`./build/thermal output/thermal` runs already-prepared inputs, followed by
`uv run python/thermal.py --postprocess-only` to build visualization artifacts.

Python reads current YAML material conductivities and mesh connectivity, orients
exterior faces, constructs CSR **topology only**, and converts mm to meters.
CUDA assembles conforming linear tetrahedral FEM stiffness matrices and uses
Jacobi-preconditioned conjugate gradients in double precision. Shared interface
nodes enforce temperature continuity and conservative assembled heat balance;
cellwise gradients need not have identical pointwise face fluxes. Source powers
are normalized by actual meshed region volume, then audited by integrating them.
Steady state uses conductivity; density and heat capacity enter the optional
transient mode below.

All assumed losses, reference temperatures, temperature coefficients, still-air
boundary coefficients, emissivities, the uncertain lumped PCB path and solver
tolerances live under local `thermal:` YAML. Existing geometry/material entries
are preserved. The operating point and on-resistances come from the existing
electrical configuration. The user disabled control logic: auxiliary loss is
zero, and nonzero auxiliary power is rejected until a real region is supplied.
The fixed core-loss placeholder is unvalidated; other core-loss models currently
raise an error rather than silently using a guessed Steinmetz law. No efficiency
target is used: efficiency is output power divided by output power plus losses.

Convection uses consistent triangle matrices; nonlinear gray radiation uses
three-point triangle quadrature and secant-coefficient Picard iteration. The
outer loop updates MOSFET/winding resistance from component volume-average
temperatures. Temperatures, individual losses, nonlinear residual and overall
energy balance must all converge. A temperature-domain guard and iteration
limits stop invalid or unconverged runs. CUDA self-tests cover an analytic
tetrahedron, integrated source loading, Robin assembly, PCG and boundary heat.

Only exterior triangles exchange heat with ambient. Upward faces (including
package ledges) use the top convection coefficient, vertical faces use the side
coefficient, and downward faces use an explicitly configured bottom coefficient
(initially zero). Radiation assumes a large ambient enclosure with unit view
factor; mutual radiative occlusion is not modeled. Bottom pin contact faces use
one total PCB conductance distributed by contact area; those faces have no
additional air convection/radiation. An explicit PCB and the lumped PCB path
cannot be enabled together. No liquid cooling is applied in this model.

Ignored `output/thermal/` contains `history.tsv`, native `solution.json`,
`temperature.bin`, `temperature.vtu`, individual component VTUs, `summary.png`,
and `report.json` with component maximum/volume-average temperatures, loss
breakdown, derived efficiency, heat-removal paths and energy error. Temperature
is the viewer's default field; the **Slice only** checkbox interpolates it on
the draggable plane. The report records the configuration/mesh hashes and
assumptions. Nonconverged iterates are explicitly labeled and return exit code 2;
initialization/errors return 1. `run_status.json` records failed CUDA execution
without presenting a temperature prediction. Old result files are removed before
a new native run.

The geometry remains hypothetical, the winding is illustrative, and the current
mesh has flagged skinny elements. No mesh-convergence or experimental validation
has been established. Contact resistances, temperature-dependent material
conductivity, saturation, AC winding physics and switching waveforms are not
modeled. The specified lumped PCB path and fixed core loss are uncertain inputs,
so even a converged numerical result is a preliminary model prediction.

## Temperature evolution and playback

```sh
cmake --build build -j 2
uv run python/thermal.py --transient
uv run python/visualize.py output/thermal_transient/temperature.vtu
```

The temperature viewer automatically detects the adjacent time-series file.
Use the **Physical time** slider, **Play / pause**, Space, or the arrow keys to
advance through saved frames. Slicing, clipping, material toggles, and element
edges remain available during playback. The temperature color scale stays fixed
over the entire series so changing colors represent changing temperatures.
Playback speed is a display setting (150 ms per saved frame), not physical time.
You can also open `output/thermal_transient/temperature_series.npz` directly.

Transient mode solves `rho*Cp*dT/dt - div(k*grad(T)) = q` with lumped nodal
heat capacity and first-order implicit backward Euler, entirely on CUDA.
Each time step converges temperature-dependent losses and radiation before
acceptance. Energy accounting includes `sum(C_i*(T_i_new-T_i_old))/dt`; it does
not require instantaneous heat removal to equal generation during warm-up.
Nonlinear solver iterations are never presented as elapsed physical time.

Local `thermal.transient` controls `initial_temperature_k`, `time_step_s`,
`duration_s`, and `snapshot_stride`. Initial settings use ambient temperature,
0.1 s steps for 60 s, with a snapshot every five accepted steps plus the initial
and final states. The configured full electrical load is applied at t=0.
These initial discretization settings need time-step refinement and can smear
fast die transients. A completed time interval is not automatically steady state.
Composite density/specific heat retain their supplied uncertainty markers;
the lumped PCB path has no board thermal inertia.

`output/thermal_transient/` contains the normal final-state outputs plus
`temperature_series.npz` (physical times and nodal Kelvin temperatures),
`component_time_history.npz` (component mean/max temperatures), and
`time_history.tsv` (heat generation, removal, storage, and energy error per step).
One mesh is reused across all frames. Only accepted steps are saved; failed
steps roll back and produce an explicitly partial history. The CUDA self-test
includes an analytic adiabatic heating step and energy-storage check. Both time
integration and its GPU tests require a working NVIDIA driver; no CPU solver or
invented thermal animation is substituted if CUDA is unavailable.

## Volume meshing (no solver)

```sh
uv run python/mesh.py
# Inspect labeled tetrahedra:
uv run python/visualize.py output/mesh/module.vtu --scalar region_id
# Or open Gmsh's mesh interface:
uv run python/mesh.py --view
# Adjust discretization without changing physical dimensions:
uv run python/mesh.py --size 0.20 --fet-size 0.02
```

The mesh viewer starts with black element edges and a movable cutaway plane.
Drag the plane to reveal interior tetrahedra, or rotate its normal to change
the cutting direction. The sidebar toggles edges, clipping, and individual
regions; hide EMC/composite to see the winding and die unobstructed. Cutaway
retains whole tetrahedra so their actual faces remain visible. Moving the camera
inside an opaque outer surface alone does not provide a useful interior cut.
These display controls do not modify the saved mesh.

The mesher rebuilds the current local model, including the explicitly labeled
illustrative winding if actual winding parameters remain unset. Use
`--model path/to/module.json` to mesh an explicit saved model instead.
Input needs finite copper pins and placed FET source regions. No CUDA, GPU, or
thermal solver is used. All geometry and mesh coordinates remain in millimeters.

Gmsh constructs rounded copper prisms, the wire's capsule union, terminals,
clearance, die and package envelopes; unions overlapping copper pins; and
fragments all solids together. FETs partition silicon rather than overlap it.
Each tetrahedron belongs to exactly one region. `geometry.ic_fill_material`
sets the remaining IC volume's material; the simplified EMC fill excludes die
and pins and introduces no leadframe. `inductor.terminal_material` sets terminal
material. Zero `insulation_clearance_mm` gives direct copper/composite contact
and creates no insulation volume. Positive clearance creates a separately
unassigned sleeve. Demo winding dimensions respect the configured clearance.
The mesh report includes each region's stored thermal properties and lists
any missing material assignments. Control logic without geometry,
PCB, solder joints, and cooling interface layers are omitted. `top_cooling_candidate`
is only a surface label; no cooling condition is applied.

Outputs under ignored `output/mesh/`:

- `module.msh`: Gmsh mesh with physical region and exterior-boundary groups.
- `module.brep`: partitioned solid geometry (labels are in the report/mesh).
- `module.vtu`: tetrahedral mesh with region and quality cell fields for PyVista.
- `module.npz`: `points_mm`, zero-based `tetrahedra`, `region_id`,
  `element_volume_mm3`, `quality_minSICN`, exterior triangles/IDs, and interface
  triangles/region pairs. Triangle orientation is not guaranteed outward.
- `report.json`: region-to-material mapping, CAD/mesh volumes, element counts,
  quality statistics, assumptions and unresolved geometry.
- `input_geometry/`: snapshot of the geometry actually meshed.

Validation checks positive tetrahedron volumes, complete region ownership,
matching assembly volume, shared interior triangle connectivity, and absence
of unexpected free/nonmanifold faces. Local refinement targets FET surfaces;
defaults are bulk size 0.30 mm and FET size 0.04 mm. These are initial meshing
settings, not a convergence claim or a guarantee of multiple layers through
thin regions. The report flags elements with minSICN below 0.05 as a quality
diagnostic. Skinny elements can remain near capsule joints/clearance boundaries;
inspect and improve them before future solving. A valid conformal mesh does
not establish thermal-model completeness or simulation accuracy.

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
uv run python/thermal.py
./build/em
```

For compilation without a working GPU driver, replace `native` with the compute
architecture of the intended GPU (for example `75` for compute capability 7.5).
This is a build target, not a device specification. The thermal executable
requires prepared inputs; both executables require an accessible CUDA device.

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
Material properties are kept in local configuration. Geometry generation and
meshing are separate commands.

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
# Thermal solver uses the mesh, not only geometry envelopes:
uv run python/thermal.py
./build/em --model data/module.geom
```

Default input/output paths are anchored to the repository, not the current working
directory. Python tasks can read `data/module.json` directly with `json.load`.
CUDA C++ tasks include `src/module_geometry.hpp` and call
`load_module_geometry(path)`; the EM scaffold supports this reader. The thermal
solver instead consumes the preprocessed volume mesh.
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
otherwise it shows labeled projections only. A separate floorplan remains visible.
The package outer box describes an envelope; its remainder uses the configured
fill material after the die and pins are removed.
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
Boolean partition and volume meshing are performed by `mesh.py`; the
geometry export itself is not a mesh. Preview triangles/tubes are display assets,
not a closed, stitched simulation mesh.

Parametric choke internals are implemented in `python/inductor.py`. Configure
`inductor.winding_geometry` in the local YAML for a round-wire helix with radial
leads and two rectangular terminals. Parameters control turns, wire diameter,
radius, winding height and bottom offset, insulation clearance, terminal size
and radial location, and tessellation. Winding bottom is relative to the
inductor bottom. An integer turn count places terminals at the same angle and
is rejected if they overlap; this topology does not represent every choke.
Materials are referenced by name and stored thermal properties are copied into
local JSON exports. A zero clearance removes the insulation region entirely. Inductance is not computed or calibrated by this generator.

Unknown device internals remain null. To inspect explicitly illustrative
parameters without changing the device configuration or canonical exports:

```sh
uv run python/geometry.py --demo-winding
uv run python/geometry.py --demo-winding --screenshot output/inductor-demo.png
uv run python/visualize.py --model output/inductor-demo/module.json
uv run python/thermal.py
./build/em --model output/inductor-demo/module.geom
```

Demo files go to `output/inductor-demo/` and carry an illustrative flag in JSON
and CUDA input. The viewer shows the composite's transparent outer boundary;
its interior is defined by constructive geometry in the exports, not a volume
mesh. Material priority is the copper capsule union, terminal boxes minus
copper, then composite envelope minus terminals and clearance-expanded capsules.
A sleeve exists only for positive clearance and then needs its own material. Envelope boxes are
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
