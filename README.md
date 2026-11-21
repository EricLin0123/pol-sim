# CUDA Power Module Simulation

Research foundation for **CUDA-Accelerated Thermal and Electromagnetic Simulation
of Integrated-Inductor DC-DC Power Modules**, including top liquid cooling.
All numerical solvers will use CUDA C++17; Python handles geometry, processing,
plots, animation, interactive viewing, and Gmsh volume meshing. Steady-state
thermal and magnetostatic assembly/solving use native CUDA C++. Magnetic
field animation currently uses an explicitly labeled no-eddy-current approximation.

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

## Selectable air and equivalent liquid cooling

```sh
uv run python/thermal.py --cooling air
uv run python/thermal.py --cooling liquid
uv run python/thermal.py --compare
uv run python/visualize.py output/thermal_liquid/temperature.vtu
# Same transient solver and playback, using the liquid boundary:
uv run python/thermal.py --cooling liquid --transient
uv run python/visualize.py output/thermal_liquid_transient/temperature.vtu
```

`thermal.cooling.mode` selects `air` or `liquid`; `--cooling` overrides it for a
run without editing the original air coefficients. Legacy `still_air` is accepted
as an alias for `air`. The existing air output paths remain `output/thermal/`
and `output/thermal_transient/`. Liquid output paths are separate. The old
root-level `cooling` block is legacy metadata; the selector is under `thermal`.

`thermal.cooling.liquid` configures coolant temperature, effective coefficient,
contact area, other-surface air coefficient, ambient/radiation temperatures, and
a **report-only** top temperature objective. Actual case values remain in the
ignored local configuration. The existing tagged inductor top must match the
configured contact area. IC ledges, sides, and the bottom are not liquid-cooled.
Other exposed faces use the liquid case's air coefficient and configured radiation model. The covered
top has neither extra air convection nor radiation. The unchanged PCB contact
path remains separate; its contact faces are not treated as exposed air faces.

The common CUDA FEM boundary assembly integrates `h*(T_surface-T_coolant)` on
the boundary triangles. Conductivity and conduction distance are already in the
tetrahedral stiffness matrix. The finite-volume half-cell series-resistance
formula applies to cell-centered finite volumes, not these surface-node finite
elements; adding it here would double-count conduction. No fixed-temperature
constraint, CFD model, cold-plate geometry, or new meshing method is introduced.

Reports add area-weighted average and maximum top temperatures, separate liquid
heat removal in watts, and the difference from the design objective. The design
objective is not sent to the CUDA boundary-condition parameters. Steady-state
energy balance includes liquid, air convection, radiation, and PCB heat removal;
transient accounting additionally includes energy storage.

`--compare` runs both steady-state modes with identical mesh and electrical/loss
parameters through the same code. Outputs go to `output/cooling_comparison/air/`
and `liquid/`. Only after both runs converge does it write
`temperature_comparison.png`, `temperature_difference.vtu` (liquid minus air),
and quantitative results in `comparison.json`. The existing viewer can open the
difference VTU directly. If either CUDA run fails, the comparison status explains
the missing result and no temperature comparison is fabricated. `--compare
--prepare-only` validates both input sets without solving.

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
`geometry.py` remains the explicit geometry generator. All 3D viewers use PyVista
for interactive rotation, zoom and part visibility controls. A desktop display
is required for interactive viewing. Use `--screenshot output/module.png` or
`--backend off_screen` for headless module rendering.

From the repository root:

```sh
git init
UV_CACHE_DIR=/tmp/pol-sim-uv-cache uv sync
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 2
uv run python/thermal.py
uv run python/em.py
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
rotatable 3D assembly and a labeled pad-plane view, using PyVista.

```sh
# Generate data/module.json and data/module.geom, then open PyVista:
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
`load_module_geometry(path)`; the legacy EM `--model` inspection command supports this reader. The thermal
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

## Single-module PCB reference viewer

```sh
uv run python/pcb.py
uv run python/pcb.py --no-view
uv run python/pcb.py --config config.yaml --output output/pcb/layout
```

The PCB plan viewer reads `pcb.layout_parameters` in the project `config.yaml`,
initialized from `ref/layout_params.yaml`, and the module footprint from
`geometry.module_footprint_mm`. It preserves the repeated tracing from
`ref/generate_pcb.py` using Matplotlib, with pan/zoom and PNG, SVG, and JSON
exports under `output/pcb/`. Default paths work from any working directory.
The view uses millimeters with x right and y down from the reference upper-left.
Module count, pitch, margin, via diameters, component marker size, and colors
are configurable. The supplied single-module reference is illustrative;
VIN segments and VOUT regions do not establish verified parallel connectivity.
The JSON records placement and unresolved EM inputs, not a conductor mesh.
Layer stackup, thicknesses, exact copper boundaries, connectivity, materials,
and ports still require definition before EM simulation. The illustration
bounds are not a physical board outline. This viewer does not alter the
single-module thermal geometry or enable an explicit thermal PCB.

## Full 3D PCB assembly

```sh
uv run python/pcb_3d.py                 # Rotate, zoom, and toggle components
uv run python/pcb_3d.py --no-view       # Assembly JSON and PNG
uv run python/pcb_3d.py --no-view --cad # Also export STEP and BREP solids
```

`pcb.assembly_3d` configures PCB copper thickness (0.02 mm), dielectric
thickness (locally assumed 1.0 mm FR4), top-copper z, and module z-axis rotation.
Coordinates are right-handed: x right, y north, z up. FR4 spans -1 to 0 mm,
copper spans 0 to 0.02 mm, and the module package bottom sits at 0.02 mm. The
single instance uses the existing module definition, including die, pins,
terminals and the labeled illustrative winding when actual winding inputs are
missing. Module pin thickness is a separate device parameter and is preserved.

Outputs are in `output/pcb_3d/`: `assembly.json` stores copper prism outlines,
via radii/depths, substrate bounds, the full module definition and rigid instance
placements; `assembly.png` previews the assembly at true scale. Checkboxes hide
package envelopes, substrate, copper, vias, pins and winding independently.
The PyVista preview uses polygon surfaces/tubes; drilled holes and exact
capsule end caps are represented in CAD rather than the lightweight preview.

`--cad` uses Gmsh/OpenCASCADE to partition the existing single module,
extrude the traced PCB copper, drill the substrate/copper and
add hollow via barrels with 0.02 mm walls and annular top lands. It exports
`assembly.step`, `assembly.brep`, and `cad_regions.json`. CAD export can take
several minutes. Region volume tags refer to the generating Gmsh session and
must be remapped after CAD import. STEP/BREP preserve the volume geometry;
`assembly.json` retains the source definitions and assumptions.

The shared module now includes assumed buck power wiring: VIN to the high-side
FET drain; its source to the switch node shared by the low-side FET drain and
the -Y inductor terminal; the +Y inductor terminal to VOUT; and the low-side
source to PGND. Separate copper contact pads and stamped copper clips (0.30 mm wide, 0.10 mm
thick) represent
these connections in the module preview, PCB assembly, and CAD generation.
`buck_topology` records the connections, external supply/load ports, and separate
high-side-on and low-side-on paths. FET channels remain silicon, not copper shorts.
Contact placement and clip routing are illustrative assumptions; solder joints
are ideal contacts without a separate solder volume. No power bond wires are
used. Control/logic wiring remains unspecified. External load,
supply, capacitor geometry, and return-current distributions remain undefined.
The existing EM filament-loop source is unchanged and does not simulate this
branched buck circuit; previous EM results do not validate the new geometry.

These are preliminary assembly solids, not a conformal Maxwell mesh. The
top copper follows the reference, with a continuous lower PGND pour joining the
center PGND island and the four lower side pins (L6/L7/R6/R7). No bottom copper
layer is added; the rectangular substrate outline comes from the illustration bounds.
Traced copper overlaps are retained and may create shorts. Module land alignment,
physical net connectivity remains unverified despite the assumed internal buck wiring. Passive body
markers are omitted because electrical terminal and body geometry are unknown.
Repair/verify connectivity and fragment the entire assembly before volume meshing;
then supply frequency-dependent EM materials, air domain, ports and boundaries.
The existing thermal mesh and solver inputs are unchanged.

## CUDA 3D magnetic field: first magnetostatic milestone

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 # Or native on a working GPU host
cmake --build build --target em -j 2
./build/em --self-test
uv run python/em.py
uv run python/visualize.py output/em/magnetic.vtu
# Export the actual native field snapshots as an animated GIF:
uv run python/em_view.py output/em/magnetic.vtu --gif output/em/field.gif
# Optional existing FR4 slab; still one energized module:
uv run python/em.py --pcb --output output/em_pcb
uv run python/visualize.py output/em_pcb/magnetic.vtu
```

This is **linear magnetostatics followed by a quasistatic, no-eddy-current
approximation**, not a transient conductive MQS/A–φ implementation. Native CUDA
assembles the operator and source, solves for oriented edge integrals of A,
computes cell vectors B=curl(A), independently checks a second current, computes
the buck waveform, and scales the vector field into physical-time snapshots.
Python constructs geometry/topology, orchestrates runs, evaluates analytical
references, computes summaries, and visualizes results. There is no CPU field
solver or fabricated field fallback. A CUDA device with double atomicAdd support
(compute capability >=6.0) is required to solve. A missing driver or unconverged
solve produces an error and no published field series. Existing thermal inputs,
mesh, executable and solvers remain separate.

The local user-authorized demonstration uses the existing illustrative helix,
0.12 mm wire radius and an explicit external return at terminal height, routed
around the package. Coordinates are saved in `em.source.return_path_mm`; this
is not a verified PCB buck loop. The public template leaves that path unset.
Actual winding dimensions remain unknown. `em.source.use_illustrative_winding`
controls the existing demo geometry; disable it only after setting real
`inductor.winding_geometry` values. The source follows terminal 1 → lead 1 →
helix → lead 2 → terminal 2 → configured return → terminal 1. All source paths
are checked against the copper material volumes. Source return coordinates must
start/end at the corresponding terminal centers; an open source is rejected.

The source is a finite-radius closed-filament area quadrature inside swept copper
volumes, not an infinitely thin centerline and not a solved conductor current
distribution. Radial/azimuthal quadrature controls are in YAML. Every straight
filament segment is clipped into tetrahedra, with exact Whitney basis line
integrals between barycentric endpoints. GPU assembly therefore preserves
current continuity at the discrete nodal-gradient level, including bends and
return conductors. Uniform cross-sectional current is only approximated; refine
source quadrature before making quantitative predictions near conductors.

The air mesh is an explicit OpenCASCADE partition of the module, winding,
configured return and external box. First-order **Nédélec/Whitney edge elements**
replace the thermal solver's scalar nodal unknowns. Global edge directions and
signs enforce tangential A continuity; their element curls produce constant-cell
B with continuous normal flux. The implementation follows the H(curl)
magnetostatic structure described in [MFEM's magnetostatic formulation](https://mfem.org/tesla-notes/).
The matrix-free GPU operator integrates curl(Ni)·curl(Nj)/μ in each tetrahedron.

The discrete gradient G maps interior nodal potentials to edge integrals, with
boundary potentials fixed to zero. Gauge freedom is removed by adding
`alpha*G*G^T` and using diagonal-preconditioned conjugate gradients. This is an
algebraic gauge, not an A mass regularization: the closed source satisfies
`G^T*f=0` and the curl-curl operator annihilates G, so the extra term enforces
`G^T*A=0` without changing the magnetostatic B solution. Mesh topology must be a
connected, filled, contractible air box with one outer boundary. Topology checks
reject unsupported domains. The outer boundary fixes tangential A to zero,
implying zero normal B there; it is a finite-domain approximation to open space,
so air-padding sensitivity must be checked. No infinite-element boundary is used.

`em.materials` exposes μr and conductivity for copper, composite, epoxy, silicon,
FR4 and air. The initial composite μr=50 is hypothetical and linear. Silicon
conductivity is unset/configurable. **All conductivities are unused in this
milestone**, so unknown conductivity cannot silently turn into an eddy-current
prediction. Selecting any mode other than `quasistatic_no_eddy` raises an error.
There is no saturation, hysteresis, skin/proximity effect, conductive shielding,
or core-loss model. Magnetic permeability comparisons show flux redistribution,
not validated shielding performance. The assumed 220 nH sets the ideal buck
current ripple only; it does not calibrate the illustrative winding's inductance.

The configured 12 V → 1 V, 20 A, 1 MHz buck point gives duty 1/12, ripple
4.166667 A peak-to-peak, and currents 17.916667–22.083333 A. The rising interval
is 83.333 ns and the falling interval 916.667 ns. Sampling includes every
switching corner and several full periods. B changes in magnitude with current;
its direction does not reverse because the current stays positive. Time-scaled
magnetostatics has no diffusion delay or conductive phase lag.

`--pcb` reuses the existing rectangular FR4 slab from `pcb.assembly_3d`, translated
so the first module's package bottom remains z=0. It adds a labeled PCB field
region to the air-domain mesh. Passive PCB copper with μr=1 is
magnetically equivalent to air in this no-eddy-current model and is omitted;
its conductivity and any real PCB return currents must be included in a future
conductive formulation. All driving return current presently follows the
explicit modeled external conductor. This does not solve the entire three-phase
PCB's current distribution.

Outputs under `output/em/` (or `--output`) include `domain.msh`, `domain.brep`,
`domain.npz`, `mesh_report.json`, `topology.npz`, native `input.bin`,
`input.json`, `parameters.txt`, `A_edges.bin`, `B_dc.bin`, `B_series.bin`,
`current.bin`, `times.bin`, `solution.json`, `magnetic.vtu`, `report.json`,
`summary.png`, `leakage_planes.npz` and `leakage.png`. Binary doubles are
little-endian; B arrays are `(tetrahedron,xyz)` or `(time,tetrahedron,xyz)` in T.
`A_edges.bin` stores edge integrals in Wb. VTK coordinates remain mm and fields
are exported in T/mT. The input report records material assumptions and config
hash. Native flux-jump, current-continuity, independent-current linearity, and
true PCG residual audits are in `solution.json`.

The viewer uses native vector snapshots, fixed logarithmic mT colors, physical
time scrubbing/playback, a draggable cutaway/field plane, directional arrows,
optional streamlines, and independent region visibility. Hidden geometry still
contributes to the solved field. Streamlines use a display-only cell-to-node
reconstruction; raw cell B and flux audits are unchanged. Leakage maps sample
planes above and beside the module; maximum and volume-average |B| for each
material region are recorded at the configured mean current. Playback speed is
a display setting, not physical time.

```sh
# Prepare without solving; useful on hosts without a working GPU:
uv run python/em.py --prepare-only
# Reuse an existing identical geometry mesh when changing material values:
uv run python/em.py --reuse-mesh output/em --output output/em_rerun
# Loop analytical reference + mesh/domain refinement sweeps:
uv run python/em.py --benchmark --output output/em_loop
uv run python/em.py --validate --output output/em
# Same module mesh, mu_r = 1, 10, 50, 100:
uv run python/em.py --compare --reuse-mesh output/em --output output/em
# Preprocessing regression checks (no CPU magnetic solve):
uv run python -m unittest discover -s tests -p 'test_em_preprocessing.py' -v
```

`--reuse-mesh` explicitly reuses saved geometry/source paths; changing dimensions
or return coordinates requires generating a fresh mesh. Permeability comparisons
save each case's VTU/time series, `permeability/comparison.json`, and a comparison
plot. μr=1 retains the composite region's shape while replacing its magnetic
properties with air-equivalent properties. A failed case never generates a
complete comparison report.

The loop validation checks the analytical axial field
`Bz=mu0*I*R²/(2*(R²+z²)^(3/2))`, positive field orientation for counterclockwise
current, independent-current linearity, normal flux continuity, mesh refinement
and air-boundary sensitivity. The reference is an ideal thin circular loop;
finite wire-radius and polygonal-source differences contribute to reported error.
Both raw containing-cell B errors and patch-recovered point-field errors are
reported. Acceptance applies to recovered probe fields on the finest mesh and
largest domain; coarse errors remain visible. Configurable thresholds and all
case results are in `validation/report.json`. These loop checks do not establish
module/core-contrast mesh convergence or experimental device validity. In
particular, local peak B near material corners and discrete source filaments can
be sensitive to mesh/source resolution; linear high-μ predictions can exceed the
range where a real magnetic composite would remain unsaturated. Full transient
conductive A–φ MQS remains a later milestone, after device-specific geometry and
return-path validation.

Device-specific sensitivity checks can reuse a solved baseline:

```sh
uv run python/em.py --validate-module --reuse-mesh output/em --output output/em
```

`module_validation/report.json` compares a refined module mesh, larger air
domain, and doubled radial source quadrature against the baseline. It reports
core volume-average and maximum B and recovered vectors at fixed leakage probes.
Configured acceptance checks use core averages and probe changes, not singular
local peaks. This is a one-step sensitivity check, not asymptotic convergence
or validation of the assumed winding/return against a real converter.
