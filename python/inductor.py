"""Parametric helical round-wire choke, not a reconstruction of device internals."""

import math


def demo_parameters(size):
    """Dimension ratios are illustrative only; never write them into device config."""
    w, d, h = size
    lateral = min(w, d)
    return dict(
        topology="round_wire_helix",
        turns=1.5,
        wire_diameter_mm=0.06 * lateral,
        centerline_radius_mm=0.27 * lateral,
        winding_height_mm=0.42 * h,
        winding_bottom_mm=0.30 * h,
        insulation_clearance_mm=0.01 * lateral,
        terminal_size_mm=[0.15 * lateral, 0.20 * lateral, 0.16 * h],
        terminal_radial_center_mm=0.40 * lateral,
        segments_per_turn=64,
    )


def add_inductor_parts(model, params, illustrative=False, terminal_material=None):
    body = next(b for b in model["bodies"] if b["name"] == "inductor")
    lo, hi = body["min_mm"], body["max_mm"]
    if params["topology"] != "round_wire_helix":
        raise ValueError("Only round_wire_helix topology is currently supported")
    required = (
        "turns",
        "wire_diameter_mm",
        "centerline_radius_mm",
        "winding_height_mm",
        "winding_bottom_mm",
        "terminal_radial_center_mm",
    )
    for key in required:
        v = params.get(key)
        if v is None or not math.isfinite(v) or v <= 0:
            raise ValueError(f"Provide a positive finite winding_geometry.{key}")
    turns = params["turns"]
    radius = params["wire_diameter_mm"] / 2
    clearance = params.get("insulation_clearance_mm")
    if clearance is None or not math.isfinite(clearance) or clearance < 0:
        raise ValueError('insulation_clearance_mm must be finite and nonnegative')
    r = params["centerline_radius_mm"]
    height = params["winding_height_mm"]
    bottom = lo[2] + params["winding_bottom_mm"]
    outer = params["terminal_radial_center_mm"]
    size = params.get("terminal_size_mm")
    segments = params.get("segments_per_turn")
    if (
        not size
        or len(size) != 3
        or any(v is None or not math.isfinite(v) or v <= 0 for v in size)
    ):
        raise ValueError("Provide three positive terminal_size_mm dimensions")
    if (
        isinstance(segments, bool)
        or not isinstance(segments, int)
        or not 12 <= segments <= 1024
    ):
        raise ValueError("segments_per_turn must be an integer between 12 and 1024")
    if turns > 100:
        raise ValueError("Preview supports at most 100 turns")
    if height / turns <= 2 * (radius + clearance):
        raise ValueError("Winding pitch must exceed insulated wire diameter")
    if r <= radius + clearance or outer <= r + radius + clearance:
        raise ValueError("Insulated winding and terminal radii are incompatible")
    if bottom - radius - clearance <= lo[2] + size[2]:
        raise ValueError("Winding must sit above terminal tops")
    n = math.ceil(turns * segments)
    points = []
    for i in range(n + 1):
        theta = math.pi + 2 * math.pi * turns * i / n
        points.append(
            [r * math.cos(theta), r * math.sin(theta), bottom + height * i / n]
        )
    terminals, leads = [], []
    for index, point in enumerate((points[0], points[-1])):
        theta = math.pi + index * 2 * math.pi * turns
        cx, cy = outer * math.cos(theta), outer * math.sin(theta)
        terminal = dict(
            name=f"terminal_{index+1}",
            material=terminal_material or "terminal_metal",
            role="solid",
            min_mm=[cx - size[0] / 2, cy - size[1] / 2, lo[2]],
            max_mm=[cx + size[0] / 2, cy + size[1] / 2, lo[2] + size[2]],
        )
        terminals.append(terminal)
        # Radial then vertical lead, ending inside its terminal for connectivity.
        leads.append([point, [cx, cy, point[2]], [cx, cy, lo[2] + size[2] / 2]])
    if all(
        max(terminals[0]["min_mm"][i], terminals[1]["min_mm"][i])
        < min(terminals[0]["max_mm"][i], terminals[1]["max_mm"][i])
        for i in range(3)
    ):
        raise ValueError("Terminal boxes overlap; change turns or terminal geometry")
    capsules = []
    for label, path in [
        ("winding", points),
        ("lead_1", leads[0]),
        ("lead_2", leads[1]),
    ]:
        for i, (a, b) in enumerate(zip(path, path[1:])):
            # End caps represent connected swept round wire; not disconnected cylinders.
            capsules.append(
                dict(
                    name=f"{label}_{i}",
                    material="copper",
                    a_mm=a,
                    b_mm=b,
                    radius_mm=radius,
                )
            )
    for c in capsules:
        # Leave insulation clearance around all wire, including leads.
        for p in (c["a_mm"], c["b_mm"]):
            if any(
                p[i] - radius - clearance < lo[i] - 1e-9
                or p[i] + radius + clearance > hi[i] + 1e-9
                for i in range(3)
            ):
                raise ValueError("Wire plus clearance leaves inductor envelope")
    for t in terminals:
        if any(
            t["min_mm"][i] < lo[i] - 1e-9 or t["max_mm"][i] > hi[i] + 1e-9
            for i in range(3)
        ):
            raise ValueError("Terminal leaves inductor envelope")
    # The shell is a difference region, not an overlapping solid box of composite.
    body.update(material='metal_composite', role="envelope")
    model["bodies"].extend(terminals)
    model["capsules"] = capsules
    model["material_priority"] = [
        "copper_capsules",
        "terminal_boxes_minus_copper",
        "metal_composite",
    ]
    for terminal in terminals:
        terminal["subtract_capsules"] = [c["name"] for c in capsules]
    model["composite_region"] = dict(
        name="metal_composite",
        envelope="inductor",
        subtract_boxes=[t["name"] for t in terminals],
        subtract_capsules=[c["name"] for c in capsules],
        wire_clearance_mm=clearance,
        clearance_material=None,
    )
    model["inductor_detail"] = dict(
        parameters=params,
        illustrative=illustrative,
        terminal_material=terminal_material or "terminal_metal",
        composite_material='metal_composite',
        winding_composite_contact='direct' if clearance == 0 else 'clearance',
        validated_against_reference=False,
        inductance_calibrated=False,
    )
    model["assumptions"].extend(
        [
            "Unverified round-wire helix with radial/vertical leads; no pin-to-terminal mapping",
            "Composite = envelope minus terminal boxes and clearance-expanded wire capsules",
            ('Direct copper/composite contact; no insulation region' if clearance == 0 else
             'Clearance region has no assigned insulation material'),
            "Geometry is not calibrated to nominal inductance; no manufacturing validation",
        ]
    )
