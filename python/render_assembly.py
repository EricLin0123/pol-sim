"""Render the complete PCB/module geometry with translucent material colors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml

from geometry import DATA, ROOT
from pcb_3d import build_assembly, transform
from view_3d import box, capsule, surface
from pins import prism_faces


MATERIALS = {
    'fr4': ('FR4 substrate', '#368b70', .24),
    'copper': ('Copper traces, vias, winding & contacts', '#d58a3b', .78),
    'ic_epoxy_molding_compound': ('Epoxy molding compound', '#549bd5', .16),
    'metal_composite': ('Magnetic composite', '#a38ac9', .18),
    'silicon': ('Silicon die & FET regions', '#3658a0', .65),
    'unassigned': ('Unassigned material', '#a2aab4', .3),
}


def render(model, output):
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1100))
    plotter.set_background('white')
    plotter.ren_win.SetMultiSamples(0)
    plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0)
    used = set()

    def add(mesh, material, placement=None):
        material = material if material in MATERIALS else 'unassigned'
        used.add(material)
        _, color, opacity = MATERIALS[material]
        if placement is not None:
            mesh.points = transform(mesh.points, placement)
        plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=False,
                         ambient=.35, diffuse=.65, specular=.15)

    substrate = model['substrate']
    if substrate:
        add(box(substrate['min_mm'], substrate['max_mm']), substrate['material'])
    for copper in model['pcb_copper']:
        lower = np.array([[x, y, copper['z_min_mm']] for x, y in copper['polygon_xy_mm']])
        upper = lower.copy()
        upper[:, 2] = copper['z_max_mm']
        faces = [lower[::-1], upper] + [
            np.array([lower[i], lower[(i+1) % len(lower)],
                      upper[(i+1) % len(lower)], upper[i]]) for i in range(len(lower))]
        add(surface(faces), 'copper')
    for via in model['vias']:
        x, y = via['center_xy_mm']
        # An annular surface preserves the open drill instead of filling the via.
        add(pv.CylinderStructured(
            radius=np.array([via['drill_radius_mm'], via['barrel_outer_radius_mm']]),
            height=via['z_max_mm']-via['z_min_mm'],
            center=(x, y, (via['z_min_mm']+via['z_max_mm'])/2),
            direction=(0, 0, 1)).extract_surface(), 'copper')
    module = model['module_definition']
    for placement in model['module_placements']:
        for body in module['bodies']:
            add(box(body['min_mm'], body['max_mm']), body.get('material'), placement)
        for pin in module.get('pin_solids', []):
            add(surface(prism_faces(pin)), 'copper', placement)
        for wire in module.get('capsules', []) + module.get('interconnect_capsules', []):
            add(capsule(wire['a_mm'], wire['b_mm'], wire['radius_mm']),
                wire.get('material', 'copper'), placement)
        for source in module.get('heat_sources', []):
            if source.get('min_mm') is not None:
                add(box(source['min_mm'], source['max_mm']), 'silicon', placement)
    plotter.add_text('PCB + integrated power module', position=(45, 1035),
                     font_size=22, color='#24354b')
    plotter.add_text('Transparent material view', position=(45, 995),
                     font_size=13, color='#536479')
    plotter.add_legend([(label, color) for key, (label, color, _) in MATERIALS.items()
                        if key in used], bcolor='white', border=False,
                       loc='lower left', size=(.39, .18))
    plotter.add_axes(viewport=(.86, .02, .99, .19))
    plotter.view_vector((1, -1.35, 1.25), viewup=(0, 0, 1))
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets/pcb-module-transparent.png')
    args = parser.parse_args()
    model = build_assembly(yaml.safe_load(args.config.read_text()),
                           json.loads((DATA/'footprint.json').read_text()))
    render(model, args.output)
