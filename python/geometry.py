"""Shared module geometry and PyVista preview; coordinates are millimeters."""
import argparse
import copy
import json
import math
from pathlib import Path
import yaml
from inductor import add_inductor_parts, demo_parameters
from die import add_die
from pins import add_pins, prism_faces

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def build_model(config, footprint, demo_winding=False):
    g = config['geometry']
    if g['units'] != 'mm' or footprint['units'] != 'mm':
        raise ValueError('Expected millimeter geometry')
    bodies = []
    for name, size, height, z in (
        ('ic_package', g['ic_footprint_mm'], g['ic_height_mm'], g['ic_z_mm']),
        ('inductor', g['inductor_footprint_mm'], g['proposed_inductor_height_mm'], g['inductor_z_mm']),
    ):
        if size is None or height is None or z is None:
            raise ValueError(f'Missing envelope dimensions or bottom z: {name}')
        if len(size) != 2 or any(not math.isfinite(v) or v <= 0 for v in [*size, height]):
            raise ValueError(f'Invalid envelope dimensions: {name}')
        if not math.isfinite(z):
            raise ValueError('Invalid z')
        bodies.append(dict(name=name, min_mm=[-size[0]/2, -size[1]/2, z],
                           max_mm=[size[0]/2, size[1]/2, z+height], material=None))
    ic, ind = bodies
    if not math.isclose(ic['max_mm'][2], ind['min_mm'][2]):
        raise ValueError('Package top and inductor bottom must meet')
    if not math.isclose(ind['max_mm'][2]-ic['min_mm'][2], g['target_total_height_mm']):
        raise ValueError('Stack height differs from target')
    if not all(math.isclose(a,b) for a,b in zip(g['ic_footprint_mm'],
            [footprint['package']['width_mm'], footprint['package']['height_mm']])):
        raise ValueError('Footprint and package dimensions differ')
    model = dict(schema_version=2, units='mm', origin='package bottom center',
                assumptions=['Centered envelopes; no interface gap or thickness',
                             'Original footprint x/y; orientation unverified, no mirroring'],
                bodies=bodies, pad_z_mm=ic['min_mm'][2], footprint=footprint)
    for body in bodies:
        body['role'] = 'envelope'
    ic['material'] = g.get('ic_fill_material')
    ind['material'] = 'metal_composite'
    model['thermal_materials'] = copy.deepcopy(config.get('materials',{}).get('thermal',{}))
    inductor = config.get('inductor', {})
    params = inductor.get('winding_geometry')
    if demo_winding:
        supplied = params or {}
        params = demo_parameters([b-a for a,b in zip(ind['min_mm'],ind['max_mm'])])
        if supplied.get('insulation_clearance_mm') is not None:
            params['insulation_clearance_mm'] = supplied['insulation_clearance_mm']
    if params and (demo_winding or params.get('topology')):
        if inductor.get('winding_composite_contact') == 'direct' and params.get('insulation_clearance_mm') != 0:
            raise ValueError('Direct copper/composite contact requires zero insulation_clearance_mm')
        add_inductor_parts(model, params, illustrative=demo_winding,terminal_material=inductor.get('terminal_material'))
    if config.get('die'):
        add_die(model, config['die'])
    add_pins(model, g)
    pins = model.get('pin_solids', [])
    overlaps = []
    for i,a in enumerate(pins):
        for b in pins[i+1:]:
            if all(max(a['min_mm'][j],b['min_mm'][j]) < min(a['max_mm'][j],b['max_mm'][j])-1e-9 for j in range(3)):
                overlaps.append([a['name'],b['name']])
    model['meshing_status'] = dict(ready=False, pin_overlap_candidates=overlaps,
        unplaced_heat_sources=[s['name'] for s in model.get('heat_sources',[]) if s['min_mm'] is None],
        unresolved=['PCB layers, copper, vias, solder joints, and cooling interface are unspecified',
                    'Conformal Boolean partition and volume mesh have not been generated'])
    if not ic['material']:
        model['meshing_status']['unresolved'].append('IC fill material is unassigned')
    else:
        model['assumptions'].append('IC package remainder uses '+ic['material']+'; no internal leadframe or separate die-attach layer')
    if not pins:
        model['meshing_status']['unresolved'].append('Pin lands still lack volume thickness')
    if not model.get('capsules'):
        model['meshing_status']['unresolved'].append('Device winding and terminal dimensions are unspecified')
    elif model.get('inductor_detail',{}).get('illustrative'):
        model['meshing_status']['unresolved'].append('Winding and terminal dimensions are illustrative, not established')
    if model.get('composite_region',{}).get('wire_clearance_mm',0) > 0:
        model['meshing_status']['unresolved'].append('Wire insulation/clearance region has no assigned material')
    return model


def export_model(model, directory):
    lines = ['POL_SIM_GEOMETRY_V2 mm']
    for b in model['bodies']:
        lines.append('BOX ' + ' '.join(json.dumps(v) for v in
                     (b['name'], b.get('material') or 'unassigned', b.get('role', 'envelope'))) + ' ' +
                     ' '.join(map(str, b['min_mm']+b['max_mm'])))
    for c in model.get('capsules', []):
        lines.append('CAPSULE ' + json.dumps(c['name']) + ' ' + json.dumps(c['material']) + ' ' +
                     ' '.join(map(str, c['a_mm']+c['b_mm']+[c['radius_mm']])))
    if model.get('composite_region'):
        region = model['composite_region']
        lines.append('COMPOSITE ' + json.dumps(region['envelope']) + ' ' +
                     str(region['wire_clearance_mm']))
    lines.append('ILLUSTRATIVE ' + str(int(model.get('inductor_detail', {}).get('illustrative', False))))
    for s in model.get('heat_sources', []):
        xy = s['footprint_xy_mm']
        values = [s['depth_mm'], s['active_area_mm2'], s['heat_source_power_w'], int(xy is not None), *(xy or [0,0,0,0])]
        lines.append('HEAT_SOURCE ' + ' '.join(json.dumps(v) for v in
                     (s['name'],s['host'],s['surface'] or 'unresolved',s['geometry_status'])) + ' ' +
                     ' '.join('unset' if v is None else str(v) for v in values))
    for p in model['footprint']['pads']:
        x, y, w, h = p['rect_mm']
        r = p['corner_radius_mm']
        if not all(math.isfinite(v) for v in (x,y,w,h,r)) or min(w,h) <= 0 or r < 0:
            raise ValueError(f"Invalid pad: {p['name']}")
        lines.append('PAD ' + ' '.join(json.dumps(p[k]) for k in ('name','net','net_status')) +
                     ' ' + ' '.join(map(str, [x,y,w,h,r,model['pad_z_mm']])))
    for p in model.get('pin_solids', []):
        values = [*p['rect_mm'],p['corner_radius_mm'],p['min_mm'][2],p['max_mm'][2]]
        lines.append('PIN_VOLUME ' + ' '.join(json.dumps(p[k]) for k in ('name','net','net_status','material')) +
                     ' ' + ' '.join(map(str,values)))
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'module.json').write_text(json.dumps(model, indent=2)+'\n')
    (directory/'module.geom').write_text('\n'.join(lines)+'\n')


def plot_model(model, screenshot=None, backend=None):
    """Show module solids, die floorplan and pad plane with PyVista."""
    import pyvista as pv
    from view_3d import add, box, capsule, controls, surface, pad_surface
    if backend and backend.lower() not in ('pyvista', 'vtk', 'agg', 'off_screen'):
        raise ValueError('Use --backend pyvista or off_screen')
    off_screen = bool(screenshot) or backend in ('agg', 'off_screen')
    if off_screen and screenshot is None:
        screenshot = ROOT/'output'/'module.png'
    has_die = bool(model.get('die'))
    plotter = pv.Plotter(shape=(1, 3 if has_die else 2), off_screen=off_screen,
                         window_size=(1900 if has_die else 1400, 800))
    groups = {}
    detailed = bool(model.get('capsules'))
    for body in model['bodies']:
        terminal = body['name'].startswith('terminal_')
        label = ('Silicon die' if body.get('material') == 'silicon' else
                 'Terminals' if terminal else 'IC package' if body['name'] == 'ic_package' else 'Inductor')
        color = 'slategray' if label == 'Silicon die' else '#b87333' if terminal else 'royalblue' if label == 'IC package' else 'dimgray'
        opacity = .1 if body['name'] == 'inductor' and detailed else .65 if body.get('role') == 'solid' else .2
        add(plotter, groups, box(body['min_mm'], body['max_mm']), color, label, opacity)
    for wire in model.get('capsules', []):
        add(plotter, groups, capsule(wire['a_mm'], wire['b_mm'], wire['radius_mm']), '#b87333',
            'Copper leads' if wire['name'].startswith('lead_') else 'Copper winding')
    for pin in model.get('pin_solids', []):
        add(plotter, groups, surface(prism_faces(pin)), '#b87333', 'Pads / pins')
    die = next((b for b in model['bodies'] if b['name'] == 'silicon_die'), None)
    for source, color in zip(model.get('heat_sources', []), ('tomato', 'limegreen', 'purple')):
        if source.get('footprint_xy_mm') is None:
            continue
        a, b, c, d = source['footprint_xy_mm']
        mesh = (box(source['min_mm'], source['max_mm']) if source.get('min_mm') is not None else
                surface([[(a,b,die['max_mm'][2]), (c,b,die['max_mm'][2]), (c,d,die['max_mm'][2]), (a,d,die['max_mm'][2])]]))
        add(plotter, groups, mesh, color, source['name'], .8)
        if has_die:
            plotter.subplot(0, 1)
            add(plotter, groups, surface([[(a,b,0), (c,b,0), (c,d,0), (a,d,0)]]), color, source['name'])
            power = source.get('heat_source_power_w')
            plotter.add_point_labels([[(a+c)/2,(b+d)/2,0]], [f"{source['name']}\nP: {power if power is not None else 'unset'} W"], font_size=10, always_visible=True)
            plotter.subplot(0, 0)
    plotter.subplot(0, 2 if has_die else 1)
    for pad in model['footprint']['pads']:
        x, y, w, h = pad['rect_mm']
        mesh = pad_surface(pad)
        color = 'gray' if pad['net'] == 'UNASSIGNED' else 'goldenrod'
        add(plotter, groups, mesh, color, 'Pads / pins')
        plotter.add_point_labels([[x+w/2,y+h/2,0]], [f"{pad['name']}\n{pad['net']}"], font_size=8, always_visible=True)
        if not model.get('pin_solids'):
            plotter.subplot(0, 0)
            add(plotter, groups, mesh.translate((0,0,model['pad_z_mm']), inplace=False), color, 'Pads / pins')
            plotter.subplot(0, 2 if has_die else 1)
    plotter.set_background('white')
    plotter.add_text('Pad plane (original x/y)', font_size=11, color='black')
    plotter.view_xy()
    plotter.enable_parallel_projection()
    if has_die:
        plotter.subplot(0, 1)
        if die:
            lo, hi = die['min_mm'], die['max_mm']
            plotter.add_mesh(surface([[(lo[0],lo[1],-.001),(hi[0],lo[1],-.001),(hi[0],hi[1],-.001),(lo[0],hi[1],-.001)]]), color='lightgray')
        plotter.set_background('white')
        plotter.add_text('Hypothetical die floorplan', font_size=11, color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()
    plotter.subplot(0, 0)
    title = 'ILLUSTRATIVE winding' if model.get('inductor_detail', {}).get('illustrative') else 'Unverified parametric geometry'
    controls(plotter, groups, title)
    if screenshot:
        screenshot = Path(screenshot)
        screenshot.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(screenshot) if screenshot else None)
    return plotter



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--footprint',type=Path,default=DATA/'footprint.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--demo-winding',action='store_true',help='Use explicitly illustrative internals; export to output/inductor-demo by default')
    parser.add_argument('--model',type=Path,help='View saved module.json without regenerating')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--backend',help='Viewer backend: pyvista or off_screen')
    parser.add_argument('--no-view',action='store_true',help='Export only')
    args = parser.parse_args()
    if args.model and args.demo_winding:
        parser.error('--model and --demo-winding cannot be combined')
    directory = args.output_dir or (ROOT/'output'/'inductor-demo' if args.demo_winding else DATA)
    if args.model:
        model = json.loads(args.model.read_text())
    else:
        model = build_model(yaml.safe_load(args.config.read_text()),json.loads(args.footprint.read_text()),args.demo_winding)
        export_model(model,directory)
        print(f'Shared model written to {directory}')
    if not args.no_view:
        plot_model(model,args.screenshot,args.backend)


if __name__ == '__main__':
    main()
