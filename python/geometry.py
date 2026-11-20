"""Shared module geometry and matplotlib preview; coordinates are millimeters."""
import argparse
import json
import math
from pathlib import Path
import yaml
from inductor import add_inductor_parts, demo_parameters

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
                             'Pads are surfaces at package bottom; thickness unknown',
                             'Original footprint x/y; orientation unverified, no mirroring'],
                bodies=bodies, pad_z_mm=ic['min_mm'][2], footprint=footprint)
    for body in bodies:
        body['role'] = 'envelope'
    params = config.get('inductor', {}).get('winding_geometry')
    if demo_winding:
        params = demo_parameters([b-a for a,b in zip(ind['min_mm'],ind['max_mm'])])
    if params and (demo_winding or params.get('topology')):
        add_inductor_parts(model, params, illustrative=demo_winding)
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
    for p in model['footprint']['pads']:
        x, y, w, h = p['rect_mm']
        r = p['corner_radius_mm']
        if not all(math.isfinite(v) for v in (x,y,w,h,r)) or min(w,h) <= 0 or r < 0:
            raise ValueError(f"Invalid pad: {p['name']}")
        lines.append('PAD ' + ' '.join(json.dumps(p[k]) for k in ('name','net','net_status')) +
                     ' ' + ' '.join(map(str, [x,y,w,h,r,model['pad_z_mm']])))
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'module.json').write_text(json.dumps(model, indent=2)+'\n')
    (directory/'module.geom').write_text('\n'.join(lines)+'\n')


def plot_model(model, screenshot=None, backend=None):
    import importlib.util
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d

    try:
        if screenshot:
            plt.switch_backend('Agg')
        elif backend:
            plt.switch_backend(backend)
        elif matplotlib.get_backend().lower() == 'agg':
            if importlib.util.find_spec('PySide6'):
                plt.switch_backend('QtAgg')
            else:
                import _tkinter
                if not getattr(_tkinter, '__file__', None):
                    raise RuntimeError('This uv Python has incompatible Tk support. Install GUI support: uv sync --extra gui')
                plt.switch_backend('TkAgg')
        fig = plt.figure(figsize=(12,6))
    except Exception as error:
        # Only backend/window initialization is guarded, not model rendering.
        if backend:
            raise RuntimeError(f'Cannot open matplotlib backend {backend}: {error}') from error
        plt.switch_backend('Agg')
        screenshot = screenshot or ROOT/'output'/'module.png'
        print(f'Interactive display unavailable: {error}')
        fig = plt.figure(figsize=(12,6))
    if fig.canvas.manager is None or fig.canvas.required_interactive_framework is None:
        if matplotlib.get_backend().lower() == 'agg':
            screenshot = screenshot or ROOT/'output'/'module.png'
    ax = fig.add_subplot(121, projection='3d')
    detailed = bool(model.get('capsules'))
    for b in model['bodies']:
        color = ('silver' if b.get('material') == 'terminal_metal' else
                 'royalblue' if b['name'] == 'ic_package' else 'dimgray' if detailed else 'orange')
        x0,y0,z0 = b['min_mm']
        x1,y1,z1 = b['max_mm']
        vertices = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                    (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        faces = [[vertices[i] for i in face] for face in
                 ((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]
        alpha = 0.10 if b['name'] == 'inductor' and detailed else 0.65 if b.get('role') == 'solid' else 0.20
        ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor='black', alpha=alpha))
    if detailed:
        import numpy as np
        faces = []
        # Draw actual-radius tubes, rather than screen-space thick lines.
        for c in model['capsules']:
            a, b = np.array(c['a_mm']), np.array(c['b_mm'])
            direction = b-a
            direction /= np.linalg.norm(direction)
            ref = np.array([0.,0.,1.]) if abs(direction[2]) < 0.9 else np.array([1.,0.,0.])
            u = np.cross(direction,ref); u /= np.linalg.norm(u)
            v = np.cross(direction,u)
            ring = np.array([c['radius_mm']*(u*np.cos(t)+v*np.sin(t)) for t in np.linspace(0,2*np.pi,12,endpoint=False)])
            for i in range(12):
                j = (i+1)%12
                faces.append([a+ring[i],a+ring[j],b+ring[j],b+ring[i]])
        ax.add_collection3d(Poly3DCollection(faces,facecolor='#b87333',edgecolor='none'))
        ax.text2D(0.02,0.95,'Copper winding / silver terminals\nTransparent composite boundary',transform=ax.transAxes)
    bottom = fig.add_subplot(122)
    for p in model['footprint']['pads']:
        x,y,w,h = p['rect_mm']
        radius = min(p['corner_radius_mm'],w/2,h/2)
        style = f'round,pad=0,rounding_size={radius}'
        color = 'gray' if p['net']=='UNASSIGNED' else 'goldenrod'
        for axis in (bottom,ax):
            patch = FancyBboxPatch((x,y),w,h,boxstyle=style,facecolor=color,edgecolor='black')
            axis.add_patch(patch)
            if axis is ax:
                pathpatch_2d_to_3d(patch,z=model['pad_z_mm'],zdir='z')
        bottom.text(x+w/2,y+h/2,f"{p['name']}\n{p['net']}",ha='center',va='center',fontsize=5)
    lo = [min(b['min_mm'][i] for b in model['bodies']) for i in range(3)]
    hi = [max(b['max_mm'][i] for b in model['bodies']) for i in range(3)]
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
           xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title='Assumed module envelopes')
    ax.set_box_aspect([b-a for a,b in zip(lo,hi)])
    bottom.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',
               xlabel='x (mm)',ylabel='y (mm)',title='Pad plane (original x/y; no mirroring)')
    label = 'ILLUSTRATIVE winding — not reference-validated' if model.get('inductor_detail',{}).get('illustrative') else 'Unverified parametric geometry'
    fig.suptitle(label + ' — pin assignments retained as supplied')
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.85, wspace=0.25)
    if screenshot:
        screenshot.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(screenshot,dpi=180)
        plt.close(fig)
        print(f'Saved module preview: {screenshot.resolve()}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--footprint',type=Path,default=DATA/'footprint.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--demo-winding',action='store_true',help='Use explicitly illustrative internals; export to output/inductor-demo by default')
    parser.add_argument('--model',type=Path,help='View saved module.json without regenerating')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--backend',help='Matplotlib backend, e.g. TkAgg or QtAgg')
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
