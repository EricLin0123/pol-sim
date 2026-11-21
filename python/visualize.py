"""View module geometry with matplotlib, or CUDA output with PyVista."""

import argparse
from pathlib import Path

import json
import yaml
from geometry import DATA, ROOT, build_model, export_model, plot_model


def mesh_view(data, scalar=None, report=None, off_screen=False):
    """Show actual element faces with a movable, whole-cell cutaway."""
    import numpy as np
    import pyvista as pv

    plotter = pv.Plotter(off_screen=off_screen,window_size=(1400,900))
    plotter.set_background('white')
    scalar = scalar or ('region_id' if 'region_id' in data.cell_data else None)
    report = report or {}
    regions = report.get('regions',{})
    ids = np.unique(data.cell_data['region_id']) if 'region_id' in data.cell_data else [None]
    actors, originals = {}, {}
    state = dict(cutaway=True,edges=True,normal=(0,1,0),origin=data.center)
    clim = data.get_data_range(scalar) if scalar else None
    for i,region_id in enumerate(ids):
        key = str(int(region_id)) if region_id is not None else 'mesh'
        grid = data.threshold([region_id,region_id],scalars='region_id') if region_id is not None else data
        originals[key] = grid
        cut = grid.clip(normal=state['normal'],origin=state['origin'],invert=False,crinkle=True)
        # Some regions may be fully outside the initial plane. Keep their actors
        # alive so moving the plane can reveal them later.
        actor = plotter.add_mesh(grid,scalars=scalar,clim=clim,cmap='tab10' if scalar=='region_id' else 'viridis',
            show_edges=True,edge_color='black',line_width=0.7,lighting=False,
            n_colors=len(ids) if scalar=='region_id' else 256,
            show_scalar_bar=bool(scalar and i==0),
            scalar_bar_args={'title':scalar or '', 'n_labels':len(ids) if scalar=='region_id' else 5},name=key)
        actor.mapper.dataset = cut
        actors[key] = actor

    def update_plane(normal,origin):
        state.update(normal=normal,origin=origin)
        for key,actor in actors.items():
            grid = originals[key]
            actor.mapper.dataset = grid.clip(normal=normal,origin=origin,invert=False,crinkle=True) if state['cutaway'] else grid
        plotter.reset_camera_clipping_range()
        plotter.render()

    plane = plotter.add_plane_widget(update_plane,normal=state['normal'],origin=state['origin'],
        bounds=data.bounds,normal_rotation=True,origin_translation=True,outline_translation=False,
        interaction_event='always',implicit=True,outline_opacity=0.15,color='gray')

    def set_edges(value):
        state['edges'] = bool(value)
        for actor in actors.values():
            actor.prop.show_edges = bool(value)
        plotter.render()

    def set_cutaway(value):
        state['cutaway'] = bool(value)
        plane.SetEnabled(bool(value))
        update_plane(state['normal'],state['origin'])

    def checkbox(label,callback,y):
        plotter.add_checkbox_button_widget(callback,value=True,position=(14,y),size=20,
            border_size=1,color_on='#356b9a',color_off='lightgray')
        plotter.add_text(label,position=(44,y),font_size=10,color='black')

    plotter.add_text('Mesh display',position=(14,865),font_size=13,color='black')
    checkbox('Element edges',set_edges,825)
    checkbox('Cutaway (drag plane)',set_cutaway,790)
    plotter.add_text('Visible regions',position=(14,745),font_size=11,color='black')
    names = {'ic_emc':'IC package EMC','copper_pins':'Copper pins','silicon_background':'Silicon die',
             'high_side_fet':'High-side FET','low_side_fet':'Low-side FET',
             'metal_composite':'Magnetic composite','copper_terminals':'Copper terminals','copper_winding':'Copper winding'}
    for i,key in enumerate(actors):
        name = regions.get(key,{}).get('name',f'Region {key}' if key!='mesh' else 'Mesh')
        def visible(value, actor=actors[key]):
            actor.SetVisibility(bool(value));plotter.render()
        checkbox(names.get(name,name),visible,705-i*33)
    plotter.add_text('Drag the plane center to cut deeper.\nDrag its normal to change direction.\nWhole tetrahedra are retained.',
                     position=(14,80),font_size=9,color='black')
    plotter.add_axes(viewport=(0.86,0.80,1.0,1.0))
    plotter.view_vector((1,-1,0.8),viewup=(0,0,1))
    plotter.reset_camera(bounds=data.bounds)
    # Retain callback references and make visibility/clip checks possible offscreen.
    plotter._mesh_view = dict(actors=actors,originals=originals,state=state,plane=plane,
                             update_plane=update_plane,set_edges=set_edges,set_cutaway=set_cutaway)
    return plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="VTK dataset, e.g. output/result.vtu")
    parser.add_argument("--scalar", help="Optional field name to color")
    parser.add_argument("--model", type=Path, help="Open a saved model instead of rebuilding the full assembly")
    parser.add_argument("--backend", help="Matplotlib backend, e.g. TkAgg")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.input is None:
        if args.model and args.model.is_file():
            model = json.loads(args.model.read_text())
        elif args.model is None:
            config = yaml.safe_load((ROOT/'config.yaml').read_text())
            winding = config.get('inductor', {}).get('winding_geometry') or {}
            model = build_model(config, json.loads((DATA/'footprint.json').read_text()),
                                demo_winding=not winding.get('topology'))
            # Viewer-only fallback must not replace canonical simulation inputs.
            export_model(model,ROOT/'output'/'visualization')
        else:
            parser.error(f"File not found: {args.model}")
        plot_model(model,args.screenshot,args.backend)
        return
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")
    import pyvista as pv
    data = pv.read(args.input)
    if args.scalar and args.scalar not in data.array_names:
        parser.error(f'Field not found: {args.scalar}')
    report_path = args.input.with_name('report.json')
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    plotter = mesh_view(data,args.scalar,report,off_screen=bool(args.screenshot))
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == "__main__":
    main()
