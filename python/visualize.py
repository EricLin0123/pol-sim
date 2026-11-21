"""View module geometry with matplotlib, or CUDA output with PyVista."""

import argparse
from pathlib import Path

import json
import yaml
from geometry import DATA, ROOT, build_model, export_model, plot_model


def mesh_view(data, scalar=None, report=None, off_screen=False, series=None):
    """Show actual element faces with a movable, whole-cell cutaway."""
    import numpy as np
    import pyvista as pv

    if not isinstance(data,pv.UnstructuredGrid):
        data = data.cast_to_unstructured_grid()
    if series is not None:
        times,temperatures=series['time_s'],series['temperature_K']
        if temperatures.shape!=(len(times),data.n_points) or len(times)==0 or not np.all(np.isfinite(temperatures)):
            raise ValueError('Time-series temperatures do not match the mesh')
        if not np.all(np.isfinite(times)) or times[0]!=0 or np.any(np.diff(times)<=0):
            raise ValueError('Invalid physical snapshot times')
        data.point_data['_thermal_node_index']=np.arange(data.n_points)
        data.point_data['temperature_K']=temperatures[0]
        data.point_data['temperature_C']=temperatures[0]-273.15
    plotter = pv.Plotter(off_screen=off_screen,window_size=(1400,900))
    plotter.set_background('white')
    scalar = scalar or ('temperature_C' if 'temperature_C' in data.point_data else
                        'temperature_difference_C' if 'temperature_difference_C' in data.point_data else
                        'region_id' if 'region_id' in data.cell_data else None)
    if series is not None and scalar not in ('temperature_C','temperature_K','mean_temperature_C','region_id','quality_minSICN'):
        raise ValueError('Time playback supports temperature and static mesh fields only')
    report = report or {}
    regions = report.get('regions',{})
    if isinstance(regions,list):
        regions = report.get('input',{}).get('regions',{})
    ids = np.unique(data.cell_data['region_id']) if 'region_id' in data.cell_data else [None]
    actors, originals = {}, {}
    state = dict(cutaway=True,slice=False,edges=True,normal=(0,1,0),origin=data.center)
    clim = data.get_data_range(scalar) if scalar else None
    if scalar=='temperature_difference_C':
        extent=max(abs(clim[0]),abs(clim[1]),1e-12)
        clim=(-extent,extent)
    if series is not None and scalar in ('temperature_C','temperature_K','mean_temperature_C'):
        offset=0 if scalar=='temperature_K' else 273.15
        clim=(float(temperatures.min()-offset),float(temperatures.max()-offset))
    for i,region_id in enumerate(ids):
        key = str(int(region_id)) if region_id is not None else 'mesh'
        grid = data.threshold([region_id,region_id],scalars='region_id') if region_id is not None else data
        originals[key] = grid
        cut = grid.clip(normal=state['normal'],origin=state['origin'],invert=False,crinkle=True)
        # Some regions may be fully outside the initial plane. Keep their actors
        # alive so moving the plane can reveal them later.
        actor = plotter.add_mesh(grid,scalars=scalar,clim=clim,cmap='tab10' if scalar=='region_id' else 'coolwarm' if scalar=='temperature_difference_C' else 'viridis',
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
            actor.mapper.dataset = (grid.slice(normal=normal,origin=origin).cast_to_unstructured_grid() if state['slice'] else
                grid.clip(normal=normal,origin=origin,invert=False,crinkle=True) if state['cutaway'] else grid)
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
        plane.SetEnabled(bool(value) or state['slice'])
        update_plane(state['normal'],state['origin'])

    def set_slice(value):
        state['slice'] = bool(value)
        plane.SetEnabled(bool(value) or state['cutaway'])
        update_plane(state['normal'],state['origin'])

    def checkbox(label,callback,y,value=True):
        plotter.add_checkbox_button_widget(callback,value=value,position=(14,y),size=20,
            border_size=1,color_on='#356b9a',color_off='lightgray')
        plotter.add_text(label,position=(44,y),font_size=10,color='black')

    title='Temperature (C)' if scalar=='temperature_C' else 'Liquid - air (C)' if scalar=='temperature_difference_C' else 'Mesh display'
    plotter.add_text(title,position=(14,865),font_size=13,color='black')
    if scalar=='temperature_C' and report.get('converged') is False:
        plotter.add_text('PARTIAL TRANSIENT' if report.get('mode')=='transient' else 'NOT CONVERGED: diagnostic iterate',position=(450,865),font_size=12,color='red')
    checkbox('Element edges',set_edges,825)
    checkbox('Cutaway (drag plane)',set_cutaway,790)
    checkbox('Slice only',set_slice,755,value=False)
    plotter.add_text('Visible regions',position=(14,710),font_size=11,color='black')
    names = {'ic_emc':'IC package EMC','copper_pins':'Copper pins','silicon_background':'Silicon die',
             'high_side_fet':'High-side FET','low_side_fet':'Low-side FET',
             'metal_composite':'Magnetic composite','copper_terminals':'Copper terminals','copper_winding':'Copper winding'}
    for i,key in enumerate(actors):
        name = regions.get(key,{}).get('name',f'Region {key}' if key!='mesh' else 'Mesh')
        def visible(value, actor=actors[key]):
            actor.SetVisibility(bool(value));plotter.render()
        checkbox(names.get(name,name),visible,670-i*33)
    plotter.add_text('Drag the plane center to cut deeper.\nDrag its normal to change direction.\nSlice only interpolates on the plane.',
                     position=(14,80),font_size=9,color='black')
    plotter.add_axes(viewport=(0.86,0.80,1.0,1.0))
    plotter.view_vector((1,-1,0.8),viewup=(0,0,1))
    plotter.reset_camera(bounds=data.bounds)
    time_controls={}
    if series is not None:
        state.update(frame=0,playing=False)
        label=plotter.add_text('Physical time: 0 s',position=(450,825),font_size=12,color='black')
        slider=None;play_widget=None
        def set_frame(index):
            index=max(0,min(int(index),len(times)-1));state['frame']=index
            frame=temperatures[index]
            for grid in originals.values():
                temperature=frame[grid.point_data['_thermal_node_index']]
                grid.point_data['temperature_K']=temperature
                grid.point_data['temperature_C']=temperature-273.15
                if 'mean_temperature_C' in grid.cell_data:
                    grid.cell_data['mean_temperature_C']=temperature[grid.cells.reshape(-1,5)[:,1:]].mean(axis=1)-273.15
            label.SetInput(f'Physical time: {times[index]:.6g} s   |   frame {index+1}/{len(times)}')
            if slider is not None:slider.GetRepresentation().SetValue(float(times[index]))
            update_plane(state['normal'],state['origin'])
        def set_playing(value):
            state['playing']=bool(value) and len(times)>1
            if state['playing'] and state['frame']==len(times)-1:set_frame(0)
            if play_widget is not None:play_widget.GetRepresentation().SetState(int(state['playing']))
            plotter.render()
        def scrub(time):
            set_playing(False)
            set_frame(np.argmin(np.abs(times-time)))
        def tick(_=None):
            if not state['playing']:return
            if state['frame']>=len(times)-1:set_playing(False)
            else:
                set_frame(state['frame']+1)
                if state['frame']==len(times)-1:set_playing(False)
        if len(times)>1:
            slider=plotter.add_slider_widget(scrub,(float(times[0]),float(times[-1])),value=float(times[0]),
                title='Physical time (s)',pointa=(.36,.16),pointb=(.93,.16),interaction_event='always',
                style='modern',fmt='%.3g',title_height=.025)
            play_widget=plotter.add_checkbox_button_widget(set_playing,value=False,position=(14,345),size=20,
                border_size=1,color_on='#356b9a',color_off='lightgray')
            plotter.add_text('Play / pause (Space)',position=(44,345),font_size=10,color='black')
            plotter.add_text('Arrow keys: previous / next frame',position=(14,310),font_size=9,color='black')
            plotter.add_key_event('space',lambda:set_playing(not state['playing']))
            def step_frame(delta):
                set_playing(False);set_frame(state['frame']+delta)
            plotter.add_key_event('Left',lambda:step_frame(-1))
            plotter.add_key_event('Right',lambda:step_frame(1))
            if not off_screen:plotter.add_timer_event(max_steps=2**31-1,duration=150,callback=tick)
        set_frame(0)
        time_controls=dict(set_frame=set_frame,set_playing=set_playing,tick=tick,slider=slider)
    # Retain callback references and make visibility/clip checks possible offscreen.
    plotter._mesh_view = dict(actors=actors,originals=originals,state=state,plane=plane,
                             update_plane=update_plane,set_edges=set_edges,set_cutaway=set_cutaway,set_slice=set_slice,
                             **time_controls)
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
    import numpy as np
    series_path=args.input if args.input.suffix=='.npz' else args.input.with_name('temperature_series.npz') if args.input.name=='temperature.vtu' else None
    series=None
    if series_path is not None and series_path.is_file():
        with np.load(series_path,allow_pickle=False) as bundle:
            if str(bundle['series_kind'])!='physical_transient':
                parser.error('Expected a physical transient temperature time series')
            series={key:bundle[key].copy() for key in ('time_s','temperature_K')}
            mesh_path=series_path.parent/str(bundle['mesh_file'])
        data=pv.read(mesh_path)
    else:
        data = pv.read(args.input)
    if args.scalar and args.scalar not in data.array_names:
        parser.error(f'Field not found: {args.scalar}')
    report_path = args.input.with_name('report.json')
    if not report_path.is_file():report_path=args.input.with_name('comparison.json')
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    plotter = mesh_view(data,args.scalar,report,off_screen=bool(args.screenshot),series=series)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == "__main__":
    main()
