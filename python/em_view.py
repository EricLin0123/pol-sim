"""PyVista magnetic-field playback. Hiding geometry never alters computed fields."""
import argparse
import json
from pathlib import Path
import numpy as np


def view(path, screenshot=None, gif=None, stride=3, streamlines=False):
    import pyvista as pv
    if stride < 1:raise ValueError('GIF stride must be positive')
    path=Path(path);directory=path.parent
    grid=pv.read(path);meta=json.loads((directory/'input.json').read_text());mesh_report=json.loads((directory/'mesh_report.json').read_text())
    times=np.fromfile(directory/'times.bin',dtype='<f8');current=np.fromfile(directory/'current.bin',dtype='<f8')
    series=np.memmap(directory/'B_series.bin',dtype='<f8',mode='r',shape=(len(times),grid.n_cells,3))
    plot=pv.Plotter(off_screen=screenshot is not None or gif is not None,window_size=(1500,950));plot.set_background('white')
    regions={int(k):r for k,r in mesh_report['regions'].items()};region_id=np.asarray(grid.cell_data['region_id'])
    state=dict(index=0,playing=False,normal=(0,1,0),origin=grid.center,glyphs=True,streamlines=streamlines,cutaway=True)
    maximum=float(np.max(grid.cell_data['B_magnitude_mT'])*np.max(np.abs(current))/meta['mean_current_a'])
    maximum=max(maximum,1e-12)
    clim=(max(maximum*1e-6,1e-8),maximum)
    physical=grid.extract_cells(np.flatnonzero(region_id!=next((i for i,r in regions.items() if r['material']=='air'),-1)))
    bounds=np.array(physical.bounds).reshape(3,2);bounds[:,0]-=.8;bounds[:,1]+=.8;roi=bounds.ravel().tolist()
    module=mesh_report.get('module')
    if module:
        core=next(b for b in module['bodies'] if b['name']=='inductor');state['origin']=(np.array(core['min_mm'])+core['max_mm'])/2
    else:state['origin']=(0,0,0)
    actors={};originals={};cell_indices={}
    for i,(rid,region) in enumerate(regions.items()):
        selected=np.flatnonzero(region_id==rid);part=grid.extract_cells(selected);cell_indices[rid]=selected;originals[rid]=part
        # Air remains in the solved field even when its volume actor is hidden.
        actors[rid]=plot.add_mesh(part,scalars='B_magnitude_mT',clim=clim,log_scale=True,cmap='viridis',opacity=.32 if region['material']!='copper' else .8,
                                  show_scalar_bar=i==0,scalar_bar_args={'title':'|B| (mT), log scale','vertical':True,'position_x':.88,'position_y':.25,'width':.08,'height':.55},name=f'region_{rid}')
        actors[rid].SetVisibility(region['material']!='air')
    empty=grid.slice(normal=state['normal'],origin=state['origin']).clip_box(roi,invert=False)
    slice_actor=plot.add_mesh(empty,scalars='B_magnitude_mT',clim=clim,log_scale=True,cmap='viridis',show_scalar_bar=False,name='field_slice')
    vector_actor=[None];stream_actor=[None]
    def refresh():
        frame=np.asarray(series[state['index']]);grid.cell_data['B_T']=frame;grid.cell_data['B_mT']=frame*1e3;grid.cell_data['B_magnitude_mT']=np.linalg.norm(frame,axis=1)*1e3
        for rid,part in originals.items():
            b=frame[cell_indices[rid]];part.cell_data['B_T']=b;part.cell_data['B_mT']=b*1e3;part.cell_data['B_magnitude_mT']=np.linalg.norm(b,axis=1)*1e3
            actors[rid].mapper.dataset=part.clip(normal=state['normal'],origin=state['origin'],crinkle=True) if state['cutaway'] else part
        sliced=grid.slice(normal=state['normal'],origin=state['origin']).clip_box(roi,invert=False);slice_actor.mapper.dataset=sliced
        if vector_actor[0] is not None:plot.remove_actor(vector_actor[0]);vector_actor[0]=None
        if state['glyphs'] and sliced.n_cells:
            centers=sliced.cell_centers();step=max(1,centers.n_points//350);sample=centers.extract_points(np.arange(0,centers.n_points,step))
            # Arrow length is fixed for direction; color retains actual field magnitude.
            glyph=sample.glyph(orient='B_T',scale=False,factor=max(grid.length*.012,.02))
            vector_actor[0]=plot.add_mesh(glyph,scalars='B_magnitude_mT',clim=clim,log_scale=True,cmap='viridis',show_scalar_bar=False,name='vectors')
        if stream_actor[0] is not None:plot.remove_actor(stream_actor[0]);stream_actor[0]=None
        if state['streamlines']:
            # Visualization reconstruction only. Raw cell vectors and their flux audit remain untouched.
            parts=[grid.extract_cells(np.flatnonzero(region_id==rid)).cell_data_to_point_data() for rid in regions]
            nodal=pv.MultiBlock(parts).combine(merge_points=False);nodal.set_active_vectors('B_T')
            streams=nodal.streamlines(vectors='B_T',source_center=state['origin'],source_radius=max(bounds[:,1]-bounds[:,0])*.15,n_points=60,max_length=grid.length*3,initial_step_length=.15,integration_direction='both',compute_vorticity=False)
            if streams.n_points:stream_actor[0]=plot.add_mesh(streams.tube(radius=grid.length*.0008),color='#e58520',name='streamlines')
        plot.add_text(f't = {times[state["index"]]*1e6:.4f} µs    I = {current[state["index"]]:.4f} A',position=(430,890),font_size=12,color='black',name='time')
        plot.render()
    def plane(normal,origin):state.update(normal=normal,origin=origin);refresh()
    plot.add_plane_widget(plane,normal=state['normal'],origin=state['origin'],bounds=roi,normal_rotation=True,outline_opacity=.15)
    def checkbox(label,callback,y,value):
        plot.add_checkbox_button_widget(callback,value=value,position=(12,y),size=18,color_on='#356b9a',color_off='lightgray');plot.add_text(label,position=(38,y),font_size=9,color='black')
    def cut(value):state['cutaway']=bool(value);refresh()
    def vectors(value):state['glyphs']=bool(value);refresh()
    def streams(value):state['streamlines']=bool(value);refresh()
    checkbox('Cutaway',cut,820,True);checkbox('Field directions',vectors,790,True);checkbox('Streamlines (display reconstruction)',streams,760,streamlines)
    for i,(rid,region) in enumerate(regions.items()):
        def visible(value,r=rid):actors[r].SetVisibility(bool(value));plot.render()
        checkbox(region['name'],visible,710-i*28,region['material']!='air')
    def set_time(t):state['index']=int(np.argmin(abs(times-float(t)*1e-6)));refresh()
    slider=plot.add_slider_widget(set_time,(0,times[-1]*1e6),value=0,title='Physical time (µs)',color='#356b9a',title_color='black',slider_width=.015,tube_width=.004,pointa=(.35,.07),pointb=(.82,.07),interaction_event='always')
    slider.GetRepresentation().GetLabelProperty().SetColor(0,0,0)
    def play(value):state['playing']=bool(value)
    checkbox('Play / pause',play,110,False)
    def tick(_=None):
        if state['playing']:
            state['index']=(state['index']+1)%len(times);slider.GetRepresentation().SetValue(times[state['index']]*1e6);refresh()
    plot.add_key_event('space',lambda:state.update(playing=not state['playing']))
    plot.add_text('Linear quasistatic field — NO EDDY CURRENTS',position=(12,920),font_size=13,color='black')
    plot.add_text('Drag plane to inspect core and air leakage.\nHide regions to reveal interior fields.\nStreamlines interpolate cell vectors for display.',position=(12,150),font_size=9,color='black')
    plot.add_axes();plot.view_isometric();plot.reset_camera(bounds=roi)
    refresh()
    if gif:
        from PIL import Image
        frames=[]
        plot.show(auto_close=False,interactive=False)
        for i in range(0,len(times),stride):
            state['index']=i;slider.GetRepresentation().SetValue(times[i]*1e6);refresh()
            frames.append(Image.fromarray(plot.screenshot(return_img=True)).convert('P',palette=Image.Palette.ADAPTIVE))
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=100,loop=0)
        plot.close()
    elif screenshot:
        plot.show(screenshot=str(screenshot),auto_close=True)
    else:
        plot.add_timer_event(max_steps=1000000,duration=int(meta.get('animation',{}).get('playback_interval_ms',80)),callback=tick);plot.show()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--screenshot',type=Path);p.add_argument('--gif',type=Path);p.add_argument('--stride',type=int,default=3);p.add_argument('--streamlines',action='store_true');args=p.parse_args();view(args.input,args.screenshot,args.gif,args.stride,args.streamlines)


if __name__=='__main__':main()
