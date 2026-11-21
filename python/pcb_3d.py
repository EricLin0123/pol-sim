"""Build/view a preliminary 3D PCB assembly in mm; optional OpenCASCADE export."""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
import yaml
from geometry import ROOT, DATA, build_model
from pcb import draw_reference
from pins import prism_faces


def build_assembly(config, footprint):
    p = config['pcb']['layout_parameters']
    opts = config['pcb']['assembly_3d']
    thickness = float(opts['copper_thickness_mm'])
    dielectric = opts.get('dielectric_thickness_mm')
    z = float(opts['top_copper_bottom_z_mm'])
    if not math.isfinite(thickness) or thickness <= 0 or not math.isfinite(z):
        raise ValueError('Positive finite copper thickness and finite z are required')
    if dielectric is not None and (not math.isfinite(dielectric) or dielectric <= 0):
        raise ValueError('Dielectric thickness must be positive or null')
    n = p['modules']; pitch = float(p['module_pitch_mm'])
    fw, fh = config['geometry']['module_footprint_mm']
    margin = float(p['board_margin_mm'])
    if isinstance(n, bool) or not isinstance(n,int) or n < 1:
        raise ValueError('Module count must be a positive integer')
    if not all(math.isfinite(v) and v > 0 for v in (fw,fh,pitch)) or pitch < fw or not math.isclose(fw,fh):
        raise ValueError('Reference needs square positive footprints and nonoverlapping pitch')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Margin must be finite and nonnegative')
    scale = fw/312
    centers = [[346*scale+i*pitch,379*scale] for i in range(n)]
    colors = p['colors']; copper=[]; vias=[]
    def points(g, pts): return [[g[0]+x*scale,g[1]-y*scale] for x,y in pts]
    def poly(g, pts, fill, *args, **kwargs):
        if fill == 'none': return
        net = next((k.upper() for k in ('pgnd','vin','vout','signal') if colors[k] == fill), 'UNASSIGNED')
        copper.append(dict(name=f'pcb_copper_{len(copper)+1}',kind='polygon_prism',material='copper',
                           net_hint=net,color=fill,polygon_xy_mm=points(g,pts),z_min_mm=z,z_max_mm=z+thickness))
    def rect(g,x,y,w,h,fill,*args,**kwargs):
        # Preserve rounded reference pad corners as an explicit polygon approximation.
        radius=kwargs.get('rx',args[2] if len(args)>2 else 0)
        if radius:
            pts=[]
            for cx,cy,start in ((x+w-radius,y+radius,-90),(x+w-radius,y+h-radius,0),
                                (x+radius,y+h-radius,90),(x+radius,y+radius,180)):
                pts.extend((cx+radius*math.cos(t),cy+radius*math.sin(t)) for t in np.linspace(math.radians(start),math.radians(start+90),9))
        else: pts=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
        poly(g,pts,fill)
    ground_via_sites = {(0,-60),(-53,49),(0,49),(53,49),(-26,77),(26,77)}
    ground_via_sites.update((x,y) for x in (-166,164) for y in (-282,-246,-211))
    ground_via_sites.update((x,y) for x in (-262,264) for y in (56,91))
    def via(g,x,y,*args):
        drill=float(p['via_drill_diameter_mm']); outer=float(p['via_outer_diameter_mm'])
        if not all(math.isfinite(v) for v in (drill,outer)) or not 0 < drill < outer or drill+2*thickness > outer:
            raise ValueError('Via drill plus twice the copper wall must fit inside its outer diameter')
        vias.append(dict(name=f'via_{len(vias)+1}',net_hint='PGND' if (x,y) in ground_via_sites else 'UNASSIGNED',center_xy_mm=points(g,[(x,y)])[0],
                         drill_radius_mm=drill/2,barrel_outer_radius_mm=drill/2+thickness,
                         pad_outer_radius_mm=outer/2,z_min_mm=z-(dielectric or 0),z_max_mm=z+thickness))
    noop=lambda *a,**kw: None
    # Positive world y points toward the north of the plan; modules retain right-handed local coordinates.
    for cx,cy in centers: draw_reference((cx,-cy),colors,rect,poly,via,noop,noop,noop)
    winding=config.get('inductor',{}).get('winding_geometry') or {}
    module=build_model(config,footprint,demo_winding=not winding.get('topology'))
    angle=float(opts.get('module_rotation_deg',0))
    if not math.isfinite(angle): raise ValueError('Module rotation must be finite')
    placements=[dict(name=f'module_{i+1}',translation_mm=[cx,-cy,z+thickness-config['geometry']['ic_z_mm']],rotation_z_deg=angle) for i,(cx,cy) in enumerate(centers)]
    bounds=[-margin,-703*scale-margin,(n-1)*pitch+762*scale+margin,margin]
    pgnd_connections = [dict(module=place['name'], pad=pad, layer='top')
                        for place in placements
                        for pad in ('PGND_BOTTOM','L6','L7','R6','R7')]
    return dict(schema='pol_sim_pcb_assembly_v1',units='mm',coordinate_system='right-handed; x right, y north, z up',
                pgnd_connections=pgnd_connections,copper_thickness_mm=thickness,pcb_copper=copper,vias=vias,module_definition=module,module_placements=placements,
                substrate=None if dielectric is None else dict(name='pcb_dielectric',material=opts['dielectric_material'],
                    min_mm=[bounds[0],bounds[1],z-dielectric],max_mm=[bounds[2],bounds[3],z]),
                em_ready=False,assumptions=['Single top copper layer; continuous lower PGND pour joins center island and four side pins',
                    '1.0 mm FR4 thickness is user-approved preliminary assumption' if dielectric == 1 else 'Configured dielectric thickness',
                    'Illustration bounds assumed to be rectangular substrate outline',
                    'Via barrel copper wall equals PCB copper thickness; no bottom copper layer inferred',
                    'Module local axes use configured z rotation; pad alignment is unverified',
                    'No solder gap; module package bottom placed at copper top',
                    'Reference component body markers omitted: no passive terminal/body geometry supplied',
                    'Rounded PCB pads approximated by polygons'],
                unresolved=['Reference copper shapes overlap and may short nets; no validated netlist',
                    'Module pin/PCB land alignment and electrical connections',
                    'Winding/lead-to-pin electrical connectivity; illustrative winding if unconfigured',
                    'Conformal assembly fragmentation and volume mesh',
                    'Frequency-dependent EM material properties, air domain, ports and boundaries'])


def transform(points, placement):
    t=math.radians(placement['rotation_z_deg']); c,s=math.cos(t),math.sin(t)
    return np.asarray(points) @ np.array([[c,s,0],[-s,c,0],[0,0,1]]) + placement['translation_mm']


def box_faces(lo,hi):
    x,y,z=lo; X,Y,Z=hi
    v=np.array([(x,y,z),(X,y,z),(X,Y,z),(x,Y,z),(x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)])
    return [v[list(f)] for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]


def view(model, output, show):
    import pyvista as pv
    from view_3d import add as add_mesh, controls, surface
    plotter = pv.Plotter(off_screen=not show, window_size=(1600, 800))
    groups = {}
    def add(faces,color,label,alpha=1):
        return add_mesh(plotter, groups, surface(faces), color, label, alpha)
    for p in model['pcb_copper']:
        xy=p['polygon_xy_mm']; lower=np.array([[x,y,p['z_min_mm']] for x,y in xy]); upper=lower.copy();upper[:,2]=p['z_max_mm']
        add([lower[::-1],upper]+[np.array([lower[i],lower[(i+1)%len(xy)],upper[(i+1)%len(xy)],upper[i]]) for i in range(len(xy))],p['color'],'PCB copper')
    substrate=model['substrate']
    if substrate: add(box_faces(substrate['min_mm'],substrate['max_mm']),'#206849','FR4',.22)
    def tube(a,b,r):
        a,b=np.asarray(a,float),np.asarray(b,float);d=b-a;d/=np.linalg.norm(d)
        u=np.cross(d,[0,0,1] if abs(d[2])<.9 else [1,0,0]);u/=np.linalg.norm(u);v=np.cross(d,u)
        ring=np.array([r*(u*np.cos(t)+v*np.sin(t)) for t in np.linspace(0,2*np.pi,16,endpoint=False)])
        return [np.array([a+ring[i],a+ring[(i+1)%16],b+ring[(i+1)%16],b+ring[i]]) for i in range(16)]
    for via in model['vias']:
        x,y=via['center_xy_mm'];add(tube([x,y,via['z_min_mm']],[x,y,via['z_max_mm']],via['barrel_outer_radius_mm']),'#b87333','Vias')
    mod=model['module_definition']
    for place in model['module_placements']:
        for body in mod['bodies']:
            label='Module envelopes' if body['name'] in ('ic_package','inductor') else 'Die / terminals'
            color='#50697b' if body['name']=='ic_package' else '#656565' if body['name']=='inductor' else '#b87333' if body.get('material')=='copper' else '#8a74ac'
            add([transform(f,place) for f in box_faces(body['min_mm'],body['max_mm'])],color,label,.14 if label=='Module envelopes' else .8)
        for pin in mod.get('pin_solids',[]): add([transform(f,place) for f in prism_faces(pin)],'#b87333','Module pins')
        for wire in mod.get('capsules',[]): add([transform(f,place) for f in tube(wire['a_mm'],wire['b_mm'],wire['radius_mm'])],'#c88232','Winding')
    controls(plotter, groups, f"{len(model['module_placements'])}-module PCB assembly")
    output.mkdir(parents=True, exist_ok=True)
    if show:
        plotter.show(auto_close=False)
    plotter.screenshot(str(output/'assembly.png'))
    plotter.close()
    return plotter


def export_cad(model, directory):
    """Analytic OCC solids for later repair/fragmentation, not a conformal mesh."""
    import gmsh
    from mesh import build_cad
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('pcb_assembly')
        regions,_,_,_=build_cad(copy.deepcopy(model['module_definition']))
        occ=gmsh.model.occ
        # Copy partitioned module before transforming its first instance.
        original=[(3,t) for region in regions.values() for t in region['volume_tags']]
        instances=[original]+[occ.copy(original) for _ in model['module_placements'][1:]]
        groups=[]
        for place,entities in zip(model['module_placements'],instances):
            occ.rotate(entities,0,0,0,0,0,1,math.radians(place['rotation_z_deg']))
            occ.translate(entities,*place['translation_mm'])
            groups.append(dict(name=place['name'],volume_tags=[t for _,t in entities]))
        drills=[]
        for via in model['vias']:
            x,y=via['center_xy_mm'];drills.append((3,occ.addCylinder(x,y,via['z_min_mm']-.01,0,0,via['z_max_mm']-via['z_min_mm']+.02,via['drill_radius_mm'])))
        copper=[]
        for p in model['pcb_copper']:
            vertices=[occ.addPoint(x,y,p['z_min_mm']) for x,y in p['polygon_xy_mm']]
            edges=[occ.addLine(v,vertices[(i+1)%len(vertices)]) for i,v in enumerate(vertices)]
            surface=occ.addPlaneSurface([occ.addCurveLoop(edges)])
            copper.extend(v for v in occ.extrude([(2,surface)],0,0,p['z_max_mm']-p['z_min_mm']) if v[0]==3)
        # Retain separate traced solids; do not silently merge potentially shorted nets.
        if drills: copper=occ.cut(copper,drills,removeTool=False)[0]
        groups.append(dict(name='pcb_copper_unverified',volume_tags=[t for d,t in copper if d==3]))
        sub=model['substrate']
        if sub:
            lo,hi=sub['min_mm'],sub['max_mm'];shape=[(3,occ.addBox(*lo,*(b-a for a,b in zip(lo,hi))))]
            if drills: shape=occ.cut(shape,drills,removeTool=False)[0]
            groups.append(dict(name='fr4',volume_tags=[t for d,t in shape if d==3]))
        barrels=[]
        for via in model['vias']:
            x,y=via['center_xy_mm'];z0,z1=via['z_min_mm'],via['z_max_mm']
            shell=[(3,occ.addCylinder(x,y,z0,0,0,z1-z0,via['barrel_outer_radius_mm']))]
            # Annular top land joins the barrel; all have the configured copper thickness.
            pad=[(3,occ.addCylinder(x,y,z1-model['copper_thickness_mm'],0,0,model['copper_thickness_mm'],via['pad_outer_radius_mm']))]
            shell=occ.fuse(shell,pad)[0]
            hole=[(3,occ.addCylinder(x,y,z0-.01,0,0,z1-z0+.02,via['drill_radius_mm']))]
            barrels.extend(occ.cut(shell,hole)[0])
        groups.append(dict(name='via_copper',volume_tags=[t for d,t in barrels if d==3]))
        occ.remove(drills,recursive=True);occ.synchronize()
        for dim in (2,3): gmsh.model.removePhysicalGroups([(dim,t) for _,t in gmsh.model.getPhysicalGroups(dim)])
        for group in groups:
            if group['volume_tags']: gmsh.model.addPhysicalGroup(3,group['volume_tags'],name=group['name'])
        gmsh.write(str(directory/'assembly.brep'));gmsh.write(str(directory/'assembly.step'))
        (directory/'cad_regions.json').write_text(json.dumps(dict(groups=groups,conformal=False,notes=model['unresolved']),indent=2)+'\n')
    finally: gmsh.finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--footprint',type=Path,default=DATA/'footprint.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pcb_3d')
    parser.add_argument('--no-view',action='store_true')
    parser.add_argument('--cad',action='store_true',help='Also export OpenCASCADE BREP and STEP solids (can take minutes)')
    args=parser.parse_args()
    model=build_assembly(yaml.safe_load(args.config.read_text()),json.loads(args.footprint.read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'assembly.json').write_text(json.dumps(model,indent=2)+'\n')
    if args.cad: export_cad(model,args.output)
    view(model,args.output,not args.no_view)
    print(f'Wrote 3D assembly to {args.output}')


if __name__=='__main__': main()
