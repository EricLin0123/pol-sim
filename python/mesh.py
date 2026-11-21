"""Gmsh tetrahedral preprocessing only: no thermal model or numerical solver."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import gmsh
import numpy as np
import yaml

from geometry import DATA, ROOT, build_model, export_model


def fuse(shapes):
    if len(shapes) <= 1:
        return shapes
    return gmsh.model.occ.fuse(shapes[:1], shapes[1:])[0]


def box(body):
    lo, hi = body['min_mm'], body['max_mm']
    return (3, gmsh.model.occ.addBox(*lo, *(b-a for a,b in zip(lo,hi))))


def wire_union(capsules, clearance=0):
    """Exact union of the shared model's cylinders and spherical end caps."""
    occ = gmsh.model.occ
    cylinders, endpoints = [], {}
    for c in capsules:
        a,b = c['a_mm'],c['b_mm']
        r = c['radius_mm']+clearance
        cylinders.append((3,occ.addCylinder(*a, *(y-x for x,y in zip(a,b)), r)))
        for p in (a,b):
            endpoints[tuple(p)+(r,)] = p,r
    spheres = [(3,occ.addSphere(*p,r)) for p,r in endpoints.values()]
    return fuse(cylinders+spheres)


def build_cad(model):
    occ = gmsh.model.occ
    if not model.get('pin_solids') or not model.get('heat_sources'):
        raise ValueError('Finite pin volumes and resolved die/FET geometry are required')
    if not model.get('capsules') or not model.get('composite_region'):
        raise ValueError('Winding/composite geometry is required; configure it or use default illustrative input')
    entries = []
    def add(name, material, shapes):
        entries.extend((name,material,s) for s in shapes)

    # Lowest to highest region priority. Maps after fragmentation resolve ownership.
    bodies = {b['name']:b for b in model['bodies']}
    fill = bodies['ic_package'].get('material') or 'unassigned'
    fill_name = 'ic_emc' if fill == 'ic_epoxy_molding_compound' else 'ic_fill_unassigned' if fill == 'unassigned' else 'ic_fill'
    add(fill_name,fill,[box(bodies['ic_package'])])
    add('metal_composite','metal_composite',[box(bodies['inductor'])])
    print('Building rounded copper lands...',flush=True)
    pins = []
    for pin in model['pin_solids']:
        x,y,w,h = pin['rect_mm']
        z0,z1 = pin['min_mm'][2],pin['max_mm'][2]
        s = occ.addRectangle(x,y,z0,w,h,roundedRadius=pin['corner_radius_mm'])
        pins.extend(d for d in occ.extrude([(2,s)],0,0,z1-z0) if d[0]==3)
    add('copper_pins','copper',fuse(pins))
    print('Building winding solids...',flush=True)
    clearance = model['composite_region']['wire_clearance_mm']
    if clearance > 0:
        add('insulation_unassigned','unassigned',wire_union(model['capsules'],clearance))
    for b in model['bodies']:
        if b['name'].startswith('terminal_'):
            material = b.get('material') or 'terminal_metal'
            add('copper_terminals' if material == 'copper' else 'terminals_unassigned',material,[box(b)])
    add('copper_winding','copper',wire_union(model['capsules']))
    die = bodies.get('silicon_die')
    if die is None:
        raise ValueError('A silicon die is required')
    add('silicon_background','silicon',[box(die)])
    unplaced = []
    for source in model['heat_sources']:
        if source['min_mm'] is None:
            unplaced.append(source['name'])
            if source.get('footprint_xy_mm') is not None:
                raise ValueError(f"Resolve active surface before meshing {source['name']}")
            continue
        add(source['name'],'silicon',[box(source)])
    print('Fragmenting all volumes into conformal regions...',flush=True)
    inputs = [entry[2] for entry in entries]
    result,mapping = occ.fragment(inputs[:1],inputs[1:])
    occ.synchronize()
    owners = {}
    for (name,material,_),children in zip(entries,mapping):
        for dim,tag in children:
            if dim == 3:
                owners[tag] = (name,material)
    volumes = {tag for dim,tag in result if dim==3}
    if volumes != set(owners) or volumes != {tag for _,tag in gmsh.model.getEntities(3)}:
        raise RuntimeError('Fragment ownership does not cover all CAD volumes')
    grouped = defaultdict(list)
    materials = {}
    for tag,(name,material) in owners.items():
        grouped[name].append(tag)
        materials[name]=material
    regions = {}
    for i,(name,tags) in enumerate(grouped.items(),1):
        gmsh.model.addPhysicalGroup(3,tags,i,name=f'region/{name}')
        regions[i] = dict(name=name,material=materials[name],volume_tags=tags,
                          cad_volume_mm3=sum(occ.getMass(3,t) for t in tags))
    volume_region = {tag:i for i,r in regions.items() for tag in r['volume_tags']}
    surfaces = {}
    boundary_groups = defaultdict(list)
    top = bodies['inductor']['max_mm'][2]
    for _,tag in gmsh.model.getEntities(2):
        adjacent,_ = gmsh.model.getAdjacencies(2,tag)
        adjacent = [int(v) for v in adjacent if int(v) in volume_region]
        if not adjacent:
            raise RuntimeError(f'Orphan CAD surface {tag}')
        if len(adjacent)>2:
            raise RuntimeError(f'Nonmanifold CAD surface {tag}')
        surfaces[tag] = [volume_region[v] for v in adjacent]
        if len(adjacent)==1:
            bounds = gmsh.model.getBoundingBox(2,tag)
            name = 'top_cooling_candidate' if abs(bounds[2]-top)<1e-5 and abs(bounds[5]-top)<1e-5 else 'exterior_other'
            boundary_groups[name].append(tag)
    boundary_ids = {}
    for i,(name,tags) in enumerate(boundary_groups.items(),1):
        gmsh.model.addPhysicalGroup(2,tags,i,name=f'boundary/{name}')
        boundary_ids[i]=dict(name=name,surface_tags=tags)
    return regions,surfaces,boundary_ids,unplaced


def generate_mesh(regions, size, fine_size):
    mesh = gmsh.model.mesh
    gmsh.option.setNumber('Mesh.MeshSizeMin',min(size,fine_size)/4)
    gmsh.option.setNumber('Mesh.MeshSizeMax',size)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',12)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
    gmsh.option.setNumber('Mesh.ElementOrder',1)
    gmsh.option.setNumber('Mesh.Algorithm3D',1)
    fine_volumes = [tag for r in regions.values() if r['name'] in ('high_side_fet','low_side_fet') for tag in r['volume_tags']]
    if not fine_volumes:
        raise ValueError('No FET refinement volumes found')
    faces = list({abs(tag) for dim,tag in gmsh.model.getBoundary([(3,v) for v in fine_volumes],combined=False) if dim==2})
    distance = mesh.field.add('Distance')
    mesh.field.setNumbers(distance,'SurfacesList',faces)
    mesh.field.setNumber(distance,'Sampling',100)
    threshold = mesh.field.add('Threshold')
    mesh.field.setNumber(threshold,'InField',distance)
    mesh.field.setNumber(threshold,'SizeMin',fine_size)
    mesh.field.setNumber(threshold,'SizeMax',size)
    mesh.field.setNumber(threshold,'DistMin',0)
    mesh.field.setNumber(threshold,'DistMax',size)
    mesh.field.setAsBackgroundMesh(threshold)
    print('Generating linear tetrahedra...',flush=True)
    mesh.generate(3)
    mesh.optimize('')


def export_mesh(directory, regions, surfaces, boundary_ids, model, size, fine_size, unplaced):
    mesh = gmsh.model.mesh
    node_tags,coords,_ = mesh.getNodes()
    order = np.argsort(node_tags)
    tags = np.asarray(node_tags)[order]
    points = np.asarray(coords).reshape(-1,3)[order]
    def indices(raw):
        raw = np.asarray(raw,dtype=np.uint64)
        idx = np.searchsorted(tags,raw)
        if np.any(idx>=len(tags)) or not np.array_equal(tags[idx],raw):
            raise RuntimeError('Connectivity references missing node tags')
        return idx.astype(np.int64)
    tets,element_tags,region_ids = [],[],[]
    for region_id,r in regions.items():
        for volume in r['volume_tags']:
            types,ids,connectivity = mesh.getElements(3,volume)
            if any(int(t)!=4 for t in types):
                raise RuntimeError('Expected only first-order tetrahedra')
            for et,nodes in zip(ids,connectivity):
                tets.append(indices(nodes).reshape(-1,4))
                element_tags.extend(et)
                region_ids.extend([region_id]*len(et))
    tets=np.concatenate(tets)
    region_ids=np.asarray(region_ids,dtype=np.int32)
    xyz=points[tets]
    determinants=np.einsum('ij,ij->i',np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),xyz[:,3]-xyz[:,0])
    if not len(tets) or np.any(determinants<=0):
        raise RuntimeError('Empty mesh or nonpositive tetrahedron volume')
    # Check each CAD surface triangle has exactly one exterior or two interior owners.
    all_faces=np.sort(np.concatenate([tets[:,f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]),axis=1)
    face_keys,face_counts=np.unique(all_faces,axis=0,return_counts=True)
    if np.any(face_counts>2):
        raise RuntimeError('Nonmanifold tetrahedral face')
    face_lookup={tuple(f):int(n) for f,n in zip(face_keys,face_counts)}
    boundary_map={s:i for i,b in boundary_ids.items() for s in b['surface_tags']}
    boundary,labels,interfaces,pairs=[],[],[],[]
    for surface,adjacent in surfaces.items():
        types,_,nodes=mesh.getElements(2,surface)
        if any(int(t)!=2 for t in types):
            raise RuntimeError('Expected only linear boundary triangles')
        triangles=np.concatenate([indices(n).reshape(-1,3) for n in nodes])
        for tri in triangles:
            if face_lookup.get(tuple(sorted(tri)))!=len(adjacent):
                raise RuntimeError(f'Nonconformal surface {surface}')
        if len(adjacent)==1:
            boundary.extend(triangles);labels.extend([boundary_map[surface]]*len(triangles))
        else:
            interfaces.extend(triangles);pairs.extend([adjacent]*len(triangles))
    if int(np.count_nonzero(face_counts==1))!=len(boundary):
        raise RuntimeError('Unexpected free tetrahedral faces inside the assembly')
    cad_total=sum(r['cad_volume_mm3'] for r in regions.values())
    envelope_total=sum(math.prod(b-a for a,b in zip(body['min_mm'],body['max_mm'])) for body in model['bodies'] if body['name'] in ('ic_package','inductor'))
    if not math.isclose(cad_total,envelope_total,rel_tol=1e-6):
        raise RuntimeError('Partition volume does not match assembly envelopes')
    if not math.isclose(float(np.sum(determinants)/6),cad_total,rel_tol=1e-6):
        raise RuntimeError('Mesh volume does not match the partitioned assembly')
    qualities=np.asarray(mesh.getElementQualities(element_tags,'minSICN'))
    if np.min(qualities)<=0:
        raise RuntimeError('Invalid element quality')
    for i,r in regions.items():
        r['tetrahedra']=int(np.count_nonzero(region_ids==i))
        r['mesh_volume_mm3']=float(np.sum(determinants[region_ids==i])/6)
        if not r['tetrahedra']:
            raise RuntimeError(f"Region {r['name']} has no elements")
        r['thermal_properties'] = model.get('thermal_materials',{}).get(r['material'])
    directory.mkdir(parents=True,exist_ok=True)
    gmsh.write(str(directory/'module.msh'))
    gmsh.write(str(directory/'module.brep'))
    np.savez_compressed(directory/'module.npz',points_mm=points,tetrahedra=tets,region_id=region_ids,
                        quality_minSICN=qualities,element_volume_mm3=determinants/6,
                        boundary_triangles=np.asarray(boundary,dtype=np.int64),boundary_id=np.asarray(labels,dtype=np.int32),
                        interface_triangles=np.asarray(interfaces,dtype=np.int64).reshape(-1,3),
                        interface_region_ids=np.asarray(pairs,dtype=np.int32).reshape(-1,2))
    import pyvista as pv
    grid=pv.UnstructuredGrid(np.column_stack([np.full(len(tets),4),tets]).ravel(),np.full(len(tets),10,dtype=np.uint8),points)
    grid.cell_data['region_id']=region_ids
    grid.cell_data['quality_minSICN']=qualities
    grid.save(directory/'module.vtu')
    report=dict(units='mm',connectivity='zero-based',gmsh_version=gmsh.__version__,
                nodes=len(points),tetrahedra=len(tets),regions=regions,boundaries=boundary_ids,
                cad_volume_mm3=cad_total,mesh_volume_mm3=float(np.sum(determinants)/6),
                minimum_quality_minSICN=float(np.min(qualities)),positive_tetrahedra=True,
                quality_percentiles_minSICN={str(p):float(np.percentile(qualities,p)) for p in (1,5,50)},
                quality_screening_threshold=0.05,elements_below_quality_threshold=int(np.count_nonzero(qualities<0.05)),
                quality_status='needs_improvement' if np.any(qualities<0.05) else 'screen_passed_not_convergence_test',
                conformal_interfaces_verified=True,mesh_size_mm=size,fet_size_mm=fine_size,
                illustrative_winding=model.get('inductor_detail',{}).get('illustrative',False),
                winding_composite_contact='direct' if model['composite_region']['wire_clearance_mm'] == 0 else 'clearance',
                unassigned_material_regions=[r['name'] for r in regions.values() if not r['thermal_properties']],
                unplaced_heat_sources=unplaced,thermal_ready=False,
                limitations=['Geometry-only mesh; no equations, losses, temperatures, or boundary conditions applied',
                    'IC package remainder uses the configured fill material; internal structure is not reconstructed',
                    'PCB, solder joints, cooling interface and unplaced control logic are omitted',
                    'Pin lands are unioned geometrically; original unconfirmed net assignments are not resolved',
                    'Curved CAD boundaries have a linear tetrahedral approximation; mesh convergence is not established'])
    if report['unassigned_material_regions']:
        report['limitations'].append('Missing thermal properties: '+', '.join(report['unassigned_material_regions']))
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,help='Shared module.json; default rebuilds from local config')
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output'/'mesh')
    parser.add_argument('--size',type=float,default=0.30,help='Bulk target edge length in mm')
    parser.add_argument('--fet-size',type=float,default=0.04,help='Target edge length near FETs in mm; refine for convergence')
    parser.add_argument('--view',action='store_true',help='Open the generated mesh in Gmsh')
    args=parser.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in (args.size,args.fet_size)) or args.fet_size>args.size:
        parser.error('Require 0 < --fet-size <= --size')
    if args.model:
        model=json.loads(args.model.read_text())
    else:
        config=yaml.safe_load(args.config.read_text())
        winding=config.get('inductor',{}).get('winding_geometry') or {}
        model=build_model(config,json.loads((DATA/'footprint.json').read_text()),not winding.get('topology'))
    if model.get('inductor_detail',{}).get('illustrative'):
        print('Using explicitly ILLUSTRATIVE winding; see output report for geometry omissions.',flush=True)
    start=time.monotonic()
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('module_mesh')
        regions,surfaces,boundaries,unplaced=build_cad(model)
        generate_mesh(regions,args.size,args.fet_size)
        report=export_mesh(args.output_dir,regions,surfaces,boundaries,model,args.size,args.fet_size,unplaced)
        export_model(model,args.output_dir/'input_geometry')
        print(f"Wrote {report['tetrahedra']:,} tetrahedra, {report['nodes']:,} nodes to {args.output_dir} ({time.monotonic()-start:.1f}s)")
        if report['elements_below_quality_threshold']:
            print(f"Quality diagnostic: {report['elements_below_quality_threshold']} elements have minSICN < 0.05; inspect/refine before later simulation.")
        if args.view:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()


if __name__=='__main__':
    main()
