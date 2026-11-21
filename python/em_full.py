"""Full PCB/module edge-FEM magnetics with distributed switched conduction sources."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from geometry import ROOT, DATA
from pcb_3d import build_assembly
from em_mesh import topology, PAIRS


def dump(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def full_mesh(config, output, size=None, padding=None):
    import gmsh
    from mesh import build_cad, box
    cfg=config['em']['mesh'];h=float(size or cfg['size_mm']);fine=float(cfg['conductor_size_mm'])*(float(size)/float(cfg['size_mm']) if size is not None else 1.)
    pad=float(padding or cfg['air_padding_mm'])
    if min(h,fine,pad)<=0: raise ValueError('Mesh sizes and padding must be positive')
    assembly=build_assembly(config,json.loads((DATA/'footprint.json').read_text()))
    if len(assembly['module_placements'])!=1: raise ValueError('Full buck simulation requires one module')
    place=assembly['module_placements'][0]
    if place['rotation_z_deg']!=0: raise ValueError('Full buck mesh currently requires zero assembly rotation')
    module=assembly['module_definition'];shift=np.array(place['translation_mm'])
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',1);gmsh.model.add('full_buck');occ=gmsh.model.occ
        regions,_,_,_=build_cad(module)
        entries=[(r['name'],r['material'],(3,t)) for r in regions.values() for t in r['volume_tags']]
        drills=[]
        for v in assembly['vias']:
            x,y=np.array(v['center_xy_mm'])-shift[:2];z0=v['z_min_mm']-shift[2];z1=v['z_max_mm']-shift[2]
            drills.append((3,occ.addCylinder(x,y,z0-.01,0,0,z1-z0+.02,v['drill_radius_mm'])))
        # Fuse each electrical net before drilling; retain unrelated reference shapes separately.
        for net in ('PGND','VIN','VOUT','SIGNAL','UNASSIGNED'):
            shapes=[]
            for p in assembly['pcb_copper']:
                if p['net_hint']!=net: continue
                z=p['z_min_mm']-shift[2]
                pts=[occ.addPoint(x-shift[0],y-shift[1],z) for x,y in p['polygon_xy_mm']]
                edges=[occ.addLine(t,pts[(i+1)%len(pts)]) for i,t in enumerate(pts)]
                face=occ.addPlaneSurface([occ.addCurveLoop(edges)])
                shapes.extend(t for t in occ.extrude([(2,face)],0,0,p['z_max_mm']-p['z_min_mm']) if t[0]==3)
            if len(shapes)>1: shapes=occ.fuse(shapes[:1],shapes[1:])[0]
            if shapes and drills: shapes=occ.cut(shapes,drills,removeTool=False)[0]
            entries.extend(('pcb_'+net,'copper',t) for t in shapes if t[0]==3)
        sub=assembly['substrate']
        if sub:
            lo=np.array(sub['min_mm'])-shift;hi=np.array(sub['max_mm'])-shift
            shapes=[(3,occ.addBox(*lo,*(hi-lo)))]
            if drills: shapes=occ.cut(shapes,drills,removeTool=False)[0]
            entries.extend(('pcb_dielectric','fr4',t) for t in shapes)
        for v in assembly['vias']:
            x,y=np.array(v['center_xy_mm'])-shift[:2];z0=v['z_min_mm']-shift[2];z1=v['z_max_mm']-shift[2]
            outer=[(3,occ.addCylinder(x,y,z0,0,0,z1-z0,v['barrel_outer_radius_mm']))]
            hole=[(3,occ.addCylinder(x,y,z0-.01,0,0,z1-z0+.02,v['drill_radius_mm']))]
            entries.extend(('pcb_vias','copper',t) for t in occ.cut(outer,hole)[0])
        occ.remove(drills,recursive=True)
        # Explicit two-terminal external circuit closures at the board perimeter.
        # These are lumped supply/load fixtures, not an artificial loop around the inductor.
        def strap(name,a,b,width=.4,thickness=.1):
            a,b=np.array(a),np.array(b);axis=int(np.argmax(abs(b-a)));half=np.full(3,thickness/2)
            half[1 if axis in (0,2) else 0]=width/2
            lo=np.minimum(a,b)-half;hi=np.maximum(a,b)+half
            entries.append((name,'copper',box(dict(min_mm=lo,max_mm=hi))))
        load_points=[[0,4.7,-.01],[0,4.7,.5],[1.4,4.7,.5],[1.9,4.7,.5],
                     [5.5,4.7,.5],[5.5,-.936,.5],[3.3,-.936,.5],[3.3,-.936,-.01]]
        for a,b in zip(load_points,load_points[1:]):
            name='external_load' if a==[1.4,4.7,.5] else 'external_load_lead'
            strap(name,a,b)
        points=[[-4.4,-.24,-.01],[-4.4,-.24,.35],[-4.4,-.9,.35],[-4.4,-.9,-.01]]
        for a,b in zip(points,points[1:]): strap('external_supply',a,b,width=.3)
        occ.synchronize()
        bounds=np.array([gmsh.model.getBoundingBox(*e[2]) for e in entries]);lo=bounds[:,:3].min(0)-pad;hi=bounds[:,3:].max(0)+pad
        air=(3,occ.addBox(*lo,*(hi-lo)))
        print('Partitioning complete PCB, clips, FETs, winding, circuit fixtures and air',flush=True)
        result,mapping=occ.fragment([air],[e[2] for e in entries]);occ.synchronize()
        owners={}
        for (name,material,_),children in zip([('air','air',air)]+entries,mapping):
            for d,t in children:
                if d==3: owners[t]=(name,material)
        gmsh.model.removePhysicalGroups()
        pairs=sorted(set(owners.values()));ids={p:i+1 for i,p in enumerate(pairs)}
        regions={ids[p]:dict(name=p[0],material=p[1],volume_tags=[t for t,q in owners.items() if q==p]) for p in pairs}
        for i,r in regions.items(): gmsh.model.addPhysicalGroup(3,r['volume_tags'],i,name=r['name'])
        gmsh.option.setNumber('Mesh.ElementOrder',1);gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.option.setNumber('Mesh.MeshSizeMin',fine*.3);gmsh.option.setNumber('Mesh.MeshSizeMax',cfg['air_size_mm'])
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',8)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        surfaces=[]
        for t,(name,mat) in owners.items():
            if mat!='air': surfaces.extend(abs(s) for d,s in gmsh.model.getBoundary([(3,t)],oriented=False) if d==2)
        field=gmsh.model.mesh.field;distance=field.add('Distance');field.setNumbers(distance,'SurfacesList',list(set(surfaces)));field.setNumber(distance,'Sampling',40)
        threshold=field.add('Threshold')
        for key,val in dict(InField=distance,SizeMin=fine,SizeMax=cfg['air_size_mm'],DistMin=.15,DistMax=2).items():field.setNumber(threshold,key,val)
        field.setAsBackgroundMesh(threshold)
        print('Meshing the entire module + PCB + surrounding air',flush=True);gmsh.model.mesh.generate(3)
        tags,coords,_=gmsh.model.mesh.getNodes();order=np.argsort(tags);tags=tags[order];xyz=np.asarray(coords).reshape(-1,3)[order]
        cells=[];rid=[]
        for t,p in owners.items():
            types,_,connections=gmsh.model.mesh.getElements(3,t)
            for kind,nodes in zip(types,connections):
                if kind!=4: raise ValueError('Only linear tetrahedra supported')
                cells.append(np.searchsorted(tags,nodes).reshape(-1,4));rid.extend([ids[p]]*(len(nodes)//4))
        tet=np.concatenate(cells).astype(np.int32);rid=np.array(rid,np.int32)
        used=np.unique(tet);reverse=np.full(len(xyz),-1);reverse[used]=np.arange(len(used));xyz=xyz[used];tet=reverse[tet].astype(np.int32)
        gmsh.write(str(output/'domain.msh'));gmsh.write(str(output/'domain.brep'))
    finally: gmsh.finalize()
    determinant=np.linalg.det((xyz[tet[:,1:]]-xyz[tet[:,0,None]]).transpose(0,2,1));negative=determinant<0
    tet[negative]=tet[negative][:,[0,2,1,3]]
    report=dict(units='mm',regions=regions,module=module,assembly=assembly,source_geometry='full_buck_distributed_conduction',
                included_geometry='Full module, FET contacts, solid copper clips, all PCB top copper, vias, FR4, external supply/load fixtures, and surrounding air',
                air_bounds_mm=[lo.tolist(),hi.tolist()],size_mm=h,conductor_size_mm=fine,air_padding_mm=pad,
                assumptions=['Illustrative supply fixture at west VIN/PGND; load fixture at north VOUT/PGND',
                             'Existing raster-traced copper and hypothetical FET layout are retained; not fabrication validated'])
    np.savez(output/'domain.npz',points_mm=xyz,tetrahedra=tet,region_id=rid)
    dump(output/'mesh_report.json',report)
    return xyz*1e-3,tet,rid,report


def element_geometry(xyz,tet):
    matrix=(xyz[tet[:,1:]]-xyz[tet[:,0,None]]).transpose(0,2,1)
    inv=np.linalg.inv(matrix);grad=np.concatenate([-inv.sum(axis=1)[:,None],inv],axis=1)
    volume=abs(np.linalg.det(matrix))/6
    if np.any(volume<=0): raise ValueError('Degenerate tetrahedron')
    return grad,volume


def conduction(config,xyz,tet,rid,report,state,edges,te):
    import pyvista as pv
    grad,volume=element_geometry(xyz,tet);sigma=np.zeros(len(tet));names={int(k):v['name'] for k,v in report['regions'].items()}
    for key,region in report['regions'].items():
        name=region['name'];value=0.
        if region['material']=='copper' and name not in ('pcb_SIGNAL','pcb_UNASSIGNED'):value=5.8e7
        if name=='external_load':
            op=config['electrical']['operating_point'];load_resistance=op['vout_v']/op['iout_a']
            value=.5e-3/(load_resistance*.4e-3*.1e-3)
        if name=='external_supply' and state=='off':value=0.
        if name==('high_side_fet' if state=='on' else 'low_side_fet'):
            source=next(s for s in report['module']['heat_sources'] if s['name']==name)
            w,d,depth=np.array(source['max_mm'])-source['min_mm']
            fraction=.5 if state=='on' else .4
            resistance=config['electrical'][name.replace('_fet','')+'_rds_on_mohm']*1e-3
            value=(fraction*w*1e-3)/(resistance*d*depth*1e-6)
        sigma[rid==int(key)]=value
    conducting=np.flatnonzero(sigma>0);cells=tet[conducting];active=np.unique(cells)
    local=np.full(len(xyz),-1,np.int32);local[active]=np.arange(len(active));conn=local[cells]
    g=grad[conducting];weight=sigma[conducting]*volume[conducting]
    matrix=coo_matrix((np.einsum('nij,nkj->nik',g,g).reshape(-1)*np.repeat(weight,16),
                      (np.repeat(conn,4,axis=1).ravel(),np.tile(conn,(1,4)).ravel())),shape=(len(active),len(active))).tocsr()
    _,components=connected_components(matrix,directed=False)
    emf=np.zeros((len(tet),3));wind=next(int(k) for k,v in report['regions'].items() if v['name']=='copper_winding')
    selected=np.flatnonzero(rid==wind)
    caps=report['module']['capsules'];mid=np.array([(np.array(c['a_mm'])+c['b_mm'])/2 for c in caps])*1e-3
    tangent=np.array([np.array(c['b_mm'])-c['a_mm'] for c in caps]);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    for i,c in enumerate(caps):
        if c['name'].startswith('lead_1_'): tangent[i]*=-1
    nearest=cKDTree(mid).query(xyz[tet[selected]].mean(1))[1];emf[selected]=tangent[nearest]
    force=weight[:,None]*np.einsum('nij,nj->ni',g,emf[conducting]);rhs=np.bincount(conn.ravel(),force.ravel(),minlength=len(active))
    driven=np.unique(components[local[np.unique(tet[selected])]])
    if len(driven)!=1: raise ValueError('Winding has disconnected conductor components')
    component=driven[0];nodes=np.flatnonzero(components==component)
    touched={names[int(r)] for r in np.unique(rid[conducting[np.any(np.isin(conn,nodes),axis=1)]])}
    required={'external_load','pcb_PGND','buck_clip_SW','buck_clip_VOUT', 'high_side_fet' if state=='on' else 'low_side_fet'}
    if state=='on':required.update(('external_supply','pcb_VIN'))
    if not required<=touched: raise ValueError(f'{state}: open buck circuit; missing {required-touched}')
    free=nodes[1:];phi=np.zeros(len(active));print(f'{state}: solving distributed conduction on {len(nodes)} connected nodes',flush=True)
    system=matrix[free][:,free].tocsc();lu=splu(system)
    phi[free]=lu.solve(rhs[free])
    for _ in range(4):
        residual=rhs[free]-system@phi[free]
        phi[free]+=lu.solve(residual)
    if not np.isfinite(phi).all():raise ValueError('Conduction solve failed')
    J=np.zeros((len(tet),3));J[conducting]=sigma[conducting,None]*(emf[conducting]-np.einsum('ni,nij->nj',phi[conn],g))
    # Normalize by actual flux through the switch-node clip into terminal 1.
    grid=pv.UnstructuredGrid(np.c_[np.full(len(tet),4),tet].ravel(),np.full(len(tet),pv.CellType.TETRA),xyz*1e3)
    grid.cell_data['J']=J;grid.cell_data['region_id']=rid
    cut=grid.slice(normal=(0,0,1),origin=(0,-1.6,1.05)).compute_cell_sizes(length=False,area=True,volume=False)
    points=cut.cell_centers().points;mask=(abs(points[:,0])<.2)&(abs(points[:,1]+1.6)<.2)
    current=float(np.sum(cut.cell_data['J'][mask,2]*cut.cell_data['Area'][mask])*1e-6)
    print(f'{state}: source current before normalization {current:.9g} A',flush=True)
    if abs(current)<1e-12:raise ValueError(f'{state}: no current reaches the inductor input')
    J/=current;phi/=current
    # Exact volume integral of lowest-order Whitney edge basis against cell-constant J.
    local_rhs=volume[:,None]*np.einsum('nej,nj->ne',grad[:,PAIRS[:,1]]-grad[:,PAIRS[:,0]],J)/4
    edge_rhs=np.bincount((abs(te)-1).ravel(),(local_rhs*np.sign(te)).ravel(),minlength=len(edges))
    divergence=np.bincount(edges[:,0],-edge_rhs,minlength=len(xyz))+np.bincount(edges[:,1],edge_rhs,minlength=len(xyz))
    raw_residual=float(abs(divergence).max())
    correction_relative=0.
    if raw_residual>1e-10:
        # Remove only numerical gradient contamination of the distributed source.
        # Use conducting edges so balancing cannot inject current into air/dielectric.
        eids=np.unique(abs(te[conducting]).ravel())-1
        ev=local[edges[eids]]
        vals=np.tile([1.,-1.,-1.,1.],len(eids))
        lap=coo_matrix((vals,(np.repeat(ev,2,axis=1).ravel(),np.tile(ev,(1,2)).ravel())),shape=matrix.shape).tocsr()
        labels=connected_components(lap,directed=False)[1]
        anchors=np.unique(labels,return_index=True)[1]
        free_balance=np.setdiff1d(np.arange(len(active)),anchors)
        q=np.zeros(len(active));q[free_balance]=splu(lap[free_balance][:,free_balance].tocsc()).solve(divergence[active][free_balance])
        correction=q[ev[:,1]]-q[ev[:,0]]
        correction_relative=float(np.linalg.norm(correction)/np.linalg.norm(edge_rhs))
        if correction_relative>1e-4:raise ValueError(f'{state}: excessive current-continuity correction {correction_relative}')
        edge_rhs[eids]-=correction
        divergence=np.bincount(edges[:,0],-edge_rhs,minlength=len(xyz))+np.bincount(edges[:,1],edge_rhs,minlength=len(xyz))
    residual=float(abs(divergence).max())
    if residual>1e-8:raise ValueError(f'{state}: distributed source violates nodal continuity: {residual}')
    np.savez(Path(report['_output'])/f'current_{state}.npz',J_A_m2_per_A=J,potential_V_per_A=np.bincount(active,phi,minlength=len(xyz)))
    return J,edge_rhs,dict(state=state,conductor_nodes=len(nodes),source_continuity_relative=residual,raw_continuity_residual=raw_residual,continuity_correction_relative=correction_relative,
                           driven_regions=sorted(touched),unscaled_inductor_current_A=current)


def audit_current_paths(points_mm,tet,J,state):
    """Independent cross-section checks reject dangling or incorrectly switched circuits."""
    import pyvista as pv
    grid=pv.UnstructuredGrid(np.c_[np.full(len(tet),4),tet].ravel(),np.full(len(tet),pv.CellType.TETRA),points_mm)
    grid.cell_data['J']=J
    probes=[('load',(1,0,0),(1.65,4.7,.5),(1,2),1.),
            ('supply',(0,1,0),(-4.4,-.55,.35),(0,2),1. if state=='on' else 0.),
            ('high_side',(0,0,1),(-.55,.65,.70),(0,1),1. if state=='on' else 0.),
            ('low_side',(1,0,0),(1.45,.20,.60),(1,2),0. if state=='on' else -1.)]
    result={}
    for name,normal,point,axes,expected in probes:
        cut=grid.slice(normal=normal,origin=point).compute_cell_sizes(length=False,area=True,volume=False)
        center=cut.cell_centers().points
        mask=np.all(abs(center[:,axes]-np.array(point)[list(axes)])<.23,axis=1)
        actual=float(np.sum((cut.cell_data['J'][mask]@np.array(normal))*cut.cell_data['Area'][mask])*1e-6)
        result[name]=dict(flux_A_per_A=actual,expected_A_per_A=expected,error_A_per_A=abs(actual-expected))
        if abs(actual-expected)>.05:raise ValueError(f'{state}: {name} branch carries {actual:.6g} A per inductor A; expected {expected}')
    return result


def buck_waveform(config):
    op=config['electrical']['operating_point'];frequency=op['switching_frequency_khz']*1e3
    vin,vout,mean=op['vin_v'],op['vout_v'],op['iout_a'];inductance=config['inductor']['nominal_inductance_h']
    if not all(np.isfinite(x) and x>0 for x in (frequency,vin,vout,mean,inductance)) or vin<=vout:
        raise ValueError('A positive step-down operating point is required')
    duty=vout/vin;ripple=vout*(1-duty)/(inductance*frequency)
    if mean<=ripple/2:raise ValueError('This buck waveform requires continuous conduction')
    anim=config['em']['animation'];periods=anim['periods'];samples=anim['samples_per_period']
    if not isinstance(periods,int) or not isinstance(samples,int) or periods<1 or samples<4:
        raise ValueError('Animation requires integer periods>=1 and samples_per_period>=4')
    phase=np.unique(np.r_[np.arange(periods*samples+1)/samples,np.arange(periods)+duty]);times=phase/frequency
    frac=np.mod(phase,1);on=frac<duty-1e-12
    current=mean-ripple/2+np.where(on,ripple*frac/duty,ripple*(1-(frac-duty)/(1-duty)))
    return times,on,current,frequency,duty,ripple,mean


def write_native(config,output,xyz,tet,rid,report,edges,te,fixed,faces,owners,rhs):
    mur=np.ones(len(tet));sigma=np.zeros(len(tet))
    for key,r in report['regions'].items():mur[rid==int(key)]=config['em']['materials'][r['material']]['relative_permeability']
    with (output/'input.bin').open('wb') as f:
        f.write(b'POLEMG01');f.write(struct.pack('<6i',len(xyz),len(tet),len(edges),0,len(faces),2))
        arrays=[(xyz,'f8'),(tet,'i4'),(edges,'i4'),(te,'i4'),(fixed,'i4'),(mur,'f8'),(sigma,'f8'),
                (np.empty(0),'i4'),(np.empty((0,4)),'f8'),(np.empty((0,4)),'f8'),(np.empty(0),'f8'),
                (np.array([0.,1e-6]),'f8'),(faces,'i4'),(owners,'i4')]
        for a,kind in arrays:np.asarray(a,dtype='<'+kind).tofile(f)
    np.asarray(rhs,dtype='<f8').tofile(output/'rhs_edges.bin')
    op=config['electrical']['operating_point'];pars=dict(current_a=1.,vin_v=op['vin_v'],vout_v=op['vout_v'],
        inductance_h=config['inductor']['nominal_inductance_h'],frequency_hz=op['switching_frequency_khz']*1e3,**config['em']['solver'])
    (output/'parameters.txt').write_text(''.join(f'{k} {v:.17g}\n' for k,v in pars.items()))


def run(config,args):
    import shutil
    import pyvista as pv
    if config['em']['mode']!='quasistatic_no_eddy':raise ValueError('Full-domain solver implements quasistatic_no_eddy only')
    if config['em']['boundary']['type']!='zero_tangential_A':raise ValueError('Only zero_tangential_A outer boundary is implemented')
    buck_waveform(config)
    output=args.output
    target=output.resolve();root=(ROOT/'output').resolve()
    if target==root or not target.is_relative_to(root):
        raise ValueError('EM overwrite destination must be a subdirectory of the project output directory')
    output.mkdir(parents=True,exist_ok=True)
    reused=None
    if args.reuse_mesh:
        source=args.reuse_mesh
        bundle=np.load(source/'domain.npz')
        reused=(bundle['points_mm']*1e-3,bundle['tetrahedra'],bundle['region_id'],json.loads((source/'mesh_report.json').read_text()))
        if reused[3]['source_geometry']!='full_buck_distributed_conduction':raise ValueError('Only a full-buck mesh can be reused')
    # Explicit overwrite request: remove only generated files inside the EM result directory.
    keep={'domain.npz','domain.msh','domain.brep','mesh_report.json'} if reused else set()
    for child in output.iterdir():
        if child.name in keep:continue
        if child.is_dir():shutil.rmtree(child)
        else:child.unlink()
    dump(output/'run_status.json',dict(status='meshing',model='full_buck_distributed_conduction'))
    try:
        xyz,tet,rid,report=reused if reused else full_mesh(config,output,args.size,args.padding)
        if reused:
            np.savez(output/'domain.npz',points_mm=xyz*1e3,tetrahedra=tet,region_id=rid)
        report['_output']=str(output.resolve());dump(output/'mesh_report.json',report)
        edges,te,fixed,faces,owners=topology(xyz,tet)
        np.savez(output/'topology.npz',edges=edges,tet_edges=te,fixed=fixed,faces=faces,owners=owners)
        if args.prepare_only:
            dump(output/'run_status.json',dict(status='prepared',model='full_buck_distributed_conduction',tetrahedra=len(tet)))
            return
        fields={};currents={};audits={}
        for state in ('on','off'):
            dump(output/'run_status.json',dict(status='solving',switch_state=state))
            J,rhs,audit=conduction(config,xyz,tet,rid,report,state,edges,te)
            audit['branch_currents']=audit_current_paths(xyz*1e3,tet,J,state)
            case=output/state
            if case.exists():shutil.rmtree(case)
            case.mkdir();write_native(config,case,xyz,tet,rid,report,edges,te,fixed,faces,owners,rhs)
            subprocess.run([str(args.executable.resolve()),str(case.resolve())],check=True)
            solution=json.loads((case/'solution.json').read_text())
            if not solution['converged']:raise RuntimeError(f'{state} magnetic solve did not converge')
            fields[state]=np.fromfile(case/'B_dc.bin',dtype='<f8').reshape(-1,3);currents[state]=J;audits[state]=dict(conduction=audit,magnetic=solution)
        times,on,current,frequency,duty,ripple,mean=buck_waveform(config)
        anim=config['em']['animation']
        dump(output/'current_path_audit.json',{state:a['conduction']['branch_currents'] for state,a in audits.items()})
        Bmean=mean*(duty*fields['on']+(1-duty)*fields['off'])
        with (output/'B_series.bin').open('wb') as f:
            for flag,amps in zip(on,current):np.asarray(fields['on' if flag else 'off']*amps,dtype='<f8').tofile(f)
        for filename,data in [('times.bin',times),('current.bin',current),('B_dc.bin',Bmean)]:np.asarray(data,dtype='<f8').tofile(output/filename)
        np.savez(output/'switch_states.npz',high_side_on=on,inductor_current_A=current,times_s=times)
        grid=pv.UnstructuredGrid(np.c_[np.full(len(tet),4),tet].ravel(),np.full(len(tet),pv.CellType.TETRA),xyz*1e3)
        grid.cell_data['region_id']=rid;grid.cell_data['B_T']=Bmean;grid.cell_data['B_mT']=Bmean*1e3;grid.cell_data['B_magnitude_mT']=np.linalg.norm(Bmean,axis=1)*1e3
        for state in fields:
            grid.cell_data['B_'+state+'_T_per_A']=fields[state];grid.cell_data['J_'+state+'_A_m2']=currents[state]*mean
        grid.save(output/'magnetic.vtu')
        limits=['Linear isotropic quasistatics: no eddy currents, saturation, hysteresis, displacement currents or frequency-dependent losses',
                'Two distributed conduction states use imposed CCM inductor current; no circuit/device transient solve',
                'Instantaneous ideal switching omits dead time and switching-edge waveforms',
                'External source/load fixtures and FET conductivity/contact geometry are assumed',
                'Reference copper overlaps and unverified contacts can affect current distribution; no hardware validation',
                'Current-source continuity is numerically equilibrated on conductor edges only; correction is audited',
                'Mesh and outer-domain convergence have not been established']
        meta=dict(mode='quasistatic_no_eddy',model='full_buck_distributed_conduction',nodes=len(xyz),tetrahedra=len(tet),edges=len(edges),frames=len(times),
            mean_current_a=mean,frequency_hz=frequency,duty_cycle=duty,ripple_peak_to_peak_a=ripple,animation=anim,limitations=limits,
            series_maximum_B_mT=max(np.linalg.norm(b,axis=1).max() for b in fields.values())*current.max()*1e3,
            config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
        dump(output/'input.json',meta);dump(output/'report.json',dict(**meta,audits=audits,included_geometry=report['included_geometry']))
        dump(output/'solution.json',dict(converged=True,mode='quasistatic_no_eddy',eddy_currents_included=False,switch_states=audits))
        dump(output/'run_status.json',dict(status='converged',model=meta['model']))
        print(f'Full buck complete: {len(tet)} tetrahedra, {len(times)} frames',flush=True)
    except Exception as error:
        dump(output/'run_status.json',dict(status='failed',error=str(error),model='full_buck_distributed_conduction'))
        raise
