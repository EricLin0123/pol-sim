"""MQS geometry, tetrahedral topology, and closed source-path preprocessing only."""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from geometry import build_model, ROOT, DATA

PAIRS=np.array([(i,j) for i in range(4) for j in range(i+1,4)])
FACES=np.array([(1,2,3),(0,2,3),(0,1,3),(0,1,2)])


def source_path(config, footprint):
    source=config['em']['source']
    use_demo=source['use_illustrative_winding']
    model=build_model(config,footprint,demo_winding=use_demo)
    capsules=model.get('capsules',[])
    if not capsules: raise ValueError('Configure actual winding geometry or explicitly authorize illustrative winding')
    winding=[c for c in capsules if c['name'].startswith('winding_')]
    lead1=[c for c in capsules if c['name'].startswith('lead_1_')]
    lead2=[c for c in capsules if c['name'].startswith('lead_2_')]
    if not all((winding,lead1,lead2)): raise ValueError('Missing winding-to-terminal lead geometry')
    def chain(parts):
        for a,b in zip(parts,parts[1:]):
            if not np.allclose(a['b_mm'],b['a_mm'],atol=1e-10,rtol=0): raise ValueError('Disconnected source capsules')
        return [parts[0]['a_mm']]+[c['b_mm'] for c in parts]
    path=list(reversed(chain(lead1)))+chain(winding)[1:]+chain(lead2)[1:]
    ret=source.get('return_path_mm')
    if ret is None: raise ValueError('em.source.return_path_mm must explicitly close terminal 2 to terminal 1; no return path is inferred')
    ret=np.asarray(ret,float)
    if ret.ndim!=2 or ret.shape[1]!=3 or len(ret)<2 or not np.isfinite(ret).all(): raise ValueError('Return path must be a finite Nx3 polyline in mm')
    if not np.allclose(ret[0],path[-1],atol=1e-8,rtol=0) or not np.allclose(ret[-1],path[0],atol=1e-8,rtol=0): raise ValueError('Return path endpoints do not join the winding terminal centers')
    radii={c['radius_mm'] for c in capsules}
    if len(radii)!=1: raise ValueError('Initial source bundle requires uniform wire radius')
    radius=radii.pop()
    return model,np.array(path+ret[1:].tolist()),radius,[dict(name=f'return_{i}',a_mm=a.tolist(),b_mm=b.tolist(),radius_mm=radius,material='copper') for i,(a,b) in enumerate(zip(ret,ret[1:]))]


def filament_bundle(path, radius, radial, angular):
    """Area quadrature of closed, finite-radius current tubes, not open line sources."""
    path=np.asarray(path,float)
    if radial<1 or angular<4 or isinstance(radial,bool) or isinstance(angular,bool): raise ValueError('Source quadrature needs >=1 radial and >=4 azimuthal samples')
    if not np.allclose(path[0],path[-1],atol=1e-10,rtol=0): raise ValueError('Source path must be closed')
    path=path[:-1]; segments=np.roll(path,-1,axis=0)-path
    lengths=np.linalg.norm(segments,axis=1)
    if np.any(lengths<1e-10): raise ValueError('Degenerate source segment')
    directions=segments/lengths[:,None]
    tangent=directions+np.roll(directions,1,axis=0)
    norms=np.linalg.norm(tangent,axis=1)
    if np.any(norms<1e-8): raise ValueError('Current path reverses at a vertex')
    tangent/=norms[:,None]
    # One reproducible transverse reference; avoid tangents parallel to that reference.
    ref=np.eye(3)[np.argmin(np.max(np.abs(tangent),axis=0))]
    u=ref-np.einsum('ij,j->i',tangent,ref)[:,None]*tangent
    norms=np.linalg.norm(u,axis=1)
    if np.any(norms<1e-8): raise ValueError('Cannot build source cross-section frame')
    u/=norms[:,None];v=np.cross(tangent,u)
    paths=[]
    for ir in range(radial):
        r=radius*np.sqrt((ir+.5)/radial)
        for ia in range(angular):
            angle=2*np.pi*(ia+.5)/angular
            shifted=path+r*(u*np.cos(angle)+v*np.sin(angle))
            paths.append(np.vstack([shifted,shifted[0]]))
    return paths,1/(radial*angular)


def make_mesh(config, output, benchmark=False, size=None, padding=None):
    import gmsh
    from mesh import build_cad,wire_union
    settings=config['em']; opts=settings['mesh']; valid=settings['validation']
    h=float(size if size is not None else opts['size_mm'])
    pad=float(padding if padding is not None else opts['air_padding_mm'])
    fine=float(opts['conductor_size_mm']);air_h=float(opts['air_size_mm']);return_fine=float(opts.get('return_size_mm',fine))
    if not all(np.isfinite(x) and x>0 for x in (h,pad,fine,air_h,return_fine)): raise ValueError('Positive finite mesh sizes and air padding required')
    if benchmark:
        r=valid['loop_radius_mm'];wire=valid['wire_radius_mm'];n=valid['loop_segments']
        if not 0<wire<r or not isinstance(n,int) or n<12: raise ValueError('Invalid circular-loop benchmark geometry')
        t=np.linspace(0,2*np.pi,n+1);path=np.c_[r*np.cos(t),r*np.sin(t),np.zeros_like(t)]
        module=None;returns=[];fine=min(fine,wire*.8,h*.12)
    else:
        module,path,wire,returns=source_path(config,json.loads((DATA/'footprint.json').read_text()))
    filaments,weight=filament_bundle(path,wire,settings['source']['radial_quadrature'],settings['source']['azimuthal_quadrature'])
    output.mkdir(parents=True,exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',int(opts.get('verbose',False)));gmsh.model.add('magnetic_domain');occ=gmsh.model.occ
        entries=[]
        if benchmark:
            entries=[('loop','copper',(3,occ.addTorus(0,0,0,r,wire)))]
        else:
            regions,_,_,_=build_cad(module)
            entries=[(region['name'],region['material'],(3,tag)) for region in regions.values() for tag in region['volume_tags']]
            if returns:
                entries.extend(('return_conductor','copper',shape) for shape in wire_union(returns))
        pcb_included=not benchmark and settings.get('geometry',{}).get('include_pcb_dielectric',False)
        if pcb_included:
            from pcb_3d import build_assembly
            board=build_assembly(config,json.loads((DATA/'footprint.json').read_text()))
            substrate=board['substrate']
            if substrate is None:raise ValueError('PCB dielectric requested but its thickness is unset')
            shift=np.array(board['module_placements'][0]['translation_mm'])
            lower=np.array(substrate['min_mm'])-shift;upper=np.array(substrate['max_mm'])-shift
            entries.append(('pcb_dielectric',substrate['material'],(3,occ.addBox(*lower,*(upper-lower)))))
        occ.synchronize();bounds=np.array([gmsh.model.getBoundingBox(*e[2]) for e in entries]);lo=bounds[:,:3].min(axis=0)-pad;hi=bounds[:,3:].max(axis=0)+pad
        air=(3,occ.addBox(*lo,*(hi-lo)))
        all_entries=[('air','air',air)]+entries
        print('Fragmenting module, explicit return, and external air...',flush=True)
        result,mapping=occ.fragment([air],[e[2] for e in entries]);occ.synchronize()
        owners={}
        for (name,material,_),children in zip(all_entries,mapping):
            for dim,tag in children:
                if dim==3: owners[tag]=(name,material)
        if set(owners)!={tag for dim,tag in result if dim==3}: raise ValueError('Incomplete air-domain material partition')
        gmsh.model.removePhysicalGroups()
        names=sorted(set(owners.values()));ids={pair:i+1 for i,pair in enumerate(names)}
        regions={ids[pair]:dict(name=pair[0],material=pair[1],volume_tags=[t for t,p in owners.items() if p==pair]) for pair in names}
        for i,region in regions.items():gmsh.model.addPhysicalGroup(3,region['volume_tags'],i,name=region['name'])
        gmsh.option.setNumber('Mesh.ElementOrder',1);gmsh.option.setNumber('Mesh.MeshSizeMin',min(fine*.5,return_fine))
        gmsh.option.setNumber('Mesh.MeshSizeMax',max(air_h,h));gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',12);gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        # Distance from all nonair surfaces controls resolution without changing dimensions.
        def threshold(tags,near,far,distance):
            field=gmsh.model.mesh.field
            d=field.add('Distance');field.setNumbers(d,'SurfacesList',list(set(tags)));field.setNumber(d,'Sampling',80)
            t=field.add('Threshold');field.setNumber(t,'InField',d);field.setNumber(t,'SizeMin',near);field.setNumber(t,'SizeMax',far);field.setNumber(t,'DistMin',max(2*h,1.5) if near==h else 0);field.setNumber(t,'DistMax',distance);return t
        module_surfaces=[];copper_surfaces=[];return_surfaces=[]
        for tag,(name,material) in owners.items():
            if material=='air':continue
            surfaces=[abs(t) for d,t in gmsh.model.getBoundary([(3,tag)],oriented=False) if d==2]
            module_surfaces.extend(surfaces)
            if material=='copper':copper_surfaces.extend(surfaces)
            if name=='return_conductor':return_surfaces.extend(surfaces)
        fields=[threshold(module_surfaces,h,air_h,max(pad*.65,h)),threshold(copper_surfaces,fine,max(h,air_h),max(.4,h))]
        if returns and return_fine<fine:
            # Analytic local fields avoid expensive closest-surface distance evaluation.
            corners=[c['b_mm'] for c in returns[:-1]]
            for point in corners:
                ball=gmsh.model.mesh.field.add('Ball')
                for key,value in dict(XCenter=point[0],YCenter=point[1],ZCenter=point[2],Radius=3*wire,Thickness=2*wire,VIn=return_fine,VOut=max(h,air_h)).items():
                    gmsh.model.mesh.field.setNumber(ball,key,value)
                fields.append(ball)
        f=gmsh.model.mesh.field.add('Min');gmsh.model.mesh.field.setNumbers(f,'FieldsList',fields);gmsh.model.mesh.field.setAsBackgroundMesh(f)
        print('Generating linear tetrahedra including air...',flush=True);gmsh.model.mesh.generate(3)
        node_tags,coords,_=gmsh.model.mesh.getNodes();order=np.argsort(node_tags);sorted_tags=node_tags[order];xyz=np.asarray(coords).reshape(-1,3)[order]
        tets=[];rid=[]
        for tag,pair in owners.items():
            types,_,conn=gmsh.model.mesh.getElements(3,tag)
            for kind,nodes in zip(types,conn):
                if kind!=4:raise ValueError('Require first-order tetrahedra')
                tets.append(np.searchsorted(sorted_tags,nodes).reshape(-1,4));rid.extend([ids[pair]]*(len(nodes)//4))
        tet=np.concatenate(tets);rid=np.array(rid,dtype=np.int32)
        # Gmsh may include unattached CAD nodes; remove them before topology construction.
        used=np.unique(tet);reverse=np.full(len(xyz),-1);reverse[used]=np.arange(len(used));xyz=xyz[used];tet=reverse[tet]
        gmsh.write(str(output/'domain.msh'));gmsh.write(str(output/'domain.brep'))
    finally:gmsh.finalize()
    xyz*=1e-3;tet=tet.astype(np.int32)
    det=np.linalg.det((xyz[tet[:,1:]]-xyz[tet[:,0,None]]).transpose(0,2,1));flip=det<0;tet[flip]=tet[flip][:,[0,2,1,3]]
    if np.any(abs(det)<1e-30):raise ValueError('Degenerate tetrahedra')
    report=dict(units='mm',regions=regions,source_geometry='circular_loop_benchmark' if benchmark else 'illustrative_helix' if module['inductor_detail']['illustrative'] else 'configured_winding',
                source_path_mm=path.tolist(),source_wire_radius_mm=wire,source_filaments=len(filaments),source_weight=weight,
                air_bounds_mm=[(lo).tolist(),(hi).tolist()],size_mm=h,conductor_size_mm=fine,air_padding_mm=pad,
                included_geometry=('single active module, explicit external return, existing PCB dielectric slab and air; passive PCB copper (mu_r=1) is magnetostatically equivalent to air and omitted' if pcb_included else 'isolated module, explicit external return and surrounding air; PCB solids are not included'),
                module=module,return_capsules=returns)
    np.savez(output/'domain.npz',points_mm=xyz*1e3,tetrahedra=tet,region_id=rid)
    (output/'mesh_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return xyz,tet,rid,report,filaments,weight


def topology(xyz,tet):
    local=tet[:,PAIRS];global_edges=np.sort(local,axis=2)
    edges,inverse=np.unique(global_edges.reshape(-1,2),axis=0,return_inverse=True)
    te=(inverse.reshape(-1,6)+1)*np.where(local[:,:,0]<local[:,:,1],1,-1)
    face_nodes=np.sort(tet[:,FACES].reshape(-1,3),axis=1)
    faces,first,fi,count=np.unique(face_nodes,axis=0,return_index=True,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError('Nonmanifold tetrahedral mesh')
    owners=np.full((len(faces),2),-1,dtype=np.int32);owners[:,0]=first//4
    order=np.argsort(fi,kind='stable');twice=np.flatnonzero(count==2)
    starts=np.r_[0,np.cumsum(count)[:-1]];owners[twice,1]=order[starts[twice]+1]//4
    boundary=faces[count==1];boundary_nodes=np.unique(boundary)
    if len(xyz)-len(edges)+len(faces)-len(tet)!=1:raise ValueError('Tree gauge currently requires a contractible filled air-box mesh (Euler characteristic 1)')
    # All mesh nodes and the outer boundary must each be connected.
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(xyz),len(xyz)))
    if connected_components(graph,directed=False,return_labels=False)!=1:raise ValueError('Disconnected mesh')
    boundary_edges=np.unique(np.sort(boundary[:,[(0,1),(0,2),(1,2)]].reshape(-1,2),axis=1),axis=0)
    bmap={int(v):i for i,v in enumerate(boundary_nodes)}
    bi=np.array([[bmap[int(v)] for v in e] for e in boundary_edges]);bg=coo_matrix((np.ones(len(bi)),(bi[:,0],bi[:,1])),shape=(len(boundary_nodes),len(boundary_nodes)))
    if connected_components(bg,directed=False,return_labels=False)!=1:raise ValueError('Multiple boundary components require additional topology treatment')
    lookup={tuple(e):i for i,e in enumerate(edges)};fixed=np.zeros(len(edges),np.int32)
    fixed[[lookup[tuple(e)] for e in boundary_edges]]=1
    parent=np.arange(len(xyz));parent[boundary_nodes]=boundary_nodes[0]
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    tree=0
    for i,(a,b) in enumerate(edges):
        aa,bb=find(a),find(b)
        if aa!=bb: parent[aa]=bb;tree+=1
    if tree!=len(xyz)-len(boundary_nodes):raise ValueError('Incomplete spanning-tree gauge')
    return edges.astype(np.int32),te.astype(np.int32),fixed,faces.astype(np.int32),owners


def source_intersections(xyz,tet,paths,weight):
    """Exact straight-segment clipping; endpoint barycentrics encode Whitney integrals."""
    corners=xyz[tet];origin=corners[:,0];matrix=(corners[:,1:]-origin[:,None]).transpose(0,2,1)
    inverse=np.linalg.inv(matrix);lo=corners.min(axis=1);hi=corners.max(axis=1)
    indices=[];aa=[];bb=[];ww=[]
    for path in paths:
        path=np.asarray(path)*1e-3
        for a,b in zip(path,path[1:]):
            selected=np.flatnonzero(np.all(hi>=np.minimum(a,b)-1e-13,axis=1)&np.all(lo<=np.maximum(a,b)+1e-13,axis=1))
            la=np.einsum('nij,nj->ni',inverse[selected],a-origin[selected]);lb=np.einsum('nij,nj->ni',inverse[selected],b-origin[selected]);la=np.c_[1-la.sum(axis=1),la];lb=np.c_[1-lb.sum(axis=1),lb];delta=lb-la
            enter=np.zeros(len(selected));leave=np.ones(len(selected));valid=np.ones(len(selected),bool)
            for j in range(4):
                slope=delta[:,j];flat=abs(slope)<1e-14
                valid&=~(flat&(la[:,j]<-1e-10))
                crossing=np.divide(-la[:,j],slope,out=np.zeros_like(slope),where=~flat)
                enter=np.where(slope>1e-14,np.maximum(enter,crossing),enter);leave=np.where(slope< -1e-14,np.minimum(leave,crossing),leave)
            selected_mask=valid&(leave-enter>1e-12)&(leave>=0)&(enter<=1)
            ids=np.flatnonzero(selected_mask)
            if not len(ids):raise ValueError('Source segment lies outside the air-domain mesh')
            # A filament can lie on a tetrahedron face: choose one owner for each interval.
            breaks=np.clip(np.r_[enter[ids],leave[ids],0.,1.],0,1);breaks.sort();breaks=breaks[np.r_[True,np.diff(breaks)>1e-11]]
            covered=0.
            for start,end in zip(breaks,breaks[1:]):
                if end-start<1e-11:continue
                mid=(start+end)/2;active=ids[(enter[ids]<=mid+1e-12)&(leave[ids]>=mid-1e-12)]
                if not len(active):raise ValueError('Uncovered source interval; refine mesh or repair geometry')
                i=active[0];indices.append(selected[i]);aa.append(la[i]+start*delta[i]);bb.append(la[i]+end*delta[i]);ww.append(weight);covered+=end-start
            if abs(covered-1)>1e-8:raise ValueError('Source segment clipping lost current')
    print(f'Closed finite-radius source: {len(paths)} filaments, {len(indices)} element intersections',flush=True)
    return np.array(indices,np.int32),np.array(aa),np.array(bb),np.array(ww)


def prepare(config, output, benchmark=False, size=None, padding=None, reuse_mesh=None):
    s=config['em']
    if s['mode']!='quasistatic_no_eddy':raise ValueError('Only quasistatic_no_eddy is implemented; transient A-phi/eddy currents are not implemented')
    if s['boundary']['type']!='zero_tangential_A':raise ValueError('Only explicitly truncated zero tangential A boundary is implemented')
    if reuse_mesh is None:
        xyz,tet,rid,report,paths,weight=make_mesh(config,output,benchmark,size,padding)
    else:
        mesh=np.load(reuse_mesh/'domain.npz');xyz=mesh['points_mm']*1e-3;tet=mesh['tetrahedra'];rid=mesh['region_id'];report=json.loads((reuse_mesh/'mesh_report.json').read_text())
        if benchmark != (report['source_geometry']=='circular_loop_benchmark'):raise ValueError('Reused mesh geometry does not match benchmark/module mode')
        paths,weight=filament_bundle(report['source_path_mm'],report['source_wire_radius_mm'],s['source']['radial_quadrature'],s['source']['azimuthal_quadrature'])
        output.mkdir(parents=True,exist_ok=True)
        np.savez(output/'domain.npz',points_mm=xyz*1e3,tetrahedra=tet,region_id=rid)
        (output/'mesh_report.json').write_text(json.dumps(report,indent=2)+'\n')
    edges,te,fixed,faces,owners=topology(xyz,tet)
    source,sa,sb,weights=source_intersections(xyz,tet,paths,weight)
    copper_ids=[int(k) for k,v in report['regions'].items() if v['material']=='copper']
    if not np.isin(rid[source],copper_ids).all():raise ValueError('Source quadrature leaves the actual copper volumes')
    mur=np.empty(len(tet));sigma=np.empty(len(tet));unknown=[]
    regions={int(k):v for k,v in report['regions'].items()}
    for key,region in regions.items():
        material=region['material'];values=s['materials'][material];mu=values['relative_permeability'];conductivity=values['conductivity_s_m']
        if not np.isfinite(mu) or mu<=0:raise ValueError('Relative permeability must be positive and finite')
        if conductivity is None:unknown.append(material);conductivity=0. # Unused placeholder; explicitly tracked, never used in a conductive calculation.
        if not np.isfinite(conductivity) or conductivity<0:raise ValueError('Conductivity must be nonnegative and finite')
        mur[rid==key]=mu;sigma[rid==key]=conductivity
    op=config['electrical']['operating_point'];frequency=op['switching_frequency_khz']*1e3;inductance=config['inductor']['nominal_inductance_h']
    vin,vout,current=op['vin_v'],op['vout_v'],op['iout_a']
    if not all(np.isfinite(v) and v>0 for v in (frequency,inductance,vin,vout,current)) or not vin>vout:raise ValueError('Invalid configured buck operating point')
    duty=vout/vin;ripple=vout*(1-duty)/(inductance*frequency)
    if current<=ripple/2:raise ValueError('Initial waveform requires continuous conduction (positive minimum current)')
    anim=s['animation'];periods=anim['periods'];samples=anim['samples_per_period']
    if not isinstance(periods,int) or not isinstance(samples,int) or periods<1 or samples<4:raise ValueError('Animation needs integer periods>=1 and samples_per_period>=4')
    # Include every switching corner even when duty does not land on a uniform sample.
    times=np.unique(np.r_[np.arange(periods*samples+1)/samples,(np.arange(periods)+duty)])/frequency
    with (output/'input.bin').open('wb') as f:
        f.write(b'POLEMG01');f.write(struct.pack('<6i',len(xyz),len(tet),len(edges),len(source),len(faces),len(times)))
        for a,kind in [(xyz,'f8'),(tet,'i4'),(edges,'i4'),(te,'i4'),(fixed,'i4'),(mur,'f8'),(sigma,'f8'),(source,'i4'),(sa,'f8'),(sb,'f8'),(weights,'f8'),(times,'f8'),(faces,'i4'),(owners,'i4')]:np.asarray(a,dtype='<'+kind).tofile(f)
    solver=s['solver']
    parameters=dict(current_a=current if not benchmark else s['validation']['current_a'],vin_v=vin,vout_v=vout,inductance_h=inductance,frequency_hz=frequency,**solver)
    (output/'parameters.txt').write_text(''.join(f'{k} {v:.17g}\n' for k,v in parameters.items()))
    meta=dict(mode=s['mode'],animation=anim,visualization=s.get('visualization',{}),benchmark=benchmark,nodes=len(xyz),tetrahedra=len(tet),edges=len(edges),free_edges=int((fixed==0).sum()),frames=len(times),
        source_continuity='closed filament loops; exact Whitney segment integrals',boundary=s['boundary'],materials=s['materials'],conductivities_unused=True,unknown_conductivities=unknown,
        duty_cycle=duty,ripple_peak_to_peak_a=ripple,minimum_current_a=current-ripple/2,maximum_current_a=current+ripple/2,frequency_hz=frequency,mean_current_a=parameters['current_a'],
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        limitations=['Linear isotropic magnetostatics, no eddy currents, saturation, hysteresis, or frequency-dependent core losses',
                     'Time-varying result is a GPU-scaled DC solution, not a transient A-phi solve',
                     'Finite-radius source uses closed-filament area quadrature, not a solved conductor current distribution',
                     'External return is an explicit modeling assumption, not a verified buck-loop interconnect',
                     report['included_geometry'],
                     'Mesh and finite air-domain convergence require validation sweeps'])
    (output/'input.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez(output/'topology.npz',edges=edges,tet_edges=te,fixed=fixed,faces=faces,owners=owners)
    print(f'Prepared {len(tet)} tetrahedra, {len(edges)} edges ({meta["free_edges"]} free); ripple {ripple:.6f} A p-p, duty {duty:.6f}',flush=True)
    return meta
