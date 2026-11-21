"""Prepare the mesh, invoke native CUDA thermal solving, and process results.

All stiffness assembly, boundary assembly, losses, PCG, and nonlinear iteration
are implemented in src/thermal.cu. Python never solves the thermal equations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from scipy.sparse import coo_matrix
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(config_path, mesh_dir, output, transient=False, cooling_mode=None):
    config = yaml.safe_load(config_path.read_text())
    settings = config['thermal']
    losses, cooling, solver = (settings[k] for k in ('losses','cooling','solver'))
    mode=cooling_mode or cooling['mode']
    mode='air' if mode=='still_air' else mode
    if mode not in ('air','liquid'):
        raise ValueError('thermal.cooling.mode must be air or liquid')
    liquid=cooling.get('liquid',{})
    ambient=liquid['ambient_temperature_k'] if mode=='liquid' else cooling['ambient_temperature_k']
    radiation=liquid.get('radiation_temperature_k',ambient) if mode=='liquid' else cooling['radiation_temperature_k']
    if losses['core_loss_model'] != 'fixed_placeholder':
        raise ValueError('Only the explicitly unvalidated fixed core-loss placeholder is implemented')
    if losses['auxiliary_ic_loss_w'] != 0:
        raise ValueError('Control logic has no region; auxiliary_ic_loss_w must remain zero')
    if losses['core_loss_validated']:
        raise ValueError('The fixed core-loss placeholder must be marked unvalidated')
    if mode=='air' and config['cooling']['method'] != 'still_air':
        raise ValueError('Legacy cooling.method conflicts with thermal.cooling.mode')
    report = json.loads((mesh_dir/'report.json').read_text())
    mesh = np.load(mesh_dir/'module.npz',allow_pickle=False)
    points = np.asarray(mesh['points_mm'],dtype='<f8')*1e-3
    tet = np.asarray(mesh['tetrahedra'],dtype='<i4')
    regions = np.asarray(mesh['region_id'],dtype='<i4')
    boundary = np.asarray(mesh['boundary_triangles'],dtype='<i4').copy()
    if not report.get('conformal_interfaces_verified') or report['units'] != 'mm':
        raise ValueError('Require a verified conformal mesh in mm')
    names = {r['name']:int(i) for i,r in report['regions'].items()}
    if set(np.unique(regions)) != {int(i) for i in report['regions']}:
        raise ValueError('Mesh region IDs do not match report.json')
    required = ['high_side_fet','low_side_fet','copper_winding','metal_composite','copper_pins']
    if any(k not in names for k in required):
        raise ValueError('Required thermal component mesh regions are missing')
    catalog = config['materials']['thermal']
    conductivity = np.empty(len(tet),dtype='<f8')
    heat_capacity = np.zeros(len(tet),dtype='<f8')
    for i,region in report['regions'].items():
        k = catalog[region['material']]['thermal_conductivity_w_m_k']
        if not isinstance(k,(float,int)) or not np.isfinite(k) or k <= 0:
            raise ValueError('This implementation needs positive scalar isotropic conductivity')
        conductivity[regions==int(i)] = k
        properties=catalog[region['material']]
        rho,cp=properties.get('density_kg_m3'),properties.get('specific_heat_j_kg_k')
        if all(isinstance(v,(float,int)) and np.isfinite(v) and v>0 for v in (rho,cp)):
            heat_capacity[regions==int(i)]=rho*cp
        elif transient:
            raise ValueError('Transient solving requires positive density and specific heat for every material')
        region['thermal_properties'] = catalog[region['material']]
    # Geometry processing only: identify each exterior face's actual material.
    all_faces = np.sort(np.concatenate([tet[:,f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]),axis=1)
    keys,first,counts = np.unique(all_faces,axis=0,return_index=True,return_counts=True)
    owners = {tuple(f):int(j%len(tet)) for f,j,n in zip(keys,first,counts) if n==1}
    if len(owners) != len(boundary):
        raise ValueError('Exterior face count mismatch')
    owner = np.array([owners[tuple(sorted(f))] for f in boundary])
    if len({tuple(sorted(f)) for f in boundary}) != len(boundary):
        raise ValueError('Duplicated exterior triangles')
    p = points[boundary]
    crosses = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    area = np.linalg.norm(crosses,axis=1)/2
    if np.any(area<=0):
        raise ValueError('Degenerate boundary triangle')
    normal = crosses/(2*area[:,None])
    inside = points[tet[owner]].mean(axis=1)
    flip = np.einsum('ij,ij->i',normal,p.mean(axis=1)-inside)<0
    boundary[flip] = boundary[flip][:,[0,2,1]]
    normal[flip] *= -1
    nz = normal[:,2]
    h = np.where(nz>.5,cooling['top_convection_w_m2_k'],
                 np.where(nz<-.5,cooling['bottom_convection_w_m2_k'],cooling['side_convection_w_m2_k'])).astype('<f8')
    epsilon = np.zeros(len(boundary),dtype='<f8')
    for i,region in report['regions'].items():
        selected = regions[owner]==int(i)
        if np.any(selected):
            epsilon[selected] = cooling['emissivity'][region['material']]
    pcb = np.zeros(len(boundary),dtype='<f8')
    modeled = cooling['pcb_explicitly_modeled'] or any(r['material']=='fr4' for r in report['regions'].values())
    contact = np.zeros(len(boundary),dtype=bool)
    if cooling['pcb_pin_path_enabled']:
        if modeled:
            raise ValueError('Lumped PCB path and explicit PCB cannot both be enabled')
        contact = (regions[owner]==names['copper_pins']) & (nz<-.5) & np.isclose(p.mean(axis=1)[:,2],points[:,2].min(),atol=1e-10)
        contact_area = float(area[contact].sum())
        resistance = cooling['pcb_pin_path_k_w']
        if not contact_area or not np.isfinite(resistance) or resistance<=0:
            raise ValueError('Missing bottom pin contacts or invalid lumped PCB resistance')
        pcb[contact] = 1/(resistance*contact_area)
        h[contact] = 0
        epsilon[contact] = 0
    else:
        contact_area = 0
    # The mesh already labels the inductor top separately from the IC ledges.
    top_ids=[int(i) for i,b in report['boundaries'].items() if b['name']=='top_cooling_candidate']
    top=np.isin(mesh['boundary_id'],top_ids)
    geometric_top=np.all(np.isclose(p[:,:,2],points[:,2].max(),atol=1e-10),axis=1)&(nz>.99)
    if not np.array_equal(top,geometric_top) or not np.any(top):
        raise ValueError('Mesh top-surface tag does not match the inductor top plane')
    liquid_h=np.zeros(len(boundary),dtype='<f8')
    coolant=np.full(len(boundary),ambient,dtype='<f8')
    if mode=='liquid':
        expected=liquid['contact_area_mm2']*1e-6
        actual=float(area[top].sum())
        if not np.isfinite(expected) or expected<=0 or not np.isclose(actual,expected,rtol=1e-6,atol=1e-12):
            raise ValueError(f'Liquid contact area mismatch: mesh {actual*1e6:g} vs config {expected*1e6:g} mm2')
        coefficient=liquid['heat_transfer_coefficient_w_m2_k']
        if not np.isfinite(coefficient) or coefficient<=0 or not np.isfinite(liquid['coolant_temperature_k']) or liquid['coolant_temperature_k']<=0:
            raise ValueError('Invalid liquid coefficient or absolute coolant temperature')
        h[:]=liquid['other_convection_w_m2_k']
        h[contact]=0 # Existing lumped PCB path remains the only pin-contact sink.
        h[top]=0;epsilon[top]=0 # Cooler-covered face has no additional air/radiation path.
        liquid_h[top]=coefficient;coolant[top]=liquid['coolant_temperature_k']
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(epsilon)) or np.any(h<0) or np.any(epsilon<0) or np.any(epsilon>1):
        raise ValueError('Invalid convection coefficient or emissivity')
    # CSR connectivity contains no numerical stiffness matrix: CUDA assembles it.
    row = np.repeat(tet,4,axis=1).ravel()
    col = np.tile(tet,(1,4)).ravel()
    csr = coo_matrix((np.ones(len(row),dtype=np.int32),(row,col)),shape=(len(points),len(points))).tocsr()
    csr.sort_indices()
    rowptr,columns = csr.indptr.astype('<i4'),csr.indices.astype('<i4')
    matrix_keys = np.repeat(np.arange(len(points),dtype=np.int64),np.diff(rowptr))*len(points)+columns
    def slots(cells):
        a = np.repeat(cells,cells.shape[1],axis=1).astype(np.int64)
        b = np.tile(cells,(1,cells.shape[1]))
        keys = a*len(points)+b
        indices = np.searchsorted(matrix_keys,keys)
        if np.any(indices>=len(matrix_keys)) or not np.array_equal(matrix_keys[indices],keys):
            raise ValueError('CSR topology missing a local coupling')
        return indices.astype('<i4')
    tet_slots,face_slots = slots(tet),slots(boundary)
    op = config['electrical']['operating_point']
    parameters = dict(vin=op['vin_v'],vout=op['vout_v'],iout=op['iout_a'],fsw=op['switching_frequency_khz']*1000,
        inductance=config['inductor']['nominal_inductance_h'],
        rhs_reference=config['electrical']['high_side_rds_on_mohm']*1e-3,
        rls_reference=config['electrical']['low_side_rds_on_mohm']*1e-3,
        reference_k=losses['reference_temperature_k'],mosfet_alpha=losses['mosfet_resistance_temperature_coefficient_per_k'],
        rise_s=losses['rise_time_s'],fall_s=losses['fall_time_s'],dead_s=losses['dead_time_per_transition_s'],
        dead_transitions=losses['dead_time_transitions_per_cycle'],diode_v=losses['body_diode_drop_v'],
        dcr_reference=losses['winding_dcr_reference_ohm'],copper_alpha=losses['copper_resistance_temperature_coefficient_per_k'],
        ac_loss_w=losses['winding_ac_loss_w'],core_loss_w=losses['core_loss_w'],auxiliary_w=losses['auxiliary_ic_loss_w'],
        ambient_k=ambient,radiation_k=radiation,liquid_mode=int(mode=='liquid'),
        hs_region=names['high_side_fet'],ls_region=names['low_side_fet'],winding_region=names['copper_winding'],core_region=names['metal_composite'])
    parameters.update(solver)
    temporal=settings.get('transient',{})
    parameters.update(transient=int(transient),time_step_s=temporal.get('time_step_s',0.1),
        duration_s=temporal.get('duration_s',60.0),snapshot_stride=temporal.get('snapshot_stride',5),
        initial_temperature_k=temporal.get('initial_temperature_k',ambient))
    if min(parameters[k] for k in ('time_step_s','duration_s','initial_temperature_k'))<=0 or parameters['snapshot_stride']<1 or int(parameters['snapshot_stride'])!=parameters['snapshot_stride']:
        raise ValueError('Invalid transient time step, duration, initial temperature, or snapshot stride')
    if parameters['initial_temperature_k']>solver['maximum_temperature_k']:
        raise ValueError('Initial temperature exceeds model temperature limit')
    if not (0<parameters['vout']<parameters['vin']) or min(parameters['iout'],parameters['fsw'],parameters['inductance'])<=0:
        raise ValueError('Invalid buck operating point')
    ripple=parameters['vout']*(1-parameters['vout']/parameters['vin'])/(parameters['inductance']*parameters['fsw'])
    if parameters['iout']<=ripple/2:
        raise ValueError('The specified loss model assumes continuous-conduction operation')
    if not 0<solver['relaxation']<=1 or any(solver[k]<=0 for k in solver if k!='relaxation'):
        raise ValueError('Invalid solver controls')
    if any(int(solver[k])!=solver[k] for k in ('max_outer_iterations','max_pcg_iterations')):
        raise ValueError('Iteration limits must be integers')
    if min(parameters['ambient_k'],parameters['radiation_k'],parameters['reference_k'])<=0:
        raise ValueError('Absolute temperatures must be positive')
    if solver['maximum_temperature_k']<=parameters['ambient_k']:
        raise ValueError('Temperature limit must exceed ambient')
    if any(not np.isfinite(v) for v in parameters.values()):
        raise ValueError('Nonfinite model parameter')
    nonnegative = ['rhs_reference','rls_reference','mosfet_alpha','rise_s','fall_s','dead_s','dead_transitions',
                   'diode_v','dcr_reference','copper_alpha','ac_loss_w','core_loss_w','auxiliary_w']
    if any(parameters[k]<0 for k in nonnegative):
        raise ValueError('Loss-model parameters must be nonnegative')
    if csr.nnz>=2**31 or len(points)>=2**31:
        raise ValueError('Mesh exceeds int32 connectivity support')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'input.bin').open('wb') as f:
        f.write(b'POLTHM03')
        f.write(struct.pack('<5i',len(points),len(tet),len(boundary),int(regions.max())+1,csr.nnz))
        for array in (points,tet,regions,conductivity,boundary,h,epsilon,pcb,tet_slots,face_slots,rowptr,columns,heat_capacity,
                      liquid_h,coolant,top.astype('<i4')):
            f.write(np.ascontiguousarray(array).tobytes())
    np.savez_compressed(output/'boundary_conditions.npz',triangles=boundary,area_m2=area,top_mask=top,
        air_h_w_m2_k=h,liquid_h_w_m2_k=liquid_h,coolant_temperature_k=coolant,emissivity=epsilon,pcb_h_w_m2_k=pcb)
    (output/'parameters.txt').write_text(''.join(f'{k} {v:.17g}\n' for k,v in parameters.items()))
    record = dict(mesh_dir=str(mesh_dir.resolve()),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        mesh_sha256=hashlib.sha256((mesh_dir/'module.npz').read_bytes()).hexdigest(),regions=report['regions'],boundaries=report['boundaries'],
        assumptions=settings,pcb_contact_area_m2=contact_area,pcb_conductance_w_k=float(np.dot(pcb,area)),
        radiation_excluded_on_pcb_contact=True,control_logic_heat_source=False,auxiliary_loss_w=0,
        mesh_quality=report.get('quality_status'),illustrative_winding=report.get('illustrative_winding'),
        mode='transient' if transient else 'steady_state',cooling_mode=mode,
        top_contact_area_m2=float(area[top].sum()),
        liquid_contact_conductance_w_k=float(np.dot(liquid_h,area)),
        liquid_settings=liquid if mode=='liquid' else None,
        top_design_target_k=liquid.get('target_top_temperature_k'),
        electrical_parameters={k:parameters[k] for k in ('vin','vout','iout','fsw','inductance','rhs_reference','rls_reference',
            'reference_k','mosfet_alpha','rise_s','fall_s','dead_s','dead_transitions','diode_v','dcr_reference','copper_alpha','ac_loss_w','core_loss_w','auxiliary_w')},
        time_parameters={k:parameters[k] for k in ('time_step_s','duration_s','snapshot_stride','initial_temperature_k')},
        method='conforming P1 tetrahedral FEM; consistent Robin matrices; three-point triangle radiation quadrature',
        time_discretization='backward Euler with lumped nodal heat capacity' if transient else None)
    (output/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Prepared {len(points):,} nodes, {len(tet):,} tetrahedra, {len(boundary):,} exterior faces; no CPU solve.',flush=True)
    return record


def postprocess_series(directory, mesh, record, result):
    """Compact nodal time history: reuse one mesh, rather than duplicating geometry per frame."""
    times=np.loadtxt(directory/'transient_times.txt',ndmin=1)
    raw=np.fromfile(directory/'transient_temperature.bin',dtype='<f8')
    if raw.size!=len(times)*mesh.n_points or len(times)==0 or not np.all(np.isfinite(raw)):
        raise ValueError('Invalid transient snapshot data')
    temperatures=raw.reshape(len(times),mesh.n_points)
    if times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Transient times must start at zero and strictly increase')
    if not np.isclose(times[-1],result['time_s']) or not np.allclose(temperatures[-1],mesh.point_data['temperature_K']):
        raise ValueError('Final transient snapshot does not match the accepted final state')
    np.savez_compressed(directory/'temperature_series.npz',time_s=times,temperature_K=temperatures,
        mesh_file=np.array('temperature.vtu'),series_kind=np.array('physical_transient'),
        complete=np.array(bool(result['converged'])))
    # Component means are postprocessing of the GPU snapshots, never a CPU time solve.
    source=np.load(Path(record['mesh_dir'])/'module.npz')
    tet=source['tetrahedra'];vol=source['element_volume_mm3'];rid=source['region_id']
    names=[];means=[];maximum=[]
    for i,r in record['regions'].items():
        mask=rid==int(i);nodes=np.unique(tet[mask]);names.append(r['name'])
        weights=np.zeros(mesh.n_points)
        np.add.at(weights,tet[mask].ravel(),np.repeat(vol[mask]/4,4))
        means.append(temperatures@weights/vol[mask].sum()-273.15)
        maximum.append(temperatures[:,nodes].max(axis=1)-273.15)
    np.savez_compressed(directory/'component_time_history.npz',time_s=times,
        names=np.array(names),mean_temperature_C=np.array(means),maximum_temperature_C=np.array(maximum))
    return times,temperatures


def postprocess(directory):
    import pyvista as pv
    record = json.loads((directory/'input.json').read_text())
    result = json.loads((directory/'solution.json').read_text())
    mesh = pv.read(Path(record['mesh_dir'])/'module.vtu')
    temperature = np.fromfile(directory/'temperature.bin',dtype='<f8')
    if len(temperature)!=mesh.n_points or not np.all(np.isfinite(temperature)):
        raise ValueError('Invalid GPU temperature output')
    mesh.point_data['temperature_K']=temperature
    mesh.point_data['temperature_C']=temperature-273.15
    cells = mesh.cells.reshape(-1,5)[:,1:]
    mean = temperature[cells].mean(axis=1)
    mesh.cell_data['mean_temperature_C']=mean-273.15
    region_id = mesh.cell_data['region_id']
    powers = {int(r['id']):r['power_w'] for r in result['regions']}
    npz = np.load(Path(record['mesh_dir'])/'module.npz')
    volumes = npz['element_volume_mm3']*1e-9
    q = np.zeros(mesh.n_cells)
    stats = {}
    components = directory/'components';components.mkdir(exist_ok=True)
    for i,r in record['regions'].items():
        mask = region_id==int(i)
        q[mask] = powers[int(i)]/volumes[mask].sum()
        selected_nodes=np.unique(cells[mask])
        stats[r['name']]=dict(minimum_c=float(temperature[selected_nodes].min()-273.15),
            maximum_c=float(temperature[selected_nodes].max()-273.15),
            volume_average_c=float(np.average(mean[mask],weights=volumes[mask])-273.15),power_w=powers[int(i)])
    mesh.cell_data['heat_generation_W_m3']=q
    mesh.save(directory/'temperature.vtu')
    series=None
    if result.get('mode')=='transient':
        series=postprocess_series(directory,mesh,record,result)
    for i,r in record['regions'].items():
        if r['name'] in ('high_side_fet','low_side_fet','copper_winding','metal_composite'):
            mesh.extract_cells(region_id==int(i)).save(components/(r['name']+'.vtu'))
    silicon=[int(i) for i,r in record['regions'].items() if r['material']=='silicon']
    mask=np.isin(region_id,silicon)
    mesh.extract_cells(mask).save(components/'silicon_die.vtu')
    stats['silicon_die']=dict(minimum_c=float(temperature[np.unique(cells[mask])].min()-273.15),
        maximum_c=float(temperature[np.unique(cells[mask])].max()-273.15),
        volume_average_c=float(np.average(mean[mask],weights=volumes[mask])-273.15))
    result.update(components=stats,input=record)
    result['cooling_mode']=record.get('cooling_mode','air')
    if result.get('top_surface_average_k') is not None:
        result['top_surface']=dict(area_mm2=result['top_surface_area_m2']*1e6,
            average_temperature_c=result['top_surface_average_k']-273.15,
            maximum_temperature_c=result['top_surface_maximum_k']-273.15)
        target=record.get('top_design_target_k')
        if target is not None:
            result['top_surface'].update(design_target_c=target-273.15,
                maximum_minus_target_k=result['top_surface_maximum_k']-target,
                design_target_imposed_as_boundary=False)
    hotspot=int(np.argmax(temperature))
    result['global_maximum_temperature_c']=float(temperature[hotspot]-273.15)
    result['global_hotspot_position_mm']=mesh.points[hotspot].tolist()
    if 'efficiency' in result:
        result['efficiency_percent']=100*result['efficiency']
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (components/'report.json').write_text(json.dumps({'regions':record['regions']},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    named=['silicon_die','high_side_fet','low_side_fet','copper_winding','metal_composite']
    x=np.arange(len(named))
    axes[0,0].bar(x-.18,[stats[n]['maximum_c'] for n in named],width=.36,label='Maximum')
    axes[0,0].bar(x+.18,[stats[n]['volume_average_c'] for n in named],width=.36,label='Volume average')
    axes[0,0].set_xticks(x,named,rotation=20);axes[0,0].set_ylabel('Temperature (C)');axes[0,0].legend()
    axes[0,1].bar(result['losses_w'].keys(),result['losses_w'].values());axes[0,1].tick_params(axis='x',rotation=45)
    axes[0,1].set_ylabel('Loss (W)')
    paths=['Convection','Radiation','PCB'];removed=[result[k] for k in ('convection_w','radiation_w','pcb_w')]
    if record.get('cooling_mode')=='liquid':
        paths.append('Liquid');removed.append(result['liquid_cooling_w'])
    axes[1,0].bar(paths,removed);axes[1,0].set_ylabel('Heat out (W)')
    history=np.genfromtxt(directory/'history.tsv',names=True,delimiter='\t',ndmin=1)
    if series is None:
        for key in ('nonlinear_relative_residual','energy_relative_error'):
            axes[1,1].semilogy(history['outer'],np.maximum(history[key],1e-16),label=key)
        axes[1,1].set_xlabel('Nonlinear iteration')
    else:
        times,temperatures=series
        axes[1,1].plot(times,temperatures.max(axis=1)-273.15,label='Module maximum')
        axes[1,1].set_xlabel('Physical time (s)');axes[1,1].set_ylabel('Temperature (C)')
    axes[1,1].legend(fontsize=8)
    label=('Transient: saved time steps converged' if result.get('mode')=='transient' else 'Converged') if result['converged'] else 'NOT COMPLETED / NOT CONVERGED'
    fig.suptitle(label+' — preliminary thermal model')
    fig.savefig(directory/'summary.png',dpi=160);plt.close(fig)
    print('Wrote temperature.vtu, component maps, and report.json; converged =',result['converged'])


def execute(config_path,mesh_dir,output,solver,transient=False,cooling_mode=None,prepare_only=False):
    record=prepare(config_path,mesh_dir,output,transient,cooling_mode)
    if prepare_only:return 0
    # Remove stale outputs so a failed GPU run cannot look like new results.
    for name in ('solution.json','temperature.bin','report.json','temperature.vtu','history.tsv','summary.png',
                 'transient_times.txt','transient_temperature.bin','temperature_series.npz','component_time_history.npz','time_history.tsv'):
        (output/name).unlink(missing_ok=True)
    if (output/'components').is_dir():
        for old in (output/'components').glob('*.vtu'):old.unlink()
        (output/'components'/'report.json').unlink(missing_ok=True)
    run=subprocess.run([str(solver.resolve()),str(output.resolve())],check=False)
    status=dict(exit_code=run.returncode,solution_available=(output/'solution.json').exists(),
                status='completed' if run.returncode==0 else 'not_converged' if run.returncode==2 else 'cuda_execution_failed',
                numerical_solution_converged=False,cooling_mode=record['cooling_mode'])
    if status['solution_available']:
        status['numerical_solution_converged']=bool(json.loads((output/'solution.json').read_text())['converged'])
    (output/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    if (output/'solution.json').exists():postprocess(output)
    return run.returncode


def compare_results(directory):
    """Compare only converged steady-state GPU outputs with identical mesh/electrical input."""
    records={m:json.loads((directory/m/'input.json').read_text()) for m in ('air','liquid')}
    if any(records['air'][k]!=records['liquid'][k] for k in ('config_sha256','mesh_sha256','electrical_parameters','mode')) or records['air']['mode']!='steady_state':
        raise ValueError('Comparison requires identical mesh, operating/loss parameters and steady-state mode')
    summary=dict(identical_operating_conditions=True,regions=records['air']['regions'],
        modes={m:json.loads((directory/m/'run_status.json').read_text()) for m in records})
    for name in ('temperature_comparison.png','temperature_difference.vtu'):
        (directory/name).unlink(missing_ok=True)
    if not all(s['numerical_solution_converged'] for s in summary['modes'].values()):
        summary.update(status='unavailable',reason='Both CUDA runs must converge before temperature comparison is generated')
        (directory/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('Temperature comparison unavailable: one or both CUDA runs did not produce a converged solution.')
        return False
    reports={m:json.loads((directory/m/'report.json').read_text()) for m in records}
    for m,r in reports.items():
        if not r['converged'] or r.get('mode')!='steady_state' or r.get('cooling_mode')!=m:
            raise ValueError('Comparison report is not a converged steady-state case for the requested cooling mode')
        if any(r['input'][k]!=records[m][k] for k in ('config_sha256','mesh_sha256','electrical_parameters')):
            raise ValueError('Comparison output is stale relative to its prepared inputs')
    import pyvista as pv
    air=pv.read(directory/'air'/'temperature.vtu');liquid=pv.read(directory/'liquid'/'temperature.vtu')
    if not np.array_equal(air.points,liquid.points) or not np.array_equal(air.cells,liquid.cells):
        raise ValueError('Temperature comparison meshes do not match')
    difference=air.copy();difference.point_data.clear()
    for key in ('mean_temperature_C','heat_generation_W_m3'):
        if key in difference.cell_data:del difference.cell_data[key]
    difference.point_data['temperature_difference_C']=liquid.point_data['temperature_C']-air.point_data['temperature_C']
    difference.save(directory/'temperature_difference.vtu')
    summary.update(status='completed',results={m:{k:r[k] for k in ('top_surface','components','total_loss_w','efficiency','convection_w','radiation_w','pcb_w','liquid_cooling_w','energy_balance_relative_error')} for m,r in reports.items()})
    (directory/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Top surface','Silicon die','High-side FET','Low-side FET','Winding','Composite']
    parts=['silicon_die','high_side_fet','low_side_fet','copper_winding','metal_composite']
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    x=np.arange(len(names))
    for ax,stat,topkey in zip(axes,('volume_average_c','maximum_c'),('average_temperature_c','maximum_temperature_c')):
        for m,offset in (('air',-.18),('liquid',.18)):
            values=[reports[m]['top_surface'][topkey]]+[reports[m]['components'][n][stat] for n in parts]
            ax.bar(x+offset,values,width=.36,label=m)
        target=reports['liquid']['top_surface'].get('design_target_c')
        if target is not None:ax.plot([-.4,.4],[target,target],'r--',label='Top target (not a BC)')
        ax.set_xticks(x,names);ax.set_ylabel(('Average' if stat=='volume_average_c' else 'Maximum')+' temperature (C)');ax.legend()
    fig.suptitle('Converged CUDA cooling comparison — identical electrical operating conditions')
    fig.savefig(directory/'temperature_comparison.png',dpi=160);plt.close(fig)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    p.add_argument('--mesh-dir',type=Path,default=ROOT/'output'/'mesh')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--transient',action='store_true',help='Solve physical time evolution with CUDA, not nonlinear-iteration playback')
    p.add_argument('--cooling',choices=('air','liquid'),help='Override thermal.cooling.mode without editing the air settings')
    p.add_argument('--compare',action='store_true',help='Run identical steady-state cases for both cooling modes and compare converged outputs')
    p.add_argument('--solver',type=Path,default=ROOT/'build'/'thermal')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--postprocess-only',action='store_true')
    a=p.parse_args()
    selected=a.cooling or yaml.safe_load(a.config.read_text())['thermal']['cooling']['mode']
    selected='air' if selected=='still_air' else selected
    if a.compare:
        if a.transient or a.postprocess_only or a.cooling:
            p.error('--compare runs both steady-state modes; do not combine it with --transient, --postprocess-only or --cooling')
        folder=a.output_dir or ROOT/'output'/'cooling_comparison'
        codes=[execute(a.config,a.mesh_dir,folder/m,a.solver,False,m,a.prepare_only) for m in ('air','liquid')]
        if not a.prepare_only and not compare_results(folder):raise SystemExit(max(1,*codes))
        return
    default='thermal'+('_liquid' if selected=='liquid' else '')+('_transient' if a.transient else '')
    a.output_dir=a.output_dir or ROOT/'output'/default
    if a.postprocess_only:
        postprocess(a.output_dir);return
    code=execute(a.config,a.mesh_dir,a.output_dir,a.solver,a.transient,selected,a.prepare_only)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
