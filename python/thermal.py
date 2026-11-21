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


def prepare(config_path, mesh_dir, output):
    config = yaml.safe_load(config_path.read_text())
    settings = config['thermal']
    losses, cooling, solver = (settings[k] for k in ('losses','cooling','solver'))
    if cooling['mode'] != 'still_air':
        raise ValueError('This initial boundary implementation requires still_air')
    if losses['core_loss_model'] != 'fixed_placeholder':
        raise ValueError('Only the explicitly unvalidated fixed core-loss placeholder is implemented')
    if losses['auxiliary_ic_loss_w'] != 0:
        raise ValueError('Control logic has no region; auxiliary_ic_loss_w must remain zero')
    if losses['core_loss_validated']:
        raise ValueError('The fixed core-loss placeholder must be marked unvalidated')
    if config['cooling']['method'] != 'still_air':
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
    for i,region in report['regions'].items():
        k = catalog[region['material']]['thermal_conductivity_w_m_k']
        if not isinstance(k,(float,int)) or not np.isfinite(k) or k <= 0:
            raise ValueError('This implementation needs positive scalar isotropic conductivity')
        conductivity[regions==int(i)] = k
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
        if not contact_area or resistance<=0:
            raise ValueError('Missing bottom pin contacts or invalid lumped PCB resistance')
        pcb[contact] = 1/(resistance*contact_area)
        h[contact] = 0
        epsilon[contact] = 0
    else:
        contact_area = 0
    if np.any(h<0) or np.any(epsilon<0) or np.any(epsilon>1):
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
        ambient_k=cooling['ambient_temperature_k'],radiation_k=cooling['radiation_temperature_k'],
        hs_region=names['high_side_fet'],ls_region=names['low_side_fet'],winding_region=names['copper_winding'],core_region=names['metal_composite'])
    parameters.update(solver)
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
        f.write(b'POLTHM01')
        f.write(struct.pack('<5i',len(points),len(tet),len(boundary),int(regions.max())+1,csr.nnz))
        for array in (points,tet,regions,conductivity,boundary,h,epsilon,pcb,tet_slots,face_slots,rowptr,columns):
            f.write(np.ascontiguousarray(array).tobytes())
    (output/'parameters.txt').write_text(''.join(f'{k} {v:.17g}\n' for k,v in parameters.items()))
    record = dict(mesh_dir=str(mesh_dir.resolve()),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        mesh_sha256=hashlib.sha256((mesh_dir/'module.npz').read_bytes()).hexdigest(),regions=report['regions'],boundaries=report['boundaries'],
        assumptions=settings,pcb_contact_area_m2=contact_area,pcb_conductance_w_k=float(np.dot(pcb,area)),
        radiation_excluded_on_pcb_contact=True,control_logic_heat_source=False,auxiliary_loss_w=0,
        mesh_quality=report.get('quality_status'),illustrative_winding=report.get('illustrative_winding'),
        method='conforming P1 tetrahedral FEM; consistent Robin matrices; three-point triangle radiation quadrature')
    (output/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Prepared {len(points):,} nodes, {len(tet):,} tetrahedra, {len(boundary):,} exterior faces; no CPU solve.',flush=True)
    return record


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
    axes[1,0].bar(['Convection','Radiation','PCB'],[result[k] for k in ('convection_w','radiation_w','pcb_w')]);axes[1,0].set_ylabel('Heat out (W)')
    history=np.genfromtxt(directory/'history.tsv',names=True,delimiter='\t',ndmin=1)
    for key in ('nonlinear_relative_residual','energy_relative_error'):
        axes[1,1].semilogy(history['outer'],np.maximum(history[key],1e-16),label=key)
    axes[1,1].set_xlabel('Nonlinear iteration');axes[1,1].legend(fontsize=8)
    fig.suptitle(('Converged' if result['converged'] else 'NOT CONVERGED')+' — preliminary thermal model')
    fig.savefig(directory/'summary.png',dpi=160);plt.close(fig)
    print('Wrote temperature.vtu, component maps, and report.json; converged =',result['converged'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    p.add_argument('--mesh-dir',type=Path,default=ROOT/'output'/'mesh')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output'/'thermal')
    p.add_argument('--solver',type=Path,default=ROOT/'build'/'thermal')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--postprocess-only',action='store_true')
    a=p.parse_args()
    if a.postprocess_only:
        postprocess(a.output_dir);return
    prepare(a.config,a.mesh_dir,a.output_dir)
    if a.prepare_only:return
    # Remove stale outputs so a failed GPU run cannot look like new results.
    for name in ('solution.json','temperature.bin','report.json','temperature.vtu','history.tsv','summary.png'):
        (a.output_dir/name).unlink(missing_ok=True)
    if (a.output_dir/'components').is_dir():
        for old in (a.output_dir/'components').glob('*.vtu'):old.unlink()
        (a.output_dir/'components'/'report.json').unlink(missing_ok=True)
    run=subprocess.run([str(a.solver.resolve()),str(a.output_dir.resolve())],check=False)
    status=dict(exit_code=run.returncode,solution_available=(a.output_dir/'solution.json').exists(),
                status='completed' if run.returncode==0 else 'not_converged' if run.returncode==2 else 'cuda_execution_failed',
                numerical_solution_converged=False)
    if status['solution_available']:
        status['numerical_solution_converged']=bool(json.loads((a.output_dir/'solution.json').read_text())['converged'])
    (a.output_dir/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    if (a.output_dir/'solution.json').exists():postprocess(a.output_dir)
    if run.returncode:raise SystemExit(run.returncode)


if __name__=='__main__':main()
