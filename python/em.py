"""Prepare and run the native CUDA magnetic solver; no Python numerical solve."""
import argparse
import hashlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml
from geometry import ROOT
from em_mesh import prepare

OUTPUT_FILES=('B_dc.bin','B_series.bin','B_series.partial','A_edges.bin','current.bin','times.bin','solution.json','magnetic.vtu','report.json','summary.png','leakage_planes.npz','leakage.png')


def invoke(output, executable):
    for name in OUTPUT_FILES:(output/name).unlink(missing_ok=True)
    (output/'run_status.json').write_text(json.dumps(dict(status='running',mode='quasistatic_no_eddy'))+'\n')
    result=subprocess.run([str(executable.resolve()),str(output.resolve())],check=False)
    if result.returncode:
        (output/'run_status.json').write_text(json.dumps(dict(status='failed',returncode=result.returncode,field_available=False))+'\n')
        raise RuntimeError('Native CUDA EM solve failed; no field prediction is available')
    (output/'run_status.json').write_text(json.dumps(dict(status='converged',mode='quasistatic_no_eddy'))+'\n')


def postprocess(output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pyvista as pv
    solution=json.loads((output/'solution.json').read_text())
    if not solution.get('converged') or solution.get('eddy_currents_included') is not False:raise ValueError('Require converged native no-eddy-current output')
    meta=json.loads((output/'input.json').read_text());report=json.loads((output/'mesh_report.json').read_text());mesh=np.load(output/'domain.npz')
    points=mesh['points_mm'];tet=mesh['tetrahedra'];rid=mesh['region_id'];nt=len(tet)
    B=np.fromfile(output/'B_dc.bin',dtype='<f8').reshape(nt,3)
    times=np.fromfile(output/'times.bin',dtype='<f8');current=np.fromfile(output/'current.bin',dtype='<f8')
    if len(times)!=meta['frames'] or len(current)!=len(times) or (output/'B_series.bin').stat().st_size!=len(times)*nt*3*8:raise ValueError('Native field series size mismatch')
    if not np.isfinite(B).all() or not np.isfinite(current).all():raise ValueError('Nonfinite field output')
    grid=pv.UnstructuredGrid(np.c_[np.full(nt,4),tet].ravel(),np.full(nt,pv.CellType.TETRA),points)
    grid.cell_data['region_id']=rid;grid.cell_data['B_T']=B;grid.cell_data['B_mT']=B*1e3;grid.cell_data['B_magnitude_mT']=np.linalg.norm(B,axis=1)*1e3
    grid.field_data['mode']=['quasistatic_no_eddy'];grid.save(output/'magnetic.vtu')
    volume=np.abs(np.linalg.det((points[tet[:,1:]]-points[tet[:,0,None]]).transpose(0,2,1)))/6
    statistics={}
    magnitude=np.linalg.norm(B,axis=1)
    for key,region in report['regions'].items():
        selected=rid==int(key)
        statistics[region['name']]=dict(material=region['material'],maximum_B_T=float(magnitude[selected].max()),volume_average_B_T=float(np.average(magnitude[selected],weights=volume[selected])),volume_mm3=float(volume[selected].sum()))
    summary=dict(**solution,input=meta,regions=statistics,geometry=dict(source=report['source_geometry'],included=report['included_geometry']),validation_status='Analytical benchmark and refinement/domain sweeps are separate; see validation/report.json',
        visualization_note='Cell B is the raw curl(A); streamlines use a display-only nodal reconstruction')
    (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(times*1e6,current);axes[0].set(xlabel='Time (µs)',ylabel='Inductor current (A)',title=f'Ideal buck: ΔI={meta["ripple_peak_to_peak_a"]:.4f} A p-p')
    names=list(statistics);x=np.arange(len(names));axes[1].bar(x,[statistics[n]['volume_average_B_T']*1e3 for n in names]);axes[1].set_xticks(x,names,rotation=45,ha='right');axes[1].set(ylabel='Volume-average |B| (mT)',title='DC field at configured mean current')
    fig.suptitle('Linear quasistatic approximation — no eddy currents');fig.savefig(output/'summary.png',dpi=160);plt.close(fig)
    leakage_planes(grid,report,output,config)
    (output/'run_status.json').write_text(json.dumps(dict(status='converged',mode='quasistatic_no_eddy'))+'\n')
    return summary


def leakage_planes(grid, mesh_report, output, config):
    import matplotlib.pyplot as plt
    vis=config['em'].get('visualization',{})
    resolution=int(vis.get('leakage_resolution',70));offset=float(vis.get('leakage_offset_mm',.5))
    if resolution<3 or offset<=0:raise ValueError('Invalid leakage-plane settings')
    module=mesh_report.get('module');parts=module['bodies'] if module else []
    if parts:lo=np.min([b['min_mm'] for b in parts],axis=0);hi=np.max([b['max_mm'] for b in parts],axis=0)
    else:r=config['em']['validation']['loop_radius_mm'];lo=np.array([-r,-r,-r]);hi=-lo
    planes=[('above',2,hi[2]+offset),('side_x',0,hi[0]+offset),('side_y',1,lo[1]-offset)]
    fields={};fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(name,normal,value) in zip(axes,planes):
        dimensions=[i for i in range(3) if i!=normal]
        u,v=np.meshgrid(np.linspace(lo[dimensions[0]]-offset,hi[dimensions[0]]+offset,resolution),np.linspace(lo[dimensions[1]]-offset,hi[dimensions[1]]+offset,resolution))
        xyz=np.zeros((u.size,3));xyz[:,normal]=value;xyz[:,dimensions[0]]=u.ravel();xyz[:,dimensions[1]]=v.ravel()
        ids=grid.find_containing_cell(xyz);valid=ids>=0;B=np.full_like(xyz,np.nan);B[valid]=grid.cell_data['B_T'][ids[valid]]
        fields[name+'_points_mm']=xyz;fields[name+'_B_T']=B
        values=np.linalg.norm(B,axis=1).reshape(u.shape)*1e3
        image=ax.pcolormesh(u,v,values,shading='auto');fig.colorbar(image,ax=ax,label='|B| (mT)');ax.set(xlabel='xyz'[dimensions[0]]+' (mm)',ylabel='xyz'[dimensions[1]]+' (mm)',title=f'{name}: {"xyz"[normal]}={value:g} mm');ax.set_aspect('equal')
    np.savez(output/'leakage_planes.npz',**fields);fig.savefig(output/'leakage.png',dpi=160);plt.close(fig)


def benchmark_error(output,config):
    import pyvista as pv
    grid=pv.read(output/'magnetic.vtu');v=config['em']['validation'];radius=v['loop_radius_mm']*1e-3;current=v['current_a'];z=np.array(v['axis_probe_z_over_radius'])*radius
    probes=np.c_[np.zeros(len(z)),np.zeros(len(z)),z*1e3];ids=grid.find_containing_cell(probes)
    if np.any(ids<0):raise ValueError('Benchmark probes outside air domain')
    raw=grid.cell_data['B_T'][ids]
    # Standard nodal patch recovery for point evaluation; retain raw cell errors too.
    recovered=pv.PolyData(probes).sample(grid.cell_data_to_point_data())
    actual=np.asarray(recovered['B_T']);expected=4e-7*np.pi*current*radius**2/(2*(radius**2+z**2)**1.5)
    errors=np.linalg.norm(actual-np.c_[np.zeros(len(z)),np.zeros(len(z)),expected],axis=1)/expected
    result=dict(probes_mm=probes.tolist(),analytical_Bz_T=expected.tolist(),computed_B_T=actual.tolist(),raw_cell_B_T=raw.tolist(),raw_cell_relative_vector_errors=(np.linalg.norm(raw-np.c_[np.zeros(len(z)),np.zeros(len(z)),expected],axis=1)/expected).tolist(),evaluation='VTK cell-to-node patch average followed by linear interpolation; raw containing-cell errors also reported',relative_vector_errors=errors.tolist(),maximum_relative_error=float(errors.max()),correct_orientation=bool(np.all(actual[:,2]>0)),
        reference='Thin circular loop axis field; finite wire-radius and polygonal-source differences are included in reported error')
    (output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n');return result


def run(config,output,args,benchmark=False,size=None,padding=None,reuse=None):
    output.mkdir(parents=True,exist_ok=True)
    for name in OUTPUT_FILES:(output/name).unlink(missing_ok=True)
    prepare(config,output,benchmark,size,padding,reuse)
    if args.prepare_only:return None
    invoke(output,args.executable);summary=postprocess(output,config)
    if benchmark:summary['analytical_benchmark']=benchmark_error(output,config)
    return summary


def validate(config,args):
    settings=config['em']['validation'];root=args.output/'validation';records=[]
    # Common probe locations make differences meaningful as mesh/domain vary.
    for kind,values in [('mesh',settings['mesh_sizes_mm']),('domain',settings['air_paddings_mm'])]:
        for value in values:
            output=root/f'{kind}_{value:g}';kwargs={'size':value} if kind=='mesh' else {'padding':value,'size':min(settings['mesh_sizes_mm'])}
            case=copy.deepcopy(config);case['em']['animation']=dict(periods=1,samples_per_period=4,playback_interval_ms=80)
            result=run(case,output,args,True,**kwargs)
            if result:records.append(dict(kind=kind,value=value,benchmark=result['analytical_benchmark'],audits={k:result[k] for k in ('source_continuity_relative','normal_flux_jump_relative','independent_current_linearity_relative')}))
    if args.prepare_only:return
    comparisons={}
    for kind in ('mesh','domain'):
        rows=[r for r in records if r['kind']==kind]
        ordered=sorted(rows,key=lambda r:r['value'],reverse=kind=='mesh')
        if len(ordered)<2:raise ValueError('Validation requires at least two mesh sizes and two domain paddings')
        a,b=[np.array(r['benchmark']['computed_B_T']) for r in ordered[-2:]]
        comparisons[kind+'_last_pair_relative_change']=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    finest=min((r for r in records if r['kind']=='mesh'),key=lambda r:r['value'])
    largest=max((r for r in records if r['kind']=='domain'),key=lambda r:r['value'])
    passed=all(r['benchmark']['correct_orientation'] and r['audits']['normal_flux_jump_relative']<=settings['max_flux_jump_relative'] and r['audits']['independent_current_linearity_relative']<=settings['max_linearity_relative_error'] for r in records)
    passed&=max(finest['benchmark']['maximum_relative_error'],largest['benchmark']['maximum_relative_error'])<=settings['max_axis_relative_error']
    passed&=comparisons['mesh_last_pair_relative_change']<=settings['max_refinement_relative_change'] and comparisons['domain_last_pair_relative_change']<=settings['max_domain_relative_change']
    report=dict(passed=bool(passed),records=records,comparisons=comparisons,thresholds=settings,limitations='Loop benchmark only; module/core-contrast and source-quadrature convergence still required for quantitative device predictions')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'run_status.json').write_text(json.dumps(dict(status='passed' if passed else 'failed',action='analytical_loop_validation'))+'\n')
    if not passed:raise RuntimeError(f'Validation thresholds failed; inspect {root/"report.json"}')


def validate_module(config,args):
    """Device-model sensitivity, distinct from the analytical loop verification."""
    import pyvista as pv
    root=args.output/'module_validation';root.mkdir(parents=True,exist_ok=True)
    baseline=args.reuse_mesh
    if baseline is None or not (baseline/'magnetic.vtu').exists():
        raise ValueError('--validate-module needs --reuse-mesh pointing to a solved baseline module')
    baseline_meta=json.loads((baseline/'input.json').read_text())
    if baseline_meta['benchmark']:raise ValueError('Device validation needs a module, not a loop benchmark')
    controls=config['em']['validation'];mesh=json.loads((baseline/'mesh_report.json').read_text())
    points=np.array(controls['module_probe_points_mm'],float)
    def metrics(directory):
        report=json.loads((directory/'report.json').read_text());grid=pv.read(directory/'magnetic.vtu')
        mesh_regions=json.loads((directory/'mesh_report.json').read_text())['regions']
        air_ids=[int(k) for k,v in mesh_regions.items() if v['material']=='air']
        air=grid.extract_cells(np.isin(grid.cell_data['region_id'],air_ids))
        sampled=pv.PolyData(points).sample(air.cell_data_to_point_data())
        if not np.all(sampled['vtkValidPointMask']):raise ValueError('Device validation probes outside air domain')
        return dict(core_average_B_T=report['regions']['metal_composite']['volume_average_B_T'],
            core_maximum_B_T=report['regions']['metal_composite']['maximum_B_T'],probe_B_T=np.asarray(sampled['B_T']).tolist())
    reference=metrics(baseline);records=[dict(case='baseline',directory=str(baseline),metrics=reference)]
    case=copy.deepcopy(config);case['em']['animation']=dict(periods=1,samples_per_period=4)
    cases=[('refined_mesh',dict(size=controls['module_refined_size_mm'],padding=mesh['air_padding_mm']),False),
           ('larger_domain',dict(size=mesh['size_mm'],padding=controls['module_larger_padding_mm']),False),
           ('conductor_refinement',dict(size=mesh['size_mm'],padding=mesh['air_padding_mm']),False),
           ('source_quadrature',dict(reuse=root/'conductor_refinement'),True)]
    for name,kwargs,source_refine in cases:
        c=copy.deepcopy(case)
        if source_refine or name=='conductor_refinement':
            c['em']['mesh']['conductor_size_mm']*=.5
            c['em']['mesh']['return_size_mm']=c['em']['mesh']['conductor_size_mm']*.5
            c['em']['mesh']['verbose']=True
        if source_refine:c['em']['source']['radial_quadrature']*=2
        output=root/name
        cached=(output/'input.json').exists() and (output/'report.json').exists()
        if cached:cached=json.loads((output/'input.json').read_text())['config_sha256']==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        try:
            result=json.loads((output/'report.json').read_text()) if cached else run(c,output,args,**kwargs)
        except (ValueError,RuntimeError) as error:
            records.append(dict(case=name,error=str(error)));continue
        if result:
            comparison=metrics(root/'conductor_refinement') if source_refine else reference
            value=metrics(output);a=np.array(comparison['probe_B_T']);b=np.array(value['probe_B_T'])
            records.append(dict(case=name,directory=str(output),metrics=value,
                compared_with='conductor_refinement' if source_refine else 'baseline',
                core_average_relative_change=abs(value['core_average_B_T']/comparison['core_average_B_T']-1),
                probe_relative_change=float(np.linalg.norm(b-a)/np.linalg.norm(a))))
    if args.prepare_only:return
    passed=all('error' not in r and r['core_average_relative_change']<=controls['max_module_core_average_change'] and r['probe_relative_change']<=controls['max_module_probe_change'] for r in records[1:])
    report=dict(passed=passed,records=records,probe_points_mm=points.tolist(),
        meaning='One-step mesh, boundary and source-quadrature sensitivity, not proof of asymptotic convergence; local peaks are reported but not used as converged observables',
        thresholds=dict(core_average=controls['max_module_core_average_change'],probes=controls['max_module_probe_change']))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'run_status.json').write_text(json.dumps(dict(status='passed' if passed else 'failed',action='module_sensitivity'))+'\n')
    if not passed:raise RuntimeError(f'Device sensitivity thresholds failed; inspect {root/"report.json"}')


def compare(config,args):
    rows=[];base=args.reuse_mesh
    for mu in config['em']['comparison']['relative_permeabilities']:
        case=copy.deepcopy(config);case['em']['materials']['metal_composite']['relative_permeability']=mu
        output=args.output/'permeability'/f'mur_{mu:g}';result=run(case,output,args,reuse=base)
        if base is None:base=output
        if result:rows.append(dict(relative_permeability=mu,regions=result['regions'],relative_residual=result['relative_residual']))
    if not args.prepare_only:
        (args.output/'permeability/comparison.json').write_text(json.dumps(dict(mode='quasistatic_no_eddy',cases=rows,mu_1_meaning='Composite region replaced by a mu_r=1 material; conductivity is unused'),indent=2)+'\n')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for region in ('metal_composite','air'):
            ax.plot([r['relative_permeability'] for r in rows],[r['regions'][region]['volume_average_B_T']*1e3 for r in rows],'o-',label=region)
        ax.set(xlabel='Composite relative permeability',ylabel='Volume-average |B| (mT)',title='Magnetic permeability comparison — no eddy currents');ax.legend();fig.savefig(args.output/'permeability/comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml');p.add_argument('--output',type=Path,default=ROOT/'output/em');p.add_argument('--executable',type=Path,default=ROOT/'build/em')
    p.add_argument('--pcb',action='store_true',help='Full PCB geometry is included by default')
    action=p.add_mutually_exclusive_group();action.add_argument('--validate-module',action='store_true');action.add_argument('--benchmark',action='store_true');action.add_argument('--validate',action='store_true');action.add_argument('--compare',action='store_true');action.add_argument('--postprocess-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--reuse-mesh',type=Path);p.add_argument('--size',type=float);p.add_argument('--padding',type=float)
    args=p.parse_args();config=yaml.safe_load(args.config.read_text())
    if args.pcb:config['em'].setdefault('geometry',{})['include_pcb_dielectric']=True
    try:
        if args.validate_module or args.compare:
            raise ValueError('Isolated-module sweeps are retired. Use full-domain runs with --size/--padding in separate output directories.')
        elif args.validate:validate(config,args)
        elif args.postprocess_only:postprocess(args.output,config)
        elif args.benchmark:run(config,args.output,args,True,args.size,args.padding,args.reuse_mesh)
        else:
            from em_full import run as full_run
            full_run(config,args)
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.CalledProcessError) as error:
        failure_dir=args.output/('module_validation' if args.validate_module else 'validation' if args.validate else 'permeability' if args.compare else '')
        failure_dir.mkdir(parents=True,exist_ok=True)
        (failure_dir/'run_status.json').write_text(json.dumps(dict(status='failed',error=str(error)))+'\n')
        print(f'EM: {error}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
