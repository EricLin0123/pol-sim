"""Assumed buck power interconnects; dimensions are illustrative, not die reconstruction."""


def add_buck_interconnects(model):
    bodies = {b['name']: b for b in model['bodies']}
    sources = {s['name']: s for s in model.get('heat_sources', [])}
    pins = {p['name']: p for p in model.get('pin_solids', [])}
    if not all(k in bodies for k in ('terminal_1', 'terminal_2')):
        return
    if not all(sources.get(k, {}).get('min_mm') is not None for k in ('high_side_fet','low_side_fet')):
        return
    if not all(k in pins for k in ('VIN','VOUT','PGND_BOTTOM')):
        return
    def center(body):
        return [(a+b)/2 for a,b in zip(body['min_mm'],body['max_mm'])]
    nodes = {}
    contacts = []
    for name, source_name, fraction, net in (
        ('HS_D','high_side_fet',-.25,'VIN'), ('HS_S','high_side_fet',.25,'SW'),
        ('LS_D','low_side_fet',-.20,'SW'), ('LS_S','low_side_fet',.20,'PGND')):
        source = sources[source_name]
        lo, hi = source['min_mm'], source['max_mm']
        x=(lo[0]+hi[0])/2+fraction*(hi[0]-lo[0])
        y=(lo[1]+hi[1])/2
        z=hi[2]+.04
        nodes[name]=[x,y,z]
        contacts.append(dict(name='contact_'+name, material='copper', role='solid', net=net,
            device=source_name, terminal=name.split('_')[1],
            min_mm=[x-.15,y-.18,hi[2]-.005], max_mm=[x+.15,y+.18,hi[2]+.08]))
    nodes.update(VIN=center(pins['VIN']), VOUT=center(pins['VOUT']),
                 PGND=center(pins['PGND_BOTTOM']), L_IN=center(bodies['terminal_1']),
                 L_OUT=center(bodies['terminal_2']), SW=[-1.75,-1.65,.85])
    paths=[]
    def link(name, net, start, end, bends):
        paths.append(dict(name=name, net=net, start=start, end=end,
                          points_mm=[nodes[start],*bends,nodes[end]],width_mm=.30,thickness_mm=.10,
                          attachment='soldered FET top metal / leadframe',
                          geometry_status='assumed stamped copper clip'))
    x,y,z=nodes['HS_D']
    link('vin_to_high_side','VIN','VIN','HS_D',
         [[nodes['VIN'][0],-.25,.10],[-1.8,-.25,.10],[-1.8,-.25,.60],[-1.8,y,.60],[x,y,.60]])
    x,y,z=nodes['HS_S']
    link('high_side_to_switch_node','SW','HS_S','SW',[[x,y,.85],[-1.75,y,.85]])
    x,y,z=nodes['LS_D']
    link('switch_node_to_low_side','SW','SW','LS_D',[[x,-1.65,.95],[x,y,.95]])
    link('switch_node_to_inductor','SW','SW','L_IN',[[0,-1.65,.85]])
    x,y,z=nodes['L_OUT'];px,py,pz=nodes['VOUT']
    link('inductor_to_vout','VOUT','L_OUT','VOUT',[[x,y,.85],[px,py,.85]])
    x,y,z=nodes['LS_S'];px,py,pz=nodes['PGND']
    link('low_side_to_pgnd','PGND','LS_S','PGND',[[x,y,.60],[1.8,y,.60],[1.8,py,.60],[1.8,py,.10]])
    clips=[]
    for path in paths:
        # Orthogonal flat straps and downturned legs form a continuous stamped clip.
        route=[list(path['points_mm'][0])]
        for target in path['points_mm'][1:]:
            for axis in range(3):
                if abs(target[axis]-route[-1][axis])>1e-9:
                    point=list(route[-1]);point[axis]=target[axis];route.append(point)
        path['points_mm']=route
        for i,(a,b) in enumerate(zip(route,route[1:])):
            axis=max(range(3),key=lambda j:abs(b[j]-a[j]))
            half=[path['thickness_mm']/2]*3
            half[1 if axis in (0,2) else 0]=path['width_mm']/2
            clip=dict(name=f"clip_{path['name']}_{i}",material='copper',role='solid',
                      primitive='box',net=path['net'],connection=path['name'],
                      min_mm=[min(a[j],b[j])-half[j] for j in range(3)],
                      max_mm=[max(a[j],b[j])+half[j] for j in range(3)])
            clips.append(clip)
    model['bodies'].extend(contacts+clips)
    model['power_clips']=[clip['name'] for clip in clips]
    model['interconnect_capsules']=[]
    model['buck_topology']=dict(
        geometry_status='assumed synchronous buck copper clips',nodes_mm=nodes,connections=paths,
        devices=[dict(name='high_side_fet',drain='HS_D',source='HS_S',conducts='high_side_on'),
                 dict(name='low_side_fet',drain='LS_D',source='LS_S',conducts='low_side_on')],
        inductor=dict(input='L_IN',output='L_OUT',input_net='SW',output_net='VOUT'),
        external_ports=dict(input=['VIN','PGND'],load=['VOUT','PGND']),
        current_paths=dict(high_side_on=['VIN','HS_D','HS_S','SW','L_IN','L_OUT','VOUT','external_load','PGND','external_supply','VIN'],
                           low_side_on=['PGND','LS_S','LS_D','SW','L_IN','L_OUT','VOUT','external_load','PGND']),
        unresolved=['External load, input supply, capacitors and their return-current distributions are not modeled',
                    'FET gate connections and switching/conduction physics are not modeled',
                    'Existing EM filament-loop solver does not solve this branched buck circuit'])
    model['assumptions'] = [s.replace('; no pin-to-terminal mapping', '; terminal-to-pin mapping is defined by buck_topology') for s in model['assumptions']]
    model['assumptions'].append('User-requested buck topology; assumed 0.30 mm wide, 0.10 mm thick stamped copper clips soldered onto separate FET contacts and leadframe; solder represented as ideal contact, not measured internal geometry')
