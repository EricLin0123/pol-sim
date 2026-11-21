"""Hypothetical die and heat-source footprints; no loss estimates or solver."""
import copy
import math


def add_die(model, proposal):
    package = next(b for b in model['bodies'] if b['name'] == 'ic_package')
    size = proposal['size_mm']
    center = proposal['center_xy_mm']
    bottom = proposal['bottom_z_mm']
    if len(size) != 3 or any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Die size must contain three positive finite dimensions')
    if len(center) != 2 or any(not math.isfinite(v) for v in [*center, bottom]):
        raise ValueError('Invalid die center or bottom')
    # Package origin is its bottom center, matching the rest of the shared model.
    z0 = package['min_mm'][2] + bottom
    die = dict(name='silicon_die', material=proposal['material'], role='solid',
               min_mm=[center[0]-size[0]/2, center[1]-size[1]/2, z0],
               max_mm=[center[0]+size[0]/2, center[1]+size[1]/2, z0+size[2]])
    if any(die['min_mm'][i] < package['min_mm'][i]-1e-9 or
           die['max_mm'][i] > package['max_mm'][i]+1e-9 for i in range(3)):
        raise ValueError('Die lies outside IC package')
    depth = proposal['heat_source_depth_mm']
    if not math.isfinite(depth) or not 0 < depth <= size[2]:
        raise ValueError('Heat-source depth must be positive and no greater than die thickness')
    surface = proposal.get('heat_source_surface')
    if surface not in (None, 'top', 'bottom'):
        raise ValueError('heat_source_surface must be null, top, or bottom')
    sources = []
    for name, region in proposal['regions'].items():
        source = dict(name=name, host='silicon_die', depth_mm=depth, surface=surface,
                      geometry_status=proposal['geometry_status'], min_mm=None, max_mm=None,
                      footprint_xy_mm=None, active_area_mm2=region.get('active_area_mm2'),
                      heat_source_power_w=region.get('heat_source_power_w'))
        for key in ('active_area_mm2', 'heat_source_power_w'):
            value = source[key]
            if value is not None and (not math.isfinite(value) or value < 0 or (key == 'active_area_mm2' and value == 0)):
                raise ValueError(f'Invalid {name}.{key}')
        if region.get('size_xy_mm') is not None:
            w, h = region['size_xy_mm']
            cx, cy = region['center_xy_mm']
            if any(not math.isfinite(v) for v in (w,h,cx,cy)) or min(w,h) <= 0:
                raise ValueError(f'Invalid {name} footprint')
            xy = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            if any(xy[i] < die['min_mm'][i]-1e-9 or xy[i+2] > die['max_mm'][i]+1e-9 for i in range(2)):
                raise ValueError(f'{name} footprint lies outside die')
            if source['active_area_mm2'] is not None and not math.isclose(w*h,source['active_area_mm2']):
                raise ValueError(f'{name} active area differs from footprint area')
            source['footprint_xy_mm'] = xy
            if surface:
                z = die['max_mm'][2]-depth if surface == 'top' else die['min_mm'][2]
                source.update(min_mm=[xy[0],xy[1],z], max_mm=[xy[2],xy[3],z+depth])
        sources.append(source)
    located = [s for s in sources if s['footprint_xy_mm'] is not None]
    for i,a in enumerate(located):
        for b in located[i+1:]:
            aa,bb = a['footprint_xy_mm'],b['footprint_xy_mm']
            if all(max(aa[j],bb[j]) < min(aa[j+2],bb[j+2])-1e-9 for j in range(2)):
                raise ValueError('Heat-source footprints overlap')
    model['bodies'].append(die)
    model['die'] = copy.deepcopy(proposal)
    model['heat_sources'] = sources
    model['assumptions'].extend([
        'Hypothetical die and FET floorplan; heat sources are annotations within silicon, not separate materials',
        'Heat-source volumes subdivide host silicon; subtract them from the background die when meshing',
        'IC outer box is an envelope; its remaining volume uses the configured fill material'])
