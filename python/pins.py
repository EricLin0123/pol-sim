"""Finite-thickness rounded rectangular copper pin lands, in millimeters."""
import math


def add_pins(model, geometry):
    thickness = geometry.get('pin_thickness_mm')
    bottom = geometry.get('pin_bottom_z_mm')
    if thickness is None:
        return
    if bottom is None or not math.isfinite(bottom) or not math.isfinite(thickness) or thickness <= 0:
        raise ValueError('Provide positive pin thickness and finite pin bottom z')
    package = next(b for b in model['bodies'] if b['name'] == 'ic_package')
    z0 = package['min_mm'][2] + bottom
    z1 = z0 + thickness
    pins = []
    for p in model['footprint']['pads']:
        x,y,w,h = p['rect_mm']
        radius = p['corner_radius_mm']
        if any(not math.isfinite(v) for v in (x,y,w,h,radius)) or min(w,h) <= 0 or radius < 0:
            raise ValueError(f"Invalid pin geometry: {p['name']}")
        pins.append(dict(name=p['name'], net=p['net'], net_status=p['net_status'], material='copper',
                         primitive='rounded_rectangle_prism', rect_mm=[x,y,w,h],
                         corner_radius_mm=min(radius,w/2,h/2),
                         min_mm=[x,y,z0], max_mm=[x+w,y+h,z1]))
    model['pin_solids'] = pins
    model['pin_thickness_mm'] = thickness
    # Keep the original footprint untouched; this marks only the exterior contact plane.
    model['pad_z_mm'] = z0
    model['assumptions'].append('Copper pin lands are inward-extruded rounded prisms; overlapping lands require union before meshing')


def prism_faces(pin, arc_segments=8):
    """Closed polygonal approximation for display only; analytic radii are exported."""
    x,y,w,h = pin['rect_mm']
    r = pin['corner_radius_mm']
    if r == 0:
        boundary = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    else:
        boundary = []
        for cx,cy,start in ((x+w-r,y+r,-90),(x+w-r,y+h-r,0),
                            (x+r,y+h-r,90),(x+r,y+r,180)):
            for i in range(arc_segments+1):
                angle=math.radians(start+90*i/arc_segments)
                boundary.append((cx+r*math.cos(angle),cy+r*math.sin(angle)))
    z0,z1 = pin['min_mm'][2],pin['max_mm'][2]
    lower=[(a,b,z0) for a,b in boundary]
    upper=[(a,b,z1) for a,b in boundary]
    faces=[lower[::-1],upper]
    for i in range(len(boundary)):
        j=(i+1)%len(boundary)
        if boundary[i] != boundary[j]:
            faces.append([lower[i],lower[j],upper[j],upper[i]])
    return faces
