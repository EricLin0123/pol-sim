"""Shared PyVista surfaces and visibility controls for geometry previews."""
import numpy as np
import pyvista as pv


def surface(faces):
    points = []
    cells = []
    for face in faces:
        cells.extend([len(face), *range(len(points), len(points) + len(face))])
        points.extend(face)
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(cells)).triangulate()


def box(lo, hi):
    return pv.Box(bounds=tuple(v for pair in zip(lo, hi) for v in pair))


def capsule(a, b, radius):
    a, b = np.asarray(a, float), np.asarray(b, float)
    tube = pv.Line(a, b).tube(radius=radius, n_sides=24, capping=True)
    return tube.merge(pv.Sphere(radius=radius, center=a)).merge(pv.Sphere(radius=radius, center=b))


def add(plotter, groups, mesh, color, label, opacity=1):
    actor = plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=False)
    groups.setdefault(label, []).append(actor)
    return actor


def controls(plotter, groups, title):
    plotter.set_background('white')
    plotter.add_text(title, position='upper_right', font_size=12, color='black')
    for i, (label, actors) in enumerate(groups.items()):
        y = 730 - 32*i
        def toggle(value, actors=actors):
            for actor in actors:
                actor.SetVisibility(bool(value))
            plotter.render()
        plotter.add_checkbox_button_widget(toggle, value=True, position=(12, y), size=20)
        plotter.add_text(label, position=(42, y), font_size=9, color='black')
    plotter.add_axes()
    plotter.show_grid(xtitle='x (mm)', ytitle='y (mm)', ztitle='z (mm)')
    plotter.view_vector((1, -1, .8), viewup=(0, 0, 1))
    plotter.reset_camera()
    plotter._visibility_controls = groups


def pad_surface(pad):
    x, y, w, h = pad['rect_mm']
    r = min(pad.get('corner_radius_mm', 0), w/2, h/2)
    if r <= 0:
        return surface([[(x,y,0),(x+w,y,0),(x+w,y+h,0),(x,y+h,0)]])
    points = []
    for cx, cy, start in ((x+w-r,y+r,-90),(x+w-r,y+h-r,0),
                           (x+r,y+h-r,90),(x+r,y+r,180)):
        for angle in np.radians(np.linspace(start, start+90, 9)):
            points.append((cx+r*np.cos(angle), cy+r*np.sin(angle), 0))
    return surface([points])
