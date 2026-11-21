"""View the project PCB reference layout; dimensions are mm, not fabrication data."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def draw_reference(g, C, rect, poly, via, chip, text, line):
    # Local coordinate origin at the centre of 4.5 mm package; source x=346,y=379.
    # Ground plane fields above / below the VIN bus, and side grounds.
    rect(g,-346, -352, 236,206,C['pgnd'])
    rect(g,107, -352, 209,206,C['pgnd'])
    rect(g,-346,39,238,64,C['pgnd'])
    rect(g,110,39,206,64,C['pgnd'])
    # VOUT large polygon at north, with thin neck to pads.
    poly(g,[(-99,-352),(101,-352),(101,-112),(-99,-112)],C['vout'])
    # PGND heat-spreader region with slot cutout around module.
    poly(g,[(-156,-158),(-111,-158),(-111,-102),(-156,-102)],C['pgnd'])
    poly(g,[(110,-158),(155,-158),(155,-102),(110,-102)],C['pgnd'])
    # Signal rails and fanouts, left/right.
    for yy in (-90,-55,-20):
        poly(g,[(-346,yy-19),(-239,yy-19),(-216,yy-3),(-119,yy-3),(-119,yy+10),(-220,yy+10),(-245,yy+1),(-346,yy+1)],C['signal'])
    poly(g,[(113,-96),(206,-96),(250,-130),(260,-130),(267,-115),(252,-98),(223,-76),(113,-76)],C['signal'])
    poly(g,[(113,-61),(215,-61),(267,-92),(285,-92),(293,-74),(265,-57),(230,-43),(113,-43)],C['signal'])
    # VIN band continuous through all repeated packages; pale-outline separation.
    poly(g,[(-346,2),(-107,2),(-90,-27),(89,-27),(106,2),(316,2),(316,32),(107,32),(89,13),(-89,13),(-107,32),(-346,32)],C['vin'])
    # VCC small polygon and vertical signal tails.
    poly(g,[(-247,111),(-110,111),(-110,157),(-247,157)],'#87afe3')
    for xx in (-88,-53,-18,17,86):
        rect(g,xx-12,121,25,189,C['signal'])
    rect(g,109,111,44,46,C['signal'])
    # QFN outline and pads; rotated image is board view.
    rect(g,-156,-157,312,314,'none','#252525',2.3)
    rect(g,-151,-155,47,49,'#a5a5a5','#333333',1.1,rx=10)
    rect(g,105,-155,48,49,'#a5a5a5','#333333',1.1,rx=10)
    rect(g,-94,-101,188,62,'#858585','#303030',1,rx=6)
    rect(g,-94,-27,188,40,'#548dd2','#2a476c',1,rx=5)
    rect(g,-94,22,188,76,'#8b8b8b','#3d3d3d',1,rx=8)
    rect(g,-94,-157,188,35,'#702e9b','#48205e',1,rx=5)
    for x in (-66,-13,40):
        rect(g,x-8,-158,18,8,'#702e9b','#3b2154',.7)
    for yy in (-128,-93,-58,-23,12,47,82,125):
        for x0 in (-156,120):
            if yy in (-128,-93,-58,-23,12,47,82,125):
                rect(g,x0,yy,35,18,'#898989','#414141',.8,rx=5)
    # Bottom solder pads.
    for xx in (-88,-53,-18,17,86):
        rect(g,xx-11,121,24,37,'#00b45e','#006a3c',1,rx=6)
    # Power & ground thermal via patterns inside package
    for x in (-53,0,53): via(g,x,-7)
    via(g,0,-60)
    for x,y in [(-53,49),(0,49),(53,49),(-26,77),(26,77)]:via(g,x,y)
    # Via rows in outer PGND regions
    for x in (-166,164):
        for y in (-282,-246,-211):via(g,x,y)
    for x in (-262,264):
        for y in (-62,18,56,91):via(g,x,y)
    for x,y in [(-229,133),(247,-112),(281,-72),(-263,12),(264,12)]:via(g,x,y)
    # North external decoupling (upper VOUT field)
    for x in (-104,104):
        for yy in (-274,-224,-182):chip(g,x,yy,'h','0402' if yy != -182 else '0201')
    # VIN caps and signal mode config resistors.
    for yy in (-114,-69,-24):chip(g,-266,yy,'h')
    chip(g,-204,46,'v')
    chip(g,177,46,'v')
    chip(g,178,-74,'v','0201')
    # Text labels scaled from reference.  Avoid claim of full schematic detail.
    for label,x,y in [('PGND',-193,-178),('PGND',193,-178),('VOUT',0,-183),
                      ('PGND',0,-78),('PGND',193,2),('VIN',-295,24),('VIN',295,24),
                      ('VCC',-179,140),('ADDR',-190,-81),('MS2',-190,-49),('MS1',-190,-15),
                      ('VOS',220,-92),('GOS',222,-48),('SW (NC)',207,138)]:text(g,label,x,y,18)
    for label,xx in [('EN',-88),('PG',-53),('PMB_DATA',-18),('PMB_CLK',17),('BOOT (NC)',86)]:
        text(g,label,xx,205,17,rotation=90)
    # Three vertical guide/keepout markers at right edge.
    for x in (238,259,280):line(g,x,-352,x,-136,'#161616',2.0,dash='13,7')




def render(config, output, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, Circle, FancyBboxPatch
    p = config['pcb']['layout_parameters']
    n = p['modules']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('pcb.layout_parameters.modules must be a positive integer')
    fw, fh = map(float, config['geometry']['module_footprint_mm'])
    pitch = float(p['module_pitch_mm'])
    margin = float(p['board_margin_mm'])
    if not all(math.isfinite(v) and v > 0 for v in (fw, fh, pitch)) or not math.isfinite(margin) or margin < 0:
        raise ValueError('Footprint/pitch must be positive; margin must be nonnegative')
    if not math.isclose(fw, fh):
        raise ValueError('The traced reference supports square module footprints only')
    if pitch < fw:
        raise ValueError('Module footprints overlap: pitch is smaller than footprint')
    scale = fw / 312.0
    fullw = (n-1)*pitch + 762*scale
    height = 703*scale
    C = p['colors']
    fig, ax = plt.subplots(figsize=(16, 6), layout='constrained')
    def xy(g, x, y):
        return (g[0] + x*scale, g[1] + y*scale)
    def rect(g, x,y,w,h,fill,stroke='none',sw=1,rx=0):
        pos = xy(g,x,y)
        kw = dict(facecolor=fill, edgecolor=stroke, linewidth=sw*.45)
        if rx:
            patch = FancyBboxPatch(pos,w*scale,h*scale,boxstyle=f'round,pad=0,rounding_size={rx*scale}',**kw)
        else:
            patch = Rectangle(pos,w*scale,h*scale,**kw)
        ax.add_patch(patch)
    def poly(g,pts,fill,stroke='none',sw=1):
        ax.add_patch(Polygon([xy(g,*v) for v in pts],facecolor=fill,edgecolor=stroke,linewidth=sw*.45))
    def circ(g,x,y,r,fill='#bcbcbc',stroke='#111',sw=2):
        ax.add_patch(Circle(xy(g,x,y),r*scale,facecolor=fill,edgecolor=stroke,linewidth=sw*.45))
    def line(g,x1,y1,x2,y2,stroke,width=1,dash=None):
        a,b=xy(g,x1,y1),xy(g,x2,y2)
        ax.plot([a[0],b[0]],[a[1],b[1]],color=stroke,lw=width*.45,linestyle='--' if dash else '-')
    def text(g,s,x,y,size=18,fill='#101010',rotation=None,anchor='middle',weight='normal'):
        ax.text(*xy(g,x,y),str(s),ha='center',va='center',fontsize=size*.40,color=fill,rotation=-(rotation or 0),weight=weight)
    def via(g,x,y,r=10):
        outer = float(p['via_outer_diameter_mm'])
        drill = float(p['via_drill_diameter_mm'])
        if not 0 < drill < outer:
            raise ValueError('Via diameters require 0 < drill < outer')
        circ(g,x,y,outer/(2*scale))
        circ(g,x,y,drill/(2*scale),'white','none',0)
    def chip(g,x,y,orient='h',type='0402'):
        # All component markers share the supplied visual size, as in the reference.
        w,h = (float(v)/scale for v in p['component_0402_size_mm'])
        if not all(math.isfinite(v) and v > 0 for v in (w,h)):
            raise ValueError('Component body dimensions must be positive')
        if orient != 'h': w,h=h,w
        rect(g,x-w/2,y-h/2,w,h,'#aaaaaa','#252525',1.8)
        rect(g,x-w*.3,y-h*.4,w*.6,h*.8,C['resistor'],'#252525',1.2)
        text(g,type,x,y,17,'#1121d1',rotation=0 if orient=='h' else 90,weight='bold')
    centers = [[346*scale + i*pitch, 379*scale] for i in range(n)]
    for center in centers:
        draw_reference(center, C, rect, poly, via, chip, text, line)
    ax.set(xlim=(-margin,fullw+margin), ylim=(height+margin,-margin),
           xlabel='x (mm)', ylabel='y (mm)',
           title=f'{n} parallel DC/DC modules — illustrative PCB reference')
    ax.set_aspect('equal')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'):
        fig.savefig(str(output)+suffix,dpi=180,bbox_inches='tight')
    meta = dict(units='mm', coordinate_system='reference upper-left, x right, y down',
        module_count=n,module_footprint_mm=[fw,fh],module_pitch_mm=pitch,
        module_center_xy_mm=centers,view_bounds_mm=[-margin,-margin,fullw+margin,height+margin],
        fabrication_ready=False,em_ready=False,layout_parameters=p,
        unknowns=['physical board outline','layer stackup',
                  'verified pad and net connectivity','exact trace boundaries',
                  'dielectric and conductor EM properties','ports and boundary conditions'],
        assumptions=['Raster-traced reference geometry; repetition does not establish electrical connectivity',
                     'View bounds are illustration bounds, not a physical board outline',
                     'Component bodies are visual markers, not solder lands'])
    Path(str(output)+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Wrote {output}.png, {output}.svg, {output}.json')
    if show: plt.show()
    else: plt.close(fig)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pcb/pcb_three_modules',help='Output prefix')
    parser.add_argument('--no-view',action='store_true',help='Export without opening a window')
    args=parser.parse_args()
    if args.no_view:
        import matplotlib
        matplotlib.use('Agg')
    config=yaml.safe_load(args.config.read_text())
    render(config,args.output,show=not args.no_view)


if __name__ == '__main__':
    main()
