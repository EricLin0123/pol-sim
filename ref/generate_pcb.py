#!/usr/bin/env python3
"""Parametric, vector redraw of a 3-module power-layout reference.

Run: python generate_pcb.py --params layout_params.yaml --device config.yaml
Outputs an SVG (editable vector), PNG preview, and JSON geometry metadata.
NOT an EDA/fabrication file; unknown PCB stackup and exact traces are assumptions.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import svgwrite
import cairosvg

# Reference image frame and package anchors in source pixel coordinates.
REF_W, REF_H = 2048, 703
REF_MODULE_W = 312.0
REF_PITCH = 661.0
REF_CENTER_X = 346.0
REF_CENTER_Y = 379.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='layout_params.yaml')
    ap.add_argument('--device', default='config.yaml')
    ap.add_argument('--output', default='pcb_single_module')
    args = ap.parse_args()
    p = yaml.safe_load(Path(args.params).read_text())
    d = yaml.safe_load(Path(args.device).read_text())
    n = int(p['modules'])
    assert n >= 1
    fw, fh = map(float, d['geometry']['module_footprint_mm'])
    pxmm = REF_MODULE_W / fw
    pitch = float(p['module_pitch_mm']) * pxmm
    # Reuse viewbox scale so all fabricated dimensions remain in mm.
    fullw = (n - 1) * pitch + 2 * REF_CENTER_X + 70
    h = REF_H
    margin = float(p.get('board_margin_mm', .25)) * pxmm
    dwg = svgwrite.Drawing(args.output+'.svg', size=(f'{(fullw+2*margin)/pxmm:.3f}mm', f'{(h+2*margin)/pxmm:.3f}mm'), viewBox=f'{-margin} {-margin} {fullw+2*margin} {h+2*margin}')
    dwg.set_desc(title='Parametric 3-module DC/DC PCB visualization', desc='Illustrative vector tracing. Position and copper connectivity are not verified for manufacturing.')
    C = p['colors']
    G = '#232323'
    def rect(g, x,y,w,h,fill,stroke='none',sw=1,rx=0):
        g.add(dwg.rect(insert=(x,y), size=(w,h), rx=rx, fill=fill, stroke=stroke, stroke_width=sw))
    def poly(g, pts, fill, stroke='none',sw=1):
        g.add(dwg.polygon(points=pts, fill=fill, stroke=stroke, stroke_width=sw, stroke_linejoin='round'))
    def circ(g,x,y,r,fill='#bcbcbc',stroke='#111',sw=2):
        g.add(dwg.circle(center=(x,y),r=r,fill=fill,stroke=stroke,stroke_width=sw))
    def line(g,x1,y1,x2,y2,stroke,width=1,dash=None):
        opt={'stroke':stroke,'stroke_width':width}
        if dash: opt['stroke_dasharray']=dash
        g.add(dwg.line((x1,y1),(x2,y2),**opt))
    def text(g,s,x,y,size=18,fill='#101010',rotation=None,anchor='middle',weight='normal'):
        obj=dwg.text(str(s),insert=(x,y),text_anchor=anchor,font_size=size,font_family='Arial, Liberation Sans, sans-serif',fill=fill,font_weight=weight)
        if rotation is not None:obj.rotate(rotation,center=(x,y))
        g.add(obj)
    def via(g,x,y,r=10): circ(g,x,y,r)
    def chip(g,x,y,orient='h',type='0402'):
        # x,y = body centre. Body includes solder-pad depiction.
        if orient=='h':
            rect(g,x-35,y-17,70,35,'#aaaaaa','#252525',1.8)
            rect(g,x-21,y-15,42,30,C['resistor'],'#252525',1.2)
        else:
            rect(g,x-17,y-35,35,70,'#aaaaaa','#252525',1.8)
            rect(g,x-15,y-21,30,42,C['resistor'],'#252525',1.2)
        if orient=='h':text(g,type,x,y+6,18,'#1121d1',weight='bold')
        else:text(g,type,x,y+6,17,'#1121d1',rotation=90,weight='bold')
    def module(g):
        # Local coordinate origin at the centre of 4.5 mm package; source x=346,y=379.
        # Ground plane fields above / below the VIN bus, and side grounds.
        rect(g,-346, -352, 236,206,C['pgnd'])
        rect(g,107, -352, 209,206,C['pgnd'])
        poly(g,[(-346,39),(-108,39),(-94,22),(94,22),(110,39),
                (316,39),(316,103),(-346,103)],C['pgnd'])
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
        rect(g,-94,22,188,76,C['pgnd'],'#3d3d3d',1,rx=8)
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

    rect(dwg,-margin,-margin,fullw+2*margin,h+2*margin,'#ffffff')
    # Three (or N) identical module cells. Module x position controlled by pitch_mm.
    for i in range(n):
        center = REF_CENTER_X + i*pitch
        g=dwg.g(id=f'module_{i+1}',transform=f'translate({center:.5f},{REF_CENTER_Y})')
        module(g)
        dwg.add(g)
    # The repeated VIN segments meet at every cell boundary (reference routing).
    dwg.save()
    try: cairosvg.svg2png(url=args.output+'.svg',write_to=args.output+'.png',output_width=2000)
    except Exception as e: print('PNG preview unavailable:',e)
    meta={'source_reference':'user PCB image', 'units':'mm', 'module_count':n,
          'module_footprint_mm':[fw,fh], 'module_pitch_mm':float(p['module_pitch_mm']),
          'estimated_inter_module_gap_mm':round(float(p['module_pitch_mm'])-fw,3),
          'bounds_mm':[(fullw+2*margin)/pxmm,(h+2*margin)/pxmm],
          'module_center_xy_mm':[[round((REF_CENTER_X+i*pitch+margin)/pxmm,4),round((REF_CENTER_Y+margin)/pxmm,4)] for i in range(n)],
          'net_colors':C,'fabrication_ready':False,
          'unknowns':['layer assignments','exact copper trace dimensions','PCB outline','via sizes and drills','copper weight','clearance and solder mask','pad coordinates and net connectivity'],
          'source_config_used':['geometry.module_footprint_mm','pcb.connections','pcb.layout']}
    Path(args.output+'.json').write_text(json.dumps(meta,indent=2))
    print('Wrote',args.output+'.svg',args.output+'.png',args.output+'.json')

if __name__=='__main__': main()
