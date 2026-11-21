"""The reference's lower PGND band connects on the first copper layer."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import yaml
from matplotlib.path import Path as PolygonPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from pcb_3d import build_assembly


class GroundPlaneTests(unittest.TestCase):
    def test_first_layer_pgnd_connects_center_and_four_side_pins(self):
        config = yaml.safe_load((ROOT/'config.yaml').read_text())
        footprint = json.loads((ROOT/'data/footprint.json').read_text())
        model = build_assembly(config, footprint)
        self.assertFalse(any(p.get('layer')=='bottom' for p in model['pcb_copper']))
        expected = {'PGND_BOTTOM','L6','L7','R6','R7'}
        self.assertEqual({p['pad'] for p in model['pgnd_connections']}, expected)
        # All five lands must fit within ONE top-layer polygon, without relying on vias.
        cx,cy,_ = model['module_placements'][0]['translation_mm']
        contacts=[]
        foreign=[]
        for pad in footprint['pads']:
            x,y,w,h=pad['rect_mm']
            points=np.array([[cx+a,cy+b] for a in np.linspace(x+1e-5,x+w-1e-5,7)
                                           for b in np.linspace(y+1e-5,y+h-1e-5,7)])
            if pad['name'] in expected: contacts.extend(points)
            elif pad['net']!='PGND': foreign.extend(points)
        candidates=[p for p in model['pcb_copper'] if p['net_hint']=='PGND' and
                    PolygonPath(p['polygon_xy_mm']).contains_points(contacts).all()]
        self.assertTrue(candidates, 'No single top PGND polygon reaches all five lands')
        for polygon in candidates:
            self.assertFalse(PolygonPath(polygon['polygon_xy_mm']).contains_points(foreign).any(),
                             'PGND pour touches a different footprint net')
            self.assertEqual(polygon['z_min_mm'],config['pcb']['assembly_3d']['top_copper_bottom_z_mm'])


if __name__=='__main__':
    unittest.main()
