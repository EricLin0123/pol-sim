"""Buck topology and conductor isolation checks for the assumed internal wiring."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from geometry import build_model
from pcb_3d import build_assembly


class BuckInterconnectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load((ROOT/'config.yaml').read_text())
        cls.footprint=json.loads((ROOT/'data/footprint.json').read_text())
        cls.model=build_model(cls.config,cls.footprint,demo_winding=True)

    def test_expected_net_connections_and_separate_fet_terminals(self):
        topology=self.model['buck_topology']
        self.assertEqual({(c['net'],c['start'],c['end']) for c in topology['connections']},
            {('VIN','VIN','HS_D'),('SW','HS_S','SW'),('SW','SW','LS_D'),
             ('SW','SW','L_IN'),('VOUT','L_OUT','VOUT'),('PGND','LS_S','PGND')})
        for path in topology['connections']:
            np.testing.assert_allclose(path['points_mm'][0],topology['nodes_mm'][path['start']],atol=1e-9)
            np.testing.assert_allclose(path['points_mm'][-1],topology['nodes_mm'][path['end']],atol=1e-9)
        contacts=[b for b in self.model['bodies'] if b['name'].startswith('contact_')]
        self.assertEqual(len(contacts),4)
        # Switching channels remain silicon; no copper connection bridges D to S.
        self.assertEqual(len(topology['devices']),2)
        self.assertEqual(topology['inductor']['output_net'],'VOUT')
        board=build_assembly(self.config,self.footprint)
        self.assertEqual(board['module_definition']['buck_topology'],topology)

    def test_power_clips_are_solid_connected_and_isolated(self):
        self.assertFalse(self.model['interconnect_capsules'])
        clips=[b for b in self.model['bodies'] if b['name'].startswith('clip_')]
        self.assertTrue(clips)
        def overlap(a,b):
            return all(min(a['max_mm'][j],b['max_mm'][j])-max(a['min_mm'][j],b['min_mm'][j])>1e-9 for j in range(3))
        other=[b for b in self.model['bodies'] if b['name'].startswith('contact_')]+self.model['pin_solids']
        for i,a in enumerate(clips):
            for b in clips[i+1:]+other:
                if a['net']!=b['net']:
                    self.assertFalse(overlap(a,b),f"{a['name']} shorts {b['name']}")
        for path in self.model['buck_topology']['connections']:
            parts=[b for b in clips if b['connection']==path['name']]
            for a,b in zip(parts,parts[1:]):
                self.assertTrue(overlap(a,b),path['name'])
            for point in (path['points_mm'][0],path['points_mm'][-1]):
                self.assertTrue(any(all(b['min_mm'][j]<=point[j]<=b['max_mm'][j] for j in range(3)) for b in parts))


if __name__=='__main__': unittest.main()
