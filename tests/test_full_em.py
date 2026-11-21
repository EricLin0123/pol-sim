"""Physics checks for full-domain FEM preprocessing and switched CCM playback."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from em_full import element_geometry,buck_waveform


class FullDomainPhysics(unittest.TestCase):
    def test_affine_tetrahedron_gradients_and_volume(self):
        points=np.array([[0.,0.,0.],[2.,0.,0.],[0.,3.,0.],[0.,0.,4.]])
        grad,vol=element_geometry(points,np.array([[0,1,2,3]]))
        self.assertAlmostEqual(vol[0],4.)
        np.testing.assert_allclose(grad.sum(axis=1),0.,atol=1e-15)
        # Linear coordinate fields must reproduce their exact constant gradients.
        np.testing.assert_allclose(points.T@grad[0],np.eye(3),atol=1e-15)

    def config(self):
        return dict(electrical=dict(operating_point=dict(vin_v=12.,vout_v=3.,iout_a=10.,switching_frequency_khz=500.)),
                    inductor=dict(nominal_inductance_h=1e-6),
                    em=dict(animation=dict(periods=2,samples_per_period=20)))

    def test_ccm_ramps_follow_inductor_voltage_in_each_switch_state(self):
        config=self.config();t,on,current,f,duty,ripple,mean=buck_waveform(config)
        slope=np.diff(current)/np.diff(t)
        mid=(t[:-1]+t[1:])/2
        expected=np.where(np.mod(mid*f,1)<duty,(12.-3.)/1e-6,-3./1e-6)
        np.testing.assert_allclose(slope,expected,rtol=1e-10)
        self.assertAlmostEqual(current.max()-current.min(),ripple)
        self.assertAlmostEqual((current.max()+current.min())/2,mean)
        self.assertTrue(np.all(np.diff(t)>0))
        corners=np.flatnonzero(np.isclose(np.mod(t*f,1),duty))
        self.assertTrue(len(corners)==2)
        self.assertFalse(on[corners].any())

    def test_discontinuous_conduction_is_not_silently_simulated(self):
        config=copy.deepcopy(self.config())
        config['electrical']['operating_point']['iout_a']=.01
        with self.assertRaisesRegex(ValueError,'continuous conduction'):buck_waveform(config)


if __name__=='__main__':unittest.main()
