"""Analytical/topological checks; no CPU magnetic field solve."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from em_mesh import topology,filament_bundle,source_intersections


class EdgePreprocessing(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)]+[[0.,0.,0.]])*1e-3
        self.tet=Delaunay(self.xyz).simplices.astype(np.int32)
        determinant=np.linalg.det((self.xyz[self.tet[:,1:]]-self.xyz[self.tet[:,0,None]]).transpose(0,2,1))
        self.tet=self.tet[abs(determinant)>1e-20]

    def test_oriented_edges_and_boundary_constraints(self):
        edges,te,fixed,faces,owners=topology(self.xyz,self.tet)
        self.assertTrue(np.all(edges[:,0]<edges[:,1]))
        boundary=set(np.unique(faces[owners[:,1]<0]).tolist())
        self.assertNotIn(8,boundary)
        boundary_faces=faces[owners[:,1]<0]
        boundary_edges={tuple(sorted(f[[i,j]])) for f in boundary_faces for i,j in ((0,1),(0,2),(1,2))}
        for edge,flag in zip(edges,fixed):self.assertEqual(bool(flag),tuple(edge) in boundary_edges)
        for vertices,local in zip(self.tet,te):
            k=0
            for i in range(4):
                for j in range(i+1,4):
                    oriented=edges[abs(local[k])-1]
                    if local[k]<0:oriented=oriented[::-1]
                    np.testing.assert_array_equal(oriented,vertices[[i,j]]);k+=1

    def test_closed_volume_source_has_no_endpoint_charge(self):
        path=np.array([[-.4,-.4,.07],[.4,-.4,.07],[.4,.4,.07],[-.4,.4,.07],[-.4,-.4,.07]])
        paths,weight=filament_bundle(path,.05,2,8)
        source,a,b,w=source_intersections(self.xyz,self.tet,paths,weight)
        charge=np.zeros(len(self.xyz))
        np.add.at(charge,self.tet[source].ravel(),((b-a)*w[:,None]).ravel())
        self.assertLess(abs(charge).max(),1e-10)
        self.assertTrue(np.all(a>=-1e-9));self.assertTrue(np.all(b>=-1e-9))
        np.testing.assert_allclose(a.sum(axis=1),1);np.testing.assert_allclose(b.sum(axis=1),1)

    def test_open_and_reversing_paths_rejected(self):
        with self.assertRaisesRegex(ValueError,'closed'):filament_bundle([[0,0,0],[1,0,0],[1,1,0]],.1,1,8)
        with self.assertRaisesRegex(ValueError,'reverses'):filament_bundle([[0,0,0],[1,0,0],[0,0,0]],.1,1,8)


if __name__=='__main__':unittest.main()
