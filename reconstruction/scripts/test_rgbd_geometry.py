"""Known-coordinate checks for metric fusion and binary PLY export."""
import tempfile
from pathlib import Path
import numpy as np
from reconstruct_rgbd_web import backproject,voxel,compare_depth,save_ply

def main():
    K=np.eye(3);E=np.array([[0,-1,0,-.07],[1,0,0,-.08],[0,0,1,.3]])
    depth=np.array([[.35]],dtype=np.float32);rgb=np.array([[[200,100,50]]],dtype=np.uint8)
    p,c=backproject(depth,K,E,rgb,stride=1)
    np.testing.assert_allclose(p,[[.08,-.07,.05]],atol=1e-7)
    np.testing.assert_array_equal(c,[[200,100,50]])
    stats,_=compare_depth(p,depth,K,E)
    assert stats['overlap_pixels']==1 and stats['median_abs_mm']<1e-4
    p2,c2=voxel(np.array([[.0011,0,0],[.0013,0,0],[.0031,0,0]]),np.array([[0,0,0],[100,100,100],[200,200,200]],np.uint8))
    np.testing.assert_allclose(p2[:,0],[.0012,.0031]);np.testing.assert_array_equal(c2[:,0],[50,200])
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'cloud.ply';save_ply(path,p,c);blob=path.read_bytes();header,data=blob.split(b'end_header\n')
        assert b'element vertex 1' in header and len(data)==15
        np.testing.assert_allclose(np.frombuffer(data[:12],dtype='<f4'),p[0],atol=1e-7)
        assert data[12:]==bytes([200,100,50])
    print('PASS: metric camera/world rotation+translation, depth reprojection, voxel averaging, binary PLY')
if __name__=='__main__':main()
