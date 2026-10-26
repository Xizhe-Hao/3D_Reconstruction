"""Select outward tag corners without changing the original local coordinate frame."""
import sys,json
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'calibration'))
from sensor_grid import load_rig,triangulate
session=Path('data/drive_test/capture_20261014_202456');grid=json.loads((session/'sensor_grid.json').read_text());rig=load_rig(session/'calibration.json');center=np.array(grid['centre_world']);normal=np.array(grid['normal_world']);labels=['R1C1','R1C16','R16C16','R16C1'];outer={};checks={};quads={}
for tag,label in enumerate(labels):
 seen={int(v):np.array(d[str(tag)]) for v,d in grid['tag_detections'].items() if str(tag) in d};points=[];errors=[]
 for k in range(4):
  if len(seen)>=2:
   point,err=triangulate(rig,{v:q[k] for v,q in seen.items()});errors.append(err)
  else:
   v=next(iter(seen));cam=rig[v];xy=cv2.undistortPoints(seen[v][k].reshape(1,1,2),cam['K'],cam['dist']).reshape(2);origin=-cam['R'].T@cam['t'];direction=cam['R'].T@np.r_[xy,1.];point=origin+direction*((center-origin)@normal)/(direction@normal)
  points.append(point)
 points=np.array(points);idx=int(np.argmax(np.linalg.norm(points-center,axis=1)));outer[label]=points[idx].tolist();quads[label]=points.tolist();checks[label]={'outer_corner_index':idx,'views':list(seen),'method':'triangulation' if len(seen)>=2 else 'single-view ray/reference-plane intersection','max_reprojection_error_px':max(errors) if errors else None}
result=dict(grid);result['inner_corners_world']=grid['corners_world'];result['corners_world']=outer;result['roi_definition']='outward detected black-border corner of each AprilTag; excludes white paper margin';result['outer_tag_quads_world']=quads;result['outer_roi_checks']=checks
out=Path('outputs/rgbd_20261014/outer_tag_roi.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(checks));print('edges mm',[float(np.linalg.norm(np.array(outer[labels[i]])-outer[labels[(i+1)%4]])) for i in range(4)])
# Compare inner and outer ROI in the undistorted reconstruction image frame.
cal=np.load('outputs/rgbd_20261014/frames/00010/depths.npz');K=cal['intrinsics'].copy();K[:,0]*=640/1512;K[:,1]*=480/1134;imgs=[]
for v in range(4):
 img=cv2.imread(f'outputs/rgbd_20261014/frames/00010/rgb_{v}.jpg');E=cal['extrinsics'][v]
 for corners,color in [(grid['corners_world'],(0,150,255)),(outer,(70,255,70))]:
  world=np.array([corners[k] for k in labels])/1000;cam=world@E[:,:3].T+E[:,3];uv=cam@K[v].T;uv=uv[:,:2]/uv[:,2:];cv2.polylines(img,[uv.round().astype(int)],True,color,2)
 cv2.putText(img,'GREEN: outer tags / ORANGE: previous inner ROI',(12,460),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);imgs.append(img)
cv2.imwrite('outputs/rgbd_20261014/outer_tag_roi_comparison.jpg',np.vstack([np.hstack(imgs[:2]),np.hstack(imgs[2:])]))
