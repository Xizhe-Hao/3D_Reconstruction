import sys,json
from pathlib import Path
import numpy as np,cv2
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parent));from surface_roi_experiment import Region
out=Path('outputs/rgbd_20261014/surface_contact');out.mkdir(exist_ok=True)
root=Path('outputs/rgbd_20261014');reg=Region(json.loads(Path('data/drive_test/capture_20261014_202456/sensor_grid.json').read_text()));cal=np.load(root/'frames/00000/depths.npz');K=cal['intrinsics'].copy();K[:,0]*=640/1512;K[:,1]*=480/1134;E=cal['extrinsics'];print('E',E.shape)
rois=[(170,280,65,145),(375,458,50,125),(210,310,85,190),(465,575,0,65)];edges=[];imgs=[]
for cam,(x0,x1,y0,y1) in enumerate(rois):
 img=cv2.imread(str(root/f'frames/00000/rgb_{cam}.jpg'));gray=img.max(2);points=[]
 for x in range(x0,x1):
  ys=np.flatnonzero(gray[y0:y1,x]<55)
  if len(ys)>8 and ys[-1]+y0<y1-2:points.append([x,ys[-1]+y0])
 pts=np.array(points,float);edges.append(pts);imgs.append(img);print(cam,pts.min(0),pts.max(0),len(pts))
def corners(p):
 u,v,w,l,angle,h=p;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);uv=np.array([[-w/2,-l/2],[w/2,-l/2],[w/2,l/2],[-w/2,l/2]])@rot.T+[u,v];return uv

def project(p,cam):
 uv=corners(p);world=(reg.center+np.column_stack([uv,np.full(4,p[-1])])@reg.axes.T)/1000;c=world@E[cam,:,:3].T+E[cam,:,3];a=c@K[cam].T;return a[:,:2]/a[:,2:]
def residual(p):
 out=[]
 for cam,obs in enumerate(edges):
  poly=project(p,cam);lo=poly[:,0].min();hi=poly[:,0].max();xs=np.clip(obs[:,0],lo+1e-5,hi-1e-5);vals=[]
  for a,b in zip(poly,np.roll(poly,-1,axis=0)):
   if abs(b[0]-a[0])<1e-8:continue
   t=(xs-a[0])/(b[0]-a[0]);ys=a[1]+t*(b[1]-a[1]);ys[(t<0)|(t>1)]=-1e3;vals.append(ys)
  pred=np.max(vals,axis=0);out.extend((pred-obs[:,1]).tolist());out.extend((np.maximum(lo-obs[:,0],0)*3+np.maximum(obs[:,0]-hi,0)*3).tolist())
 return out
fit=least_squares(residual,[10,34,24,40,-.07,5],bounds=([-15,10,10,15,-.8,-5],[35,60,50,65,.8,20]),loss='soft_l1',f_scale=1.5,max_nfev=600)
print('FIT',fit.x,'rms',np.sqrt(np.mean(np.array(residual(fit.x))**2)))
for cam,img in enumerate(imgs):
 cv2.polylines(img,[project(fit.x,cam).round().astype(int)],True,(0,255,255),1)
 for x,y in edges[cam].astype(int):cv2.circle(img,(x,y),1,(0,0,255),-1)
cv2.imwrite(str(out/'indenter_initial_overlay.jpg'),np.vstack([np.hstack(imgs[:2]),np.hstack(imgs[2:])]))
(out/'indenter_calibration.json').write_text(json.dumps({'params':fit.x.tolist(),'residual_pixel_rms':float(np.sqrt(np.mean(np.array(residual(fit.x))**2))),'corners_uv_mm':corners(fit.x).tolist(),'initial_frame':0,'initial_distance_mm':0.05,'method':'robust four-view initial bottom silhouette fit; raw depth lower-edge cross-check','parameter_order':['center_u_mm','center_v_mm','width_mm','length_mm','angle_rad','initial_bottom_height_mm'],'assumptions':['rigid rectangular face parallel to reference plane','vertical world-z translation','dimensions estimated, not independently measured'],'edge_rois_xyxy':rois},indent=2))
cloud_checks=[]
for cam,obs in enumerate(edges):
 uv=np.column_stack([obs[:,0]*1512/640,(obs[:,1]-2)*1134/480]);uv=np.rint(uv).astype(int);z=cal['depths_m'][cam,uv[:,1],uv[:,0]];kh=cal['intrinsics'][cam];rays=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(kh).T;world=(rays*z[:,None]-E[cam,:,3])@E[cam,:,:3];q=(world*1000-reg.center)@reg.axes;print('edge cloud camera',cam,'h',np.percentile(q[:,2],[10,50,90]),'uv median',np.median(q[:,:2],axis=0))

 cloud_checks.append({'camera':cam,'height_p10_p50_p90_mm':np.percentile(q[:,2],[10,50,90]).tolist()})
config=json.loads((out/'indenter_calibration.json').read_text());config['raw_depth_lower_edge_checks']=cloud_checks;config['uncertainty_note']='initial bottom height and dimensions are estimated; per-view depth edges disagree by several mm';(out/'indenter_calibration.json').write_text(json.dumps(config,indent=2))
res=np.asarray(residual(fit.x));cursor=0;view_res=[]
for cam,edge in enumerate(edges):
 e=res[cursor:cursor+len(edge)];cursor+=2*len(edge);view_res.append({'camera':cam,'median_abs_bottom_edge_error_px':float(np.median(abs(e))),'p95_abs_bottom_edge_error_px':float(np.percentile(abs(e),95))})
config['bottom_edge_reprojection']=view_res;(out/'indenter_calibration.json').write_text(json.dumps(config,indent=2))
config['display_intrinsics']=K.tolist();config['extrinsics_world_to_camera_m']=E.tolist();(out/'indenter_calibration.json').write_text(json.dumps(config,indent=2))
