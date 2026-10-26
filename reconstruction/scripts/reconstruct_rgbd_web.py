"""Calibrated DA3 + measured D455 clouds, with reproducible browser exports."""
from pathlib import Path
import argparse, json, csv, time, hashlib
import numpy as np
import cv2
import torch
from test_session_adapter import load_session_metadata, _decode_camera
from depth_anything_3.api import DepthAnything3

ROOT=Path(__file__).resolve().parents[1]

def save_ply(path, points, colors):
    a=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,v in zip(('x','y','z'),points.T): a[k]=v
    for k,v in zip(('red','green','blue'),colors.T): a[k]=v
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'%len(a)).encode());a.tofile(f)

def voxel(points,colors,size=.001):
    if not len(points): return points,colors
    _,idx,inv=np.unique(np.floor(points/size).astype(np.int32),axis=0,return_index=True,return_inverse=True)
    n=np.bincount(inv)
    p=np.stack([np.bincount(inv,weights=points[:,k])/n for k in range(3)],axis=1)
    c=np.stack([np.bincount(inv,weights=colors[:,k])/n for k in range(3)],axis=1).astype(np.uint8)
    return p,c

def crop(p):
    return np.isfinite(p).all(1)&(p[:,0]>-.05)&(p[:,0]<.22)&(p[:,1]>-.23)&(p[:,1]<.08)&(p[:,2]>-.04)&(p[:,2]<.20)

def backproject(depth,K,E,rgb,stride=2):
    v,u=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[::stride,::stride].ravel(); pix=np.stack([u.ravel(),v.ravel(),np.ones(u.size)],axis=1)
    p=((pix@np.linalg.inv(K).T)*z[:,None]-E[:,3])@E[:,:3]
    valid=(z>0)&crop(p)
    return p[valid],rgb[::stride,::stride].reshape(-1,3)[valid]

def measured(session,row,world,meta):
    raw=cv2.imread(str(session/row['d455_depth_png']),cv2.IMREAD_UNCHANGED)
    if raw is None:raise ValueError('Missing D455 depth')
    z=raw.astype(np.float32)*meta['depth_scale_m'];c=world['depth_correction'];valid=z>0
    z[valid]=(z[valid]-c['offset_mm']/1000)/(1+c['scale_error'])
    z[(z<meta['depth_min_range_m'])|(z>meta['depth_max_range_m'])]=0
    # Suppress edge flying pixels where neighbouring depths differ by > 12 mm.
    er=cv2.erode(z,np.ones((3,3),np.uint8));di=cv2.dilate(z,np.ones((3,3),np.uint8));z[(di-er)>.012]=0
    intr=meta['depth_intrinsics'];K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1]])
    E=np.column_stack([world['R_world_to_depth'],np.array(world['t_world_to_depth'])/1000])
    p,_=backproject(z,K,E,np.zeros((*z.shape,3),np.uint8))
    rgb=cv2.cvtColor(cv2.imread(str(session/row['d455_color_jpg'])),cv2.COLOR_BGR2RGB)
    cam=p@np.array(world['R_world_to_color']).T+np.array(world['t_world_to_color'])/1000
    intr=meta['color_intrinsics'];uv=cam[:,:2]/cam[:,2:]
    # Inverse Brown-Conrady maps distorted coordinates to ideal coordinates.
    # Invert that polynomial to map ideal projected rays to recorded pixels.
    if 'inverse_brown' in intr.get('model',''):
        uv=cv2.undistortPoints(uv.reshape(-1,1,2),np.eye(3),np.array(intr['coeffs'])).reshape(-1,2)
    uv=uv*np.array([intr['fx'],intr['fy']])+np.array([intr['ppx'],intr['ppy']]);uv=np.rint(uv).astype(int)
    inside=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
    colors=np.tile(np.array([65,195,215],np.uint8),(len(p),1));colors[inside]=rgb[uv[inside,1],uv[inside,0]]
    return p,colors,z,K,E,rgb

def compare_depth(p,depth,K,E):
    cam=p@E[:,:3].T+E[:,3];uvw=cam@K.T;uv=np.rint(uvw[:,:2]/uvw[:,2:]).astype(int)
    valid=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])
    uv=uv[valid];z=cam[valid,2];render=np.full(depth.size,np.inf)
    np.minimum.at(render,uv[:,1]*depth.shape[1]+uv[:,0],z);render=render.reshape(depth.shape)
    mask=np.isfinite(render)&(depth>0);diff=(render[mask]-depth[mask])*1000
    stats={'overlap_pixels':int(mask.sum())}
    if len(diff):stats.update(median_abs_mm=float(np.median(abs(diff))),p90_abs_mm=float(np.percentile(abs(diff),90)),signed_median_mm=float(np.median(diff)))
    img=np.zeros((*depth.shape,3),np.uint8)
    if len(diff):img[mask]=cv2.applyColorMap(np.clip(abs(diff)/20*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO).reshape(-1,3)
    return stats,img

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resolution',type=int,default=1008);ap.add_argument('--frames',default='0');ap.add_argument('--device',default='cuda:0');ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    torch.set_num_threads(6);cv2.setNumThreads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cameras,rows,export=load_session_metadata(args.session);rows={int(r['video_frame_index']):r for r in rows}
    world=json.loads((ROOT.parent/'calibration/d455_world.json').read_text());meta=json.loads((args.session/'d455/d455.json').read_text())
    if str(world['serial'])!=str(meta['serial']):raise ValueError('D455 serial mismatch')
    frames=[int(f) for f in args.frames.split(',')]
    model=DepthAnything3.from_pretrained(str(ROOT/'outputs/models/DA3-GIANT')).to(args.device).eval()
    config={'pipeline_version':1,'model_revision':'7cd62ae9315b9dff094d2d300e4ad012640607dd','resolution':args.resolution,'calibration_sha256':hashlib.sha256((args.session/'calibration.json').read_bytes()).hexdigest(),'d455_sha256':hashlib.sha256((ROOT.parent/'calibration/d455_world.json').read_bytes()).hexdigest()}
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    for frame in frames:
        dest=out/f'{frame:05d}';dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists() and json.loads((dest/'metrics.json').read_text()).get('fingerprint')==fingerprint:continue
        start=time.perf_counter();images=[];ks=[];exts=[]
        for camera in cameras:
            images_v,K=_decode_camera(camera,[frame],camera.image_size,False)
            images.append(images_v[0].transpose(1,2,0));ks.append(K);exts.append(camera.extrinsic_w2c_m)
        K=np.array(ks);E=np.array(exts);E4=np.repeat(np.eye(4,dtype=np.float32)[None],4,0);E4[:,:3]=E
        torch.cuda.reset_peak_memory_stats(args.device)
        prediction=model.inference(images,extrinsics=E4.copy(),intrinsics=K.copy(),align_to_input_ext_scale=True,process_res=args.resolution,process_res_method='upper_bound_resize')
        torch.cuda.synchronize(args.device)
        raw=np.asarray(prediction.depth,np.float32);conf=np.asarray(prediction.conf,np.float32);kh=np.asarray(prediction.intrinsics);eh=np.asarray(prediction.extrinsics)
        expected=K.copy();expected[:,0]*=raw.shape[2]/images[0].shape[1];expected[:,1]*=raw.shape[1]/images[0].shape[0]
        if not np.allclose(kh,expected,rtol=1e-4,atol=1e-3) or not np.allclose(eh,E,atol=1e-5):raise ValueError('DA3 changed calibrated field of view or poses')
        stats={'frame':frame,'time_s':frame/export['fps'],'resolution':args.resolution,'depth_shape':list(raw.shape),'peak_allocated_gib':torch.cuda.max_memory_allocated(args.device)/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved(args.device)/1024**3,'inference_decode_s':time.perf_counter()-start,'fingerprint':fingerprint}
        if args.benchmark:
            (out/f'benchmark_{args.resolution}.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True);return
        clouds=[];allcolors=[]
        for v in range(4):
            valid=np.isfinite(raw[v])&(raw[v]>0)&np.isfinite(conf[v]);threshold=np.percentile(conf[v][valid],20);depth=np.where(valid&(conf[v]>=threshold),raw[v],0)
            rgb=cv2.resize(images[v],(raw.shape[2],raw.shape[1]),interpolation=cv2.INTER_AREA)
            p,c=backproject(depth,kh[v],E[v],rgb);clouds.append(p);allcolors.append(c)
            pv,cv=voxel(p,c);save_ply(dest/f'camera_{v}.ply',pv,cv)
            cv2.imwrite(str(dest/f'rgb_{v}.jpg'),cv2.cvtColor(cv2.resize(rgb,(640,480)),cv2.COLOR_RGB2BGR))
            visual=cv2.applyColorMap(np.clip((raw[v]-.15)/.65*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);visual[depth==0]=0;cv2.imwrite(str(dest/f'depth_{v}.jpg'),visual)
        p=np.concatenate(clouds);c=np.concatenate(allcolors);pr,cr=voxel(p,c);save_ply(dest/'rgb4.ply',pr,cr)
        pd,cd,zd,kd,ed,dimg=measured(args.session,rows[frame],world,meta);pd,cd=voxel(pd,cd);save_ply(dest/'d455.ply',pd,cd)
        pf,cf=voxel(np.concatenate([pr,pd]),np.concatenate([cr,cd]));save_ply(dest/'fused.ply',pf,cf)
        comparison,errorimg=compare_depth(p,zd,kd,ed);stats.update(comparison)
        stats.update(rgb_points=len(pr),d455_points=len(pd),fused_points=len(pf),d455_dt_ms=float(rows[frame]['d455_dt_ms']),force_load=float(rows[frame]['force_load']) if rows[frame]['force_load'] else None)
        cv2.imwrite(str(dest/'d455_rgb.jpg'),cv2.cvtColor(dimg,cv2.COLOR_RGB2BGR));cv2.imwrite(str(dest/'difference.jpg'),errorimg)
        np.savez_compressed(dest/'depths.npz',depths_m=raw,confidence=conf,intrinsics=kh,extrinsics=E)
        stats['total_s']=time.perf_counter()-start
        (dest/'metrics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
        del prediction
        torch.cuda.empty_cache()
if __name__=='__main__':main()
