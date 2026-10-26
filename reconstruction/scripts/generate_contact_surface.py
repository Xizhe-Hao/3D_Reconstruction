"""Rigid rectangular indenter travel constraints plus bounded minimum-curvature completion.
All heights/displacements in mm; preserve separate observed/contact/transition masks.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,re,time,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy import sparse,ndimage
from scipy.optimize import lsq_linear
from surface_roi_experiment import Region,export_surface,complete

ROOT=Path('outputs/rgbd_20261014');OUT=ROOT/os.environ.get('CONTACT_OUTPUT','surface_contact');BASE=ROOT/os.environ.get('SURFACE_BASE','surface_all');GRID=Path(os.environ.get('SURFACE_GRID','data/drive_test/capture_20261014_202456/sensor_grid.json'));SESSION=Path('data/drive_test/capture_20261014_202456');STATE=None

def laplacian(domain):
    ids=np.full(domain.shape,-1,int);ids[domain]=np.arange(domain.sum());rows=[];cols=[];vals=[]
    for y,x in np.argwhere(domain):
        k=ids[y,x];degree=0
        for yy,xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
            if 0<=yy<domain.shape[0] and 0<=xx<domain.shape[1] and domain[yy,xx]:rows.append(k);cols.append(ids[yy,xx]);vals.append(-1.);degree+=1
        rows.append(k);cols.append(k);vals.append(float(degree))
    return sparse.csr_matrix((vals,(rows,cols)),shape=(domain.sum(),domain.sum()))

def minimum_curvature(base,reference,domain,observed,contact,near,bottom,L):
    # Contact overwrites all optical candidates in its footprint; a 4mm ring is
    # explicitly reclassified as inferred transition, not silently kept observed.
    trusted=observed&~near;fixed=trusted|contact;unknown=domain&~fixed
    out=base.copy();out[contact]=bottom
    if unknown.any() and fixed.any():
        f=fixed[domain];u=unknown[domain];values=out[domain];target=reference[domain][u];target=np.where(np.isfinite(target),target,float(np.nanmedian(reference)))
        A=L[:,u];b=-(L[:,f]@values[f]);prior=.035
        A=sparse.vstack([A,prior*sparse.eye(u.sum())],format='csr');b=np.r_[b,prior*target]
        # The footprint itself is fixed to the plane. Bound the transition
        # to prevent curvature overshoot below the tool or above the rest surface.
        lower=np.full(u.sum(),min(bottom,float(np.nanmin(base)))-.5)
        upper=np.maximum(target+2.0,lower+1.)
        solution=lsq_linear(A,b,bounds=(lower,upper),tol=1e-7,lsmr_tol=1e-8,max_iter=100)
        if not solution.success:raise RuntimeError(solution.message)
        out[unknown]=solution.x
    inferred=domain&~trusted&np.isfinite(out)
    return out,trusted,inferred

def clean(a):return np.where(np.isfinite(a),a,0).tolist()

def frame_job(frame):
    region,cal,alignment,L=STATE;stem=f'frame_{frame:05d}';src=np.load(BASE/(stem+'.npz'));base=src['completed_height_mm'];raw=src['height_mm'];reference=src['reference_mm'];original=src['observed'];domain=src['domain'];row=alignment[frame]
    display_reference,reference_inferred=complete(reference,domain,2,200)
    value=row.get('force_distance','');distance=float(value) if value else None
    # No force-run record: no contact constraint is fabricated.
    u,v,w,length,angle,h0=cal['params'];delta=(distance-cal['initial_distance_mm']) if distance is not None else 0.
    bottom=h0+delta*region.axes[2,2];shift=delta*region.axes[2,:2];center=np.array([u,v])+shift
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);q=(region.uv-center)@rotation
    outside=np.maximum(abs(q)-[w/2,length/2],0);edge_distance=np.linalg.norm(outside,axis=-1);footprint=domain&(abs(q[:,:,0])<=w/2)&(abs(q[:,:,1])<=length/2)
    # Under the user's full rigid-face contact assumption. No forced upward
    # displacement while hovering; onset is estimated from the rest surface.
    ref_level=float(np.nanmedian(reference[footprint]));active=distance is not None and bottom<ref_level-.5
    contact=footprint if active else np.zeros(domain.shape,bool);near=domain&(edge_distance<=4) if active else np.zeros(domain.shape,bool)
    if active:
        completed,observed,inferred=minimum_curvature(base,reference,domain,original,contact,near,bottom,L)
        # Limit non-contact completion of large unsupported holes; contact plus
        # its surrounding 20mm zone are supported by explicit physical anchors.
        anchors=observed|contact;dist=ndimage.distance_transform_edt(~anchors)*2
        supported=(dist<=12)|(edge_distance<=20);completed[domain&~supported]=np.nan;inferred &= supported
    else:completed=base.copy();observed=original.copy();inferred=src['inferred'].copy()
    valid=np.isfinite(completed)&domain;transition=inferred&~contact
    assert np.array_equal(completed[observed],raw[observed])
    if active:assert np.max(abs(completed[contact]-bottom))<1e-9
    np.savez_compressed(OUT/(stem+'.npz'),uv_mm=region.uv,domain=domain,height_mm=raw,completed_height_mm=completed,reference_mm=reference,completed_reference_mm=display_reference,reference_inferred=reference_inferred,depression_mm=display_reference-completed,observed=observed,inferred=inferred,contact=contact,transition=transition,original_observed=original,reclassified_observed=original&~observed,force_distance_mm=distance if distance is not None else np.nan,indenter_bottom_mm=bottom if distance is not None else np.nan)
    parts=[];total=0
    for name,mask,color in [('observed',observed,(65,195,215)),('transition',transition,(245,155,50)),('contact',contact,(190,115,245))]:
        export_surface(OUT/(stem+'_'+name+'.ply'),region,completed,mask,color);b=(OUT/(stem+'_'+name+'.ply')).read_bytes();end=b.index(b'end_header\n')+11;parts.append(b[end:]);total+=int(mask.sum());head=b[:end]
    (OUT/(stem+'_display.ply')).write_bytes(re.sub(rb'element vertex \d+',f'element vertex {total}'.encode(),head)+b''.join(parts))
    corners=np.array([[-w/2,-length/2],[w/2,-length/2],[w/2,length/2],[-w/2,length/2]])@rotation.T+center
    world=(region.center+np.column_stack([corners,np.full(4,bottom)])@region.axes.T)/1000;projections=[]
    for K,E in zip(np.array(cal['display_intrinsics']),np.array(cal['extrinsics_world_to_camera_m'])):
        camera=world@E[:,:3].T+E[:,3];pixels=camera@K.T;projections.append((pixels[:,:2]/pixels[:,2:]).tolist())
    metadata=dict(projected_bottom_polygons=projections,frame=frame,time_s=float(row['video_time_s']),d455_dt_ms=float(row['d455_dt_ms']),d455_excluded=bool(src['d455_excluded']),force_distance_mm=distance,force_load=float(row['force_load']) if row['force_load'] else None,indenter_bottom_mm=bottom if distance is not None else None,contact_active=bool(active),contact_cells=int(contact.sum()),rectangle_uv_mm=corners.tolist(),initial_bottom_mm=h0,initial_distance_mm=cal['initial_distance_mm'])
    data=dict(shape=list(domain.shape),uv=clean(region.uv),height=clean(completed),depression=clean(display_reference-completed),valid=valid.tolist(),observed=observed.tolist(),inferred=inferred.tolist(),contact=contact.tolist(),transition=transition.tolist(),**metadata)
    (OUT/(stem+'.json')).write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    return dict(**metadata,observed=int(observed.sum()),inferred=int(inferred.sum()),domain=int(domain.sum()),reclassified_observed=int((original&~observed).sum()))

def tests():
    y,x=np.mgrid[:25,:25];domain=np.ones(x.shape,bool);L=laplacian(domain);ref=np.zeros(x.shape);base=ref.copy();contact=(abs(x-12)<=3)&(abs(y-12)<=4);near=ndimage.distance_transform_edt(~contact)<=3;obs=~near
    out,o,i=minimum_curvature(base,ref,domain,obs,contact,near,-10.,L)
    assert np.all(out[contact]==-10) and np.array_equal(out[o],base[o]);assert not (o&i).any();assert out.min()>=-10.5 and out.max()<=2
    assert np.isfinite(out).all();return {'passed':True,'checks':['rigid rectangular flat floor','observed values unchanged outside transition','bounded smooth interpolation','separate masks']}

def main():
    global STATE
    OUT.mkdir(exist_ok=True);test=tests();print(test,flush=True);cal=json.loads((OUT/'indenter_calibration.json').read_text());region=Region(json.loads(GRID.read_text()));alignment={int(r['video_frame_index']):r for r in csv.DictReader((SESSION/'multimodal_video_alignment.csv').open())};STATE=(region,cal,alignment,laplacian(region.domain));results=[];start=time.time()
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        for future in as_completed([pool.submit(frame_job,f) for f in range(871)]):
            results.append(future.result())
            if len(results)%50==0:print(len(results),'/871',round(time.time()-start,1),'s',flush=True)
    m=dict(frames=sorted(results,key=lambda f:f['frame']),count=len(results),calibration=cal,synthetic_tests=test,method='bounded biharmonic interpolation with rigid rectangular contact anchors; 4mm inferred transition band',limitations=['estimated indenter dimensions and initial height','full footprint contact assumption','curve-aligned travel has unquantified time uncertainty','no travel record means no rigid constraint','optical candidates in footprint and 4mm band explicitly reclassified','tag ROI clips any rectangle extending outside domain'])
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2));print('DONE',len(results),flush=True)
if __name__=='__main__':main()
