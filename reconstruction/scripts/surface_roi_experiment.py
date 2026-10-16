"""Exploratory tag-ROI surface fusion. Not production training labels.

Millimetres throughout. Dark-neutral filtering is a heuristic, not semantic
rod segmentation. Missing measurements and inferred values remain separate.
"""
import argparse,json,warnings
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage,sparse
from scipy.sparse.linalg import spsolve

def med(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);return np.nanmedian(a,axis=0)

def consensus(values,min_views=2,tolerance=3.):
    a=np.array(values);center=med(a);keep=np.isfinite(a)&(abs(a-center)<=tolerance)
    votes=keep.sum(0);out=med(np.where(keep,a,np.nan));out[votes<min_views]=np.nan
    return out,votes,med(np.where(keep,abs(a-out),np.nan))

class Region:
    def __init__(self,grid,step=2.):
        self.center=np.array(grid['centre_world']);x=np.array(grid['x_axis_world_C1_to_C16']);y=np.array(grid['y_axis_world_R1_to_R16']);n=np.array(grid['normal_world']);n=n if n[2]>0 else -n
        self.axes=np.stack([x,y,n],1);self.step=step
        self.tag_polys=[((np.array(q)-self.center)@self.axes)[:,:2].astype(np.float32) for q in grid.get("outer_tag_quads_world",{}).values()]
        corners=np.array([grid['corners_world'][k] for k in ['R1C1','R1C16','R16C16','R16C1']]);self.poly=((corners-self.center)@self.axes)[:,:2].astype(np.float32)
        self.lo=self.poly.min(0);hi=self.poly.max(0);self.shape=tuple(np.ceil((hi-self.lo)[::-1]/step).astype(int));v,u=np.indices(self.shape)
        self.uv=np.stack([self.lo[0]+(u+.5)*step,self.lo[1]+(v+.5)*step],-1)
        self.domain=np.array([cv2.pointPolygonTest(self.poly,tuple(map(float,p)),False)>=0 for p in self.uv.reshape(-1,2)]).reshape(self.shape)
    def heights(self,path,reference=None):
        blob=path.read_bytes();off=blob.index(b'end_header\n')+11
        dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')]);a=np.frombuffer(blob,dt,offset=off)
        p=np.stack([a[k] for k in 'xyz'],1)*1000;c=np.stack([a[k] for k in 'rgb'],1);q=(p-self.center)@self.axes
        u,v=np.floor((q[:,:2]-self.lo)/self.step).astype(int).T
        keep=(u>=0)&(u<self.shape[1])&(v>=0)&(v<self.shape[0])&(q[:,2]>-35)&(q[:,2]<8)
        dark=(c.max(1)<65)&(c.max(1).astype(int)-c.min(1)<25);
        if self.tag_polys:
            # AprilTag black pixels belong to the requested outer surface ROI.
            tag=np.zeros(len(q),bool)
            for poly in self.tag_polys:
                candidates=np.flatnonzero(dark&keep&(q[:,2]<3)&np.all(q[:,:2]>=poly.min(0)-1,axis=1)&np.all(q[:,:2]<=poly.max(0)+1,axis=1))
                for k in candidates:
                    if cv2.pointPolygonTest(poly,tuple(map(float,q[k,:2])),True)>=-1:tag[k]=True
            dark &= ~tag
        keep &= ~dark
        idx=np.flatnonzero(keep);idx=idx[self.domain[v[idx],u[idx]]]
        if reference is not None:
            lim=reference[v[idx],u[idx]]+3;idx=idx[np.isfinite(lim)&(q[idx,2]<=lim)]
        flat=v[idx]*self.shape[1]+u[idx];order=np.argsort(flat);flat=flat[order];z=q[idx[order],2]
        out=np.full(self.shape,np.nan);ids,starts=np.unique(flat,return_index=True)
        for k,chunk in zip(ids,np.split(z,starts[1:])):
            if len(chunk)>=3:out.flat[k]=np.median(chunk)
        return out

def complete(height,domain,step,max_distance=12.):
    """Laplace baseline, fixed measurements, no flux at ROI edge, distance cap."""
    observed=np.isfinite(height)&domain;missing=domain&~observed;out=height.copy();predicted=np.zeros(domain.shape,bool)
    if not observed.any() or not missing.any():return out,predicted
    # Do not solve domain components having no measured boundary/anchor.
    labels,n=ndimage.label(domain)
    supported=np.isin(labels,[i for i in range(1,n+1) if (observed&(labels==i)).any()]);missing &= supported
    if not missing.any():return out,predicted
    coords=np.argwhere(missing);index=np.full(domain.shape,-1,int);index[missing]=np.arange(len(coords));rows=[];cols=[];data=[];rhs=np.zeros(len(coords))
    for k,(y,x) in enumerate(coords):
        degree=0
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if not (0<=yy<domain.shape[0] and 0<=xx<domain.shape[1] and supported[yy,xx]):continue
            degree+=1
            if observed[yy,xx]:rhs[k]+=height[yy,xx]
            else:rows.append(k);cols.append(index[yy,xx]);data.append(-1.)
        rows.append(k);cols.append(k);data.append(float(degree))
    sol=spsolve(sparse.csr_matrix((data,(rows,cols)),shape=(len(coords),len(coords))),rhs)
    distance=ndimage.distance_transform_edt(~observed)*step;predicted=missing&(distance<=max_distance)
    temp=np.full(domain.shape,np.nan);temp[missing]=sol;out[predicted]=temp[predicted]
    return out,predicted

def holdout(height,domain,step,seed=73):
    rng=np.random.default_rng(seed);candidates=np.argwhere(np.isfinite(height)&domain);yy,xx=np.indices(height.shape);results=[]
    for radius in (4,8,12):
        trials=[]
        for y,x in candidates[rng.permutation(len(candidates))[:30]]:
            disk=((yy-y)**2+(xx-x)**2)*step**2<=radius**2;target=disk&np.isfinite(height)&domain
            if target.sum()<8 or (disk&domain).sum()<disk.sum():continue
            hidden=height.copy();hidden[disk]=np.nan;estimate,pred=complete(hidden,domain,step,radius+step);valid=target&pred&np.isfinite(estimate)
            if valid.sum()<8:continue
            err=estimate[valid]-height[valid];trials.append({'cells':int(valid.sum()),'mae_mm':float(abs(err).mean()),'bias_mm':float(err.mean()),'p95_mm':float(np.percentile(abs(err),95))})
        results.append({'radius_mm':radius,'trials':len(trials),'median_trial_mae_mm':float(np.median([t['mae_mm'] for t in trials])) if trials else None,'worst_trial_mae_mm':max([t['mae_mm'] for t in trials],default=None),'details':trials})
    return results

def synthetic_tests():
    y,x=np.mgrid[-30:31,-30:31];domain=np.ones(x.shape,bool);plane=1+.02*x-.03*y;mask=x*x+y*y<=8**2
    a=plane.copy();a[mask]=np.nan;out,pred=complete(a,domain,1,12)
    assert pred.sum()==mask.sum() and np.max(abs(out[mask]-plane[mask]))<1e-8
    assert np.array_equal(out[~mask],plane[~mask])
    a=np.full((21,21),np.nan);out,pred=complete(a,np.ones(a.shape,bool),1);assert not pred.any() and np.isnan(out).all()
    a=plane.copy();a[x*x+y*y<20**2]=np.nan;out,pred=complete(a,domain,1,5);assert np.isnan(out[30,30])
    vals=np.array([np.full((2,2),v) for v in [2.,2.2,2.1,15.,np.nan]]);out,votes,_=consensus(vals);assert np.all(votes==3) and np.allclose(out,2.1)
    results=[]
    for radius in (4,8,12):
        truth=plane-12*np.exp(-(x*x+y*y)/(2*8**2));mask=x*x+y*y<=radius**2;a=truth.copy();a[mask]=np.nan;out,_=complete(a,domain,1,radius+2);err=out[mask]-truth[mask]
        results.append({'radius_mm':radius,'mae_mm':float(abs(err).mean()),'center_underestimated_depression_mm':float(out[30,30]-truth[30,30])})
    return {'assertions':'passed: plane recovery, observations unchanged, no-data refusal, distance cap, view outlier rejection','hidden_gaussian_dent_12mm':results}

def export_surface(path,region,height,mask,color):
    uv=region.uv[mask];h=height[mask];local=np.column_stack([uv,h]);world=(region.center+local@region.axes.T)/1000
    a=np.empty(len(world),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,v in zip(('x','y','z'),world.T):a[k]=v
    for k,v in zip(('red','green','blue'),color):a[k]=v
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'%len(a)).encode());a.tofile(f)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--grid',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    region=Region(json.loads(args.grid.read_text()));sources=[f'camera_{i}' for i in range(4)]+['d455']
    refs=[med([region.heights(args.run/'frames'/f'{frame:05d}'/f'{s}.ply') for frame in (0,10,20,30,40)]) for s in sources];reference,_,_=consensus(refs)
    report={'synthetic':synthetic_tests(),'step_mm':2,'roi':'actual tag inner-corner quadrilateral, not assumed 120mm active sensor','grid_shape':list(map(int,region.shape)),'frames':[],'limitations':['dark neutral filtering is heuristic, not semantic rod segmentation','existing sampled/voxelized PLY, not original depths','reference frames 0,10,20,30,40 must be verified unloaded','holdout evaluates visible surface interpolation, not actual hidden contact truth']}
    for frame in (0,200,435,500,800):
        heights=[region.heights(args.run/'frames'/f'{frame:05d}'/f'{s}.ply',r) for s,r in zip(sources,refs)]
        delta,votes,spread=consensus([r-h for r,h in zip(refs,heights)]);height=np.where(np.isfinite(delta)&np.isfinite(reference),reference-delta,np.nan)
        filled,predicted=complete(height,region.domain,2,12);observed=np.isfinite(height)&region.domain
        np.savez_compressed(args.output/f'frame_{frame:05d}.npz',uv_mm=region.uv,domain=region.domain,height_mm=height,completed_height_mm=filled,observed=observed,inferred=predicted,depression_mm=delta,views=votes,view_mad_mm=spread,reference_mm=reference)
        export_surface(args.output/f'frame_{frame:05d}_observed.ply',region,height,observed,(65,195,215))
        export_surface(args.output/f'frame_{frame:05d}_inferred.ply',region,filled,predicted,(245,155,50))
        result={'frame':frame,'domain_cells':int(region.domain.sum()),'measured_cells':int(observed.sum()),'measured_fraction':float(observed.sum()/region.domain.sum()),'inferred_cells':int(predicted.sum()),'median_interview_mad_mm':float(np.nanmedian(spread[observed])),'holdout':holdout(height,region.domain,2)};report['frames'].append(result);print(json.dumps({k:v for k,v in result.items() if k!='holdout'}),flush=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,5,figsize=(17,9),constrained_layout=True)
    for col,frame in enumerate((0,200,435,500,800)):
        with np.load(args.output/f'frame_{frame:05d}.npz') as d:
            for row,key in enumerate(('height_mm','completed_height_mm','inferred')):
                im=axes[row,col].imshow(d[key],origin='lower',cmap='viridis' if row<2 else 'gray',vmin=-15 if row<2 else 0,vmax=5 if row<2 else 1);axes[row,col].set_title(f'Frame {frame}: '+['observed mm','observed + inferred mm','inferred mask'][row]);fig.colorbar(im,ax=axes[row,col],shrink=.6)
    fig.savefig(args.output/'surface_experiment.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
