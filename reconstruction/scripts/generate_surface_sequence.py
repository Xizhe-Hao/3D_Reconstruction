"""Batch the exploratory surface baseline and publish synchronized frame metadata."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,multiprocessing as mp,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from surface_roi_experiment import Region,med,consensus,complete,export_surface,synthetic_tests

STATE=None

def clean(a):
    return np.where(np.isfinite(a),a,0).tolist()

def generate(frame):
    region,refs,reference,run,out=STATE
    folder=run/'frames'/f'{frame:05d}';metrics=json.loads((folder/'metrics.json').read_text())
    sources=[f'camera_{i}' for i in range(4)]+['d455']
    heights=[region.heights(folder/f'{s}.ply',r) for s,r in zip(sources,refs)]
    excluded=abs(metrics.get('d455_dt_ms',0))>40
    if excluded:heights[-1][:]=np.nan
    delta,votes,spread=consensus([r-h for r,h in zip(refs,heights)])
    height=np.where(np.isfinite(delta)&np.isfinite(reference),reference-delta,np.nan)
    filled,pred=complete(height,region.domain,2,12);obs=np.isfinite(height)&region.domain
    assert not (obs&pred).any() and np.array_equal(filled[obs],height[obs])
    stem=f'frame_{frame:05d}'
    np.savez_compressed(out/(stem+'.npz'),uv_mm=region.uv,domain=region.domain,height_mm=height,completed_height_mm=filled,observed=obs,inferred=pred,depression_mm=delta,completed_depression_mm=reference-filled,views=votes,view_mad_mm=spread,reference_mm=reference,d455_excluded=excluded)
    export_surface(out/(stem+'_observed.ply'),region,height,obs,(65,195,215))
    export_surface(out/(stem+'_inferred.ply'),region,filled,pred,(245,155,50))
    chunks=[];total=0
    for suffix in ('observed','inferred'):
        b=(out/(stem+'_'+suffix+'.ply')).read_bytes();offset=b.index(b'end_header\n')+11;chunks.append(b[offset:]);total+=len(b[offset:])//15;head=b[:offset]
    import re
    (out/(stem+'_display.ply')).write_bytes(re.sub(rb'element vertex \d+',f'element vertex {total}'.encode(),head)+b''.join(chunks))
    data=dict(shape=list(height.shape),uv=clean(region.uv),height=clean(filled),depression=clean(reference-filled),valid=np.isfinite(filled).tolist(),observed=obs.tolist(),inferred=pred.tolist(),frame=frame,time_s=metrics['time_s'],d455_dt_ms=metrics.get('d455_dt_ms'),d455_excluded=excluded,force_load=metrics.get('force_load'))
    (out/(stem+'.json')).write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    return dict(frame=frame,time_s=metrics['time_s'],observed=int(obs.sum()),inferred=int(pred.sum()),domain=int(region.domain.sum()),d455_excluded=excluded)

def main():
    global STATE
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(synthetic_tests()),flush=True)
    region=Region(json.loads(a.grid.read_text()));sources=[f'camera_{i}' for i in range(4)]+['d455']
    refs=[med([region.heights(a.run/'frames'/f'{frame:05d}'/f'{s}.ply') for frame in (0,10,20,30,40)]) for s in sources];reference,_,_=consensus(refs)
    STATE=(region,refs,reference,a.run,a.output)
    frames=[int(p.parent.name) for p in sorted((a.run/'frames').glob('*/metrics.json'))];results=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        pending={pool.submit(generate,f):f for f in frames}
        for future in as_completed(pending):
            results.append(future.result())
            if len(results)%25==0 or len(results)==len(frames):print(f'{len(results)}/{len(frames)} frames, {time.time()-start:.1f}s',flush=True)
    manifest=dict(frames=sorted(results,key=lambda f:f['frame']),count=len(results),step_mm=2,reference_frames=[0,10,20,30,40],limitations=['heuristic rod filtering','unverified unloaded reference','inferred surface is interpolation, not measured contact truth','tag ROI differs from configured active electrode area'],d455_excluded_count=sum(f['d455_excluded'] for f in results))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',len(results),'frames',flush=True)
if __name__=='__main__':main()
