"""Audit frame coverage, binary PLY payloads and cached depth shapes."""
import argparse,json,zipfile,statistics
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    run=json.loads((root/'run.json').read_text());errors=[];frames=[];ply_count=0
    for frame in range(run['target_frames']):
        d=root/'frames'/f'{frame:05d}'
        try:
            m=json.loads((d/'metrics.json').read_text());frames.append(m)
            assert m['frame']==frame and m['resolution']==run['resolution']
            assert all(np.isfinite(m[k]) for k in ['median_abs_mm','p90_abs_mm','signed_median_mm','d455_dt_ms'])
            for name in ['fused','rgb4','d455','camera_0','camera_1','camera_2','camera_3']:
                path=d/f'{name}.ply'
                with path.open('rb') as handle:
                    header=[]
                    for _ in range(32):
                        line=handle.readline();header.append(line)
                        if line==b'end_header\n':break
                    else:raise ValueError(f'{path}: no PLY header')
                    assert b'format binary_little_endian 1.0\n' in header
                    n=int(next(line for line in header if line.startswith(b'element vertex ')).split()[-1])
                    assert n>0 and path.stat().st_size==handle.tell()+n*15
                ply_count+=1
            with zipfile.ZipFile(d/'depths.npz') as z:
                with z.open('depths_m.npy') as handle:
                    version=np.lib.format.read_magic(handle)
                    shape,_,dtype=np.lib.format._read_array_header(handle,version)
                    assert list(shape)==m['depth_shape'] and dtype==np.dtype('float32')
            for name in ['d455_rgb','difference']+[f'{mode}_{v}' for mode in ['rgb','depth'] for v in range(4)]:
                path=d/f'{name}.jpg';assert path.stat().st_size>100
        except Exception as e:errors.append({'frame':frame,'error':repr(e)})
    result={'expected_frames':run['target_frames'],'metric_frames':len(frames),'validated_ply_files':ply_count,'errors':errors}
    if frames:
        result['resolutions']=sorted(set(f['resolution'] for f in frames))
        result['sync_warnings_over_40ms']=sum(abs(f['d455_dt_ms'])>40 for f in frames)
        for name in ['median_abs_mm','p90_abs_mm','signed_median_mm','fused_points','peak_allocated_gib','peak_reserved_gib','total_s']:
            values=[f[name] for f in frames];result[name]={'min':min(values),'median':statistics.median(values),'max':max(values)}
    (root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
