"""Run one independent four-view reconstruction worker per selected GPU."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--resolution',type=int,default=1512);p.add_argument('--frames-file',type=Path,help='JSON with a "frames" list, e.g. active_segments.json from find_active_segments.py');args=p.parse_args()
    gpus=args.gpus.split(',');export=json.loads((args.session/'multimodal_video_export.json').read_text())
    selected=json.loads(args.frames_file.read_text())['frames'] if args.frames_file else list(range(export['frame_count']));n=len(selected);processes=[];logs=[]
    args.output.mkdir(parents=True,exist_ok=True)
    # Interleave frames so every completed round covers the full time interval.
    segments=[selected[i*n//len(gpus):(i+1)*n//len(gpus)] for i in range(len(gpus))]
    for gpu,frames in zip(gpus,segments):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        log=(args.output/f'worker_{gpu}.log').open('w');logs.append(log)
        cmd=[sys.executable,str(Path(__file__).with_name('reconstruct_rgbd_web.py')),'--session',str(args.session),'--output',str(args.output/'frames'),'--resolution',str(args.resolution),'--frames',','.join(map(str,frames))]
        processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    print('Started workers:',[(g,p.pid) for g,p in zip(gpus,processes)],flush=True)
    while any(p.poll() is None for p in processes):
        count=sum((args.output/'frames'/f'{f:05d}'/'metrics.json').exists() for f in selected)
        print(f'Completed {count}/{n}; worker status {[p.poll() for p in processes]}',flush=True);time.sleep(30)
    codes=[p.wait() for p in processes]
    for log in logs:log.close()
    manifest=args.output/'run.json';meta=json.loads(manifest.read_text());count=sum((args.output/'frames'/f'{f:05d}'/'metrics.json').exists() for f in selected)
    meta['status']='complete' if count==n and all(c==0 for c in codes) else 'incomplete';meta['worker_exit_codes']=codes;manifest.write_text(json.dumps(meta,indent=2))
    print('DONE',meta['status'],count,n,codes,flush=True)
    sys.exit(0 if meta['status']=='complete' else 1)
if __name__=='__main__':main()
