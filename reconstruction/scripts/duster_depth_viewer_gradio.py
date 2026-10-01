#!/usr/bin/env python3
"""Interactive four-view Gradio depth player for DUSt3R and DA3."""
import argparse, json, os, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.test_session_adapter import load_session_metadata
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"outputs/data_test/frames_0_None_target_96_duster/duster"
def resolve_folder(folder):
    p=Path(folder).expanduser(); p=(p if p.is_absolute() else ROOT/p).resolve()
    for backend in ("da3", "duster"):
        if (p/backend).is_dir():
            return p/backend
    return p

def files(folder):
    p=resolve_folder(folder)
    if (p/"input.npz").is_file():
        with np.load(p/"input.npz") as data:
            out=[p/f"frame_{int(frame):05d}.npz" for frame in data["frame_indices"]]
        missing=[path.name for path in out if not path.is_file()]
        if missing: raise FileNotFoundError(f"Missing DA3 frame files: {missing}")
    else:
        out=sorted(p.glob("3d_model__*__scene.npz"))
    if not out: raise FileNotFoundError(f"No DA3 or DUSt3R depth files in {p}")
    return out

_SOURCE_CACHE = {}
def rgb_frames(folder,index):
    result_dir=resolve_folder(folder)
    if (result_dir/"input.npz").is_file():
        key=str(result_dir/"input.npz")
        if key not in _SOURCE_CACHE:
            with np.load(key) as data:
                _SOURCE_CACHE[key]=(data["frame_indices"].copy(), data["rgbs"].copy())
        frames,rgbs=_SOURCE_CACHE[key]
        return int(frames[index]),[image.transpose(1,2,0) for image in rgbs[:,index]]
    key=str(result_dir)
    if key not in _SOURCE_CACHE:
        run_dir=Path(key).parent; meta=json.loads((run_dir/"run.json").read_text())
        cameras,_,_=load_session_metadata(Path(meta["session_dir"])); frames=np.asarray(meta["frames"],dtype=int)
        maps=[]; caps=[]
        for camera in cameras:
            maps.append(cv2.initUndistortRectifyMap(camera.intrinsic,camera.distortion,None,camera.intrinsic,camera.image_size,cv2.CV_32FC1))
            cap=cv2.VideoCapture(str(camera.video_path))
            if not cap.isOpened(): raise RuntimeError(f"Cannot open {camera.video_path}")
            caps.append(cap)
        _SOURCE_CACHE[key]=(cameras,frames,maps,caps)
    cameras,frames,maps,caps=_SOURCE_CACHE[key]; frame=int(frames[index]); result=[]
    for camera,(mx,my),cap in zip(cameras,maps,caps):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,bgr=cap.read()
        if not ok: raise RuntimeError(f"Cannot decode frame {frame} from {camera.video_path}")
        bgr=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        result.append(cv2.cvtColor(cv2.resize(bgr,(512,384),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
    return frame,result
def draw_all(folder,timestamp,auto,lo,hi):
    ps=files(folder); path=ps[int(np.clip(round(timestamp),0,len(ps)-1))]
    result_dir=resolve_folder(folder)
    with np.load(path) as d:
        if "depths_m" in d:
            depths=np.float32(d["depths_m"]); masks=np.ones(depths.shape,dtype=bool)
        else:
            depths=np.float32(d["depths"]); masks=np.bool_(d["cleaned_mask"])
    final_depths=result_dir.parent/"depths_da3_m.npz"
    if (result_dir/"input.npz").is_file() and final_depths.is_file():
        with np.load(final_depths) as d:
            matches=np.flatnonzero(d["frame_indices"]==int(path.stem.split("_")[-1]))
            if len(matches)!=1: raise ValueError("DA3 frame is absent from tracking depth cache")
            depths=np.float32(d["depths_m"][:,int(matches[0])])
        masks=depths>0
    if len(depths)!=4: raise ValueError(f"Expected four views, got {len(depths)}")
    valid=np.isfinite(depths)&(depths>0)&masks; sample=depths[valid]
    if auto: lo,hi=np.percentile(sample,[2,98]) if sample.size else (0.,1.)
    elif float(hi)<=float(lo): raise ValueError("Depth max must be greater than depth min")
    def color(a,ok):
        x=np.uint8(np.clip((a-lo)/max(hi-lo,1e-9),0,1)*255); x=cv2.cvtColor(cv2.applyColorMap(x,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB); x[~ok]=0; return x
    images=[color(depths[v],valid[v]) for v in range(4)]
    frame,rgbs=rgb_frames(folder,int(np.clip(round(timestamp),0,len(ps)-1)))
    stats=[f"V{v}: valid={valid[v].mean():.1%}, median={np.median(depths[v][valid[v]]):.6f}m" if valid[v].any() else f"V{v}: no valid depth" for v in range(4)]
    paired=[item for pair in zip(rgbs,images) for item in pair]
    return *paired,f"{path.name}; source frame={frame}; shared range={lo:.6f}-{hi:.6f}m; "+" | ".join(stats)
def main():
    os.environ["NO_PROXY"]="127.0.0.1,localhost"; os.environ["no_proxy"]="127.0.0.1,localhost"
    import gradio as gr
    p=argparse.ArgumentParser(); p.add_argument("--result-dir", "--duster-dir",dest="result_dir",default=str(DEFAULT)); p.add_argument("--server-name",default="127.0.0.1"); p.add_argument("--server-port",type=int,default=7862); a=p.parse_args(); count=len(files(a.result_dir))
    with gr.Blocks(title="Four-view depth player") as app:
        gr.Markdown("# Four-view depth player\nDUSt3R / DA3: synchronized RGB and metric depth")
        folder=gr.Textbox(a.result_dir,label="Result directory (run, duster, or da3)")
        with gr.Row(): play=gr.Button("Play",variant="primary"); t=gr.Slider(0,count-1,step=1,label="Timestamp"); fps=gr.Slider(.5,10,value=2,step=.5,label="Playback FPS")
        with gr.Row(): auto=gr.Checkbox(True,label="Shared auto range (P2-P98)"); lo=gr.Number(.05,label="Fixed min (m)"); hi=gr.Number(.5,label="Fixed max (m)")
        panels=[]
        for view in range(4):
            with gr.Row():
                panels.extend([gr.Image(label=f"Camera {view} RGB"),gr.Image(label=f"Camera {view} depth")])
        info=gr.Textbox(label="Frame statistics"); playing=gr.State(False); timer=gr.Timer(value=.5,active=False)
        ins=[folder,t,auto,lo,hi]; outs=[*panels,info]
        for c in (t,auto,lo,hi): c.change(draw_all,ins,outs,show_progress="hidden")
        def change_folder(value):
            total=len(files(value))
            return gr.update(maximum=total-1,value=0),*draw_all(value,0,auto.value,lo.value,hi.value)
        folder.submit(change_folder,folder,[t,*outs],show_progress="hidden")
        app.load(draw_all,ins,outs)
        play.click(lambda state:(not state,gr.update(active=not state),"Pause" if not state else "Play"),playing,[playing,timer,play],show_progress="hidden")
        timer.tick(lambda directory,value:(int(value)+1)%len(files(directory)),[folder,t],t)
        fps.change(lambda value:gr.update(value=1.0/value),fps,timer)
    app.launch(server_name=a.server_name,server_port=a.server_port,show_error=True)
if __name__=="__main__": main()
