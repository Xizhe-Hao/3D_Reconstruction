"""Serve only reconstruction assets, and build an up-to-date analysis report."""
import argparse,json,statistics
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/report.json':
            root=Path(self.directory)
            config=json.loads((root/'run.json').read_text())
            frames=[]
            for p in sorted((root/'frames').glob('*/metrics.json')):
                try: frames.append(json.loads(p.read_text()))
                except (ValueError,OSError):pass
            errors=[f['median_abs_mm'] for f in frames if 'median_abs_mm' in f]
            median=statistics.median(errors) if errors else None
            config['frames']=frames
            config['summary']=(f"已完成 {len(frames)}/{config['target_frames']} 个时间点；DA3 推理长边 {config['resolution']}px，四视角同时输入。"+(f"各帧深度绝对差中位数的中位数为 {median:.2f}mm。" if median is not None else '')+'此差异用于检查一致性，不代表绝对重建精度。')
            body=json.dumps(config,ensure_ascii=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        else:super().do_GET()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8080);args=p.parse_args()
    server=ThreadingHTTPServer(('0.0.0.0',args.port),partial(Handler,directory=str(args.root.resolve())))
    print(f'Serving reconstruction on 0.0.0.0:{args.port}',flush=True);server.serve_forever()
