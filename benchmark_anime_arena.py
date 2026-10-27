from pathlib import Path
import sys,time,subprocess,json,threading
import numpy as np
import cv2,psutil,onnxruntime as ort
root=Path(__file__).resolve().parent;model=root/'models/face_paint_v2.onnx'

def worker(arena):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.enable_cpu_mem_arena=bool(arena)
    proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.005):peak.append(proc.memory_info().rss)
    thread=threading.Thread(target=monitor);thread.start()
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    x=np.random.default_rng(7).uniform(-1,1,(1,3,480,640)).astype(np.float32)
    times=[]
    try:
        for _ in range(4):
            t=time.perf_counter();output=session.run(None,{'image':x})[0];times.append(time.perf_counter()-t)
    finally:stop.set();thread.join()
    return {'arena':bool(arena),'cold_seconds':times[0],'runs':times[1:],'median_seconds':float(np.median(times[1:])),'peak_mib':max(peak)/1024**2,'result_sum':float(output.sum())}

if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(int(sys.argv[1]))))
    else:
        rows=[]
        for arena in (0,1):
            r=subprocess.run([sys.executable,__file__,str(arena)],capture_output=True,text=True,check=True);row=json.loads(r.stdout);rows.append(row);print(row,flush=True)
        (root/'outputs/anime_arena_benchmark.json').write_text(json.dumps(rows,indent=2))
