"""Bounded bead export CPU/RSS measurements in isolated workers."""
from pathlib import Path
import json,sys,subprocess,threading,time
import psutil,numpy as np
from cartoonify.core import load_image
from cartoonify.beads import BeadParams,make_pattern,export_files,package_files
ROOT=Path(__file__).parent

def worker(grid,colors):
    image=load_image(ROOT/'samples/high_resolution/reference_4k.jpg')
    p=BeadParams(grid=grid,colors=colors);make_pattern(image,p)
    stop=threading.Event();proc=psutil.Process();peaks=[proc.memory_info().rss]
    def monitor():
        while not stop.wait(.005):peaks.append(proc.memory_info().rss)
    thread=threading.Thread(target=monitor);thread.start();alg=[];export=[]
    try:
        for _ in range(3):
            t=time.perf_counter();pattern=make_pattern(image,p);alg.append(time.perf_counter()-t)
            t=time.perf_counter();files=export_files(pattern);bundle=package_files(files);export.append(time.perf_counter()-t)
    finally:stop.set();thread.join()
    return {'grid':grid,'colors':colors,'source_dimensions':[image.shape[1],image.shape[0]],'algorithm_runs':alg,'algorithm_median':float(np.median(alg)),
            'export_runs':export,'export_median':float(np.median(export)),'peak_rss_mib':max(peaks)/1024**2,'zip_bytes':len(bundle),'total_beads':pattern.metrics['total_beads']}

def main():
    rows=[]
    for grid,colors in ((72,10),(96,18),(160,32)):
        r=subprocess.run([sys.executable,__file__,'--worker',str(grid),str(colors)],capture_output=True,text=True,encoding='utf-8',check=True)
        row=json.loads(r.stdout.strip().splitlines()[-1]);rows.append(row);print(grid,row['algorithm_median'],row['export_median'],row['peak_rss_mib'],flush=True)
    (ROOT/'outputs/beads/benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(int(sys.argv[2]),int(sys.argv[3]))))
    else:main()
