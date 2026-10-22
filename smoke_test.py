"""Actual HTTP + Gradio API checks for v2 preview, export, batch and cancellation."""
from dataclasses import astuple
import json
from pathlib import Path
import time
import zipfile
import requests
from PIL import Image
from gradio_client import Client,handle_file
from app import build_app
from cartoonify.core import STYLES,PRESETS,Params

ROOT=Path(__file__).parent

def main():
    demo=build_app();results=[]
    try:
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7861,prevent_thread_lock=True,inbrowser=False,quiet=True,max_file_size='40mb')
        assert requests.get(url,timeout=20).status_code==200
        client=Client(url,verbose=False)
        path=ROOT/'samples/03_portrait.jpg'
        for style_index,style in enumerate(STYLES):
            values=list(astuple(Params(*PRESETS[style_index])))
            started=time.perf_counter()
            pending=client.submit(handle_file(str(path)),style,*values,api_name='/preview')
            steps=list(pending);response=pending.result()
            assert len(steps)>=2 and '草稿' in steps[0][3] and '最终预览' in response[3]
            assert response[2] is None
            assert Image.open(response[1]).width<=1000
            results.append({'style':style,'preview_stages':len(steps),'status':response[3],'http_end_to_end_seconds':time.perf_counter()-started})
            print('Preview passed',style_index,flush=True)
        values=list(astuple(Params()))
        response=client.predict(handle_file(str(path)),STYLES[0],*values,'JPG','精细',api_name='/export')
        im=Image.open(response[2]);assert im.format=='JPEG' and im.size==Image.open(path).size and im.info.get('icc_profile')
        jpg_size=im.size
        response=client.predict(handle_file(str(path)),STYLES[2],*values,'PNG','精细',api_name='/export')
        assert Image.open(response[2]).format=='PNG'
        valid=[handle_file(str(path)),handle_file(str(ROOT/'samples/05_portrait.jpg'))]
        response=client.predict(valid,STYLES[1],*values,'PNG','标准',api_name='/batch')
        with zipfile.ZipFile(response[0]) as z:
            assert len([n for n in z.namelist() if n.endswith('.png')])==2
        cancelled=client.predict(api_name='/cancel');assert isinstance(cancelled,str)
        # Actual in-flight cancellation shares the client's session state.
        pending=client.submit(valid*5,STYLES[1],*values,'PNG','精细',api_name='/batch')
        # Uploading inputs happens before the job enters the CPU queue.
        from cartoonify.service import _JOBS,_JOBS_LOCK
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            with _JOBS_LOCK: active=bool(_JOBS)
            if active: break
            if pending.done(): raise AssertionError('Batch completed before entering cancellation observation')
            time.sleep(.05)
        assert active,'Batch did not start in time'
        cancel_response=client.predict(api_name='/cancel')
        print('Cancellation response:',cancel_response,flush=True)
        cancellation_result=pending.result(timeout=90)
        assert '取消 True' in cancellation_result[1],cancellation_result
        report={'http_status':200,'exports':results,'full_resolution_jpg':list(jpg_size),'full_png':True,'batch_png_count':2,'cancel_response':cancel_response,'cancelled_batch_status':cancellation_result[1]}
        (ROOT/'outputs/smoke_test.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('HTTP, 4 staged previews, original JPG/PNG, batch ZIP and in-flight cancellation passed',flush=True)
    finally: demo.close()

if __name__=='__main__': main()
