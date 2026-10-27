"""Actual Gradio HTTP pencil preview/export/batch verification."""
from pathlib import Path
from dataclasses import astuple,replace
import json,time,zipfile
import requests
from PIL import Image
from gradio_client import Client,handle_file
from app import build_app
from cartoonify.core import Params,PRESETS,STYLES
ROOT=Path(__file__).resolve().parent


def main():
    demo=build_app();report={}
    try:
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7861,prevent_thread_lock=True,inbrowser=False,quiet=True,max_file_size='40mb')
        assert requests.get(url,timeout=20).status_code==200
        client=Client(url,verbose=False);path=ROOT/'samples/05_portrait.jpg'
        p=Params(*PRESETS[3]);started=time.perf_counter()
        pending=client.submit(handle_file(str(path)),STYLES[3],*astuple(p),api_name='/preview')
        stages=list(pending);response=pending.result()
        assert len(stages)>=2 and '草稿' in stages[0][3] and '最终预览' in response[3]
        report['preview']={'stages':len(stages),'status':response[3],'end_to_end_seconds':time.perf_counter()-started}
        exports=[]
        for fmt,colored in [('PNG',False),('JPG',True)]:
            started=time.perf_counter()
            response=client.predict(handle_file(str(path)),STYLES[3],*astuple(replace(p,colored=colored)),fmt,'精细',api_name='/export')
            with Image.open(response[2]) as im:
                assert im.size==Image.open(path).size and im.info.get('icc_profile')
                assert im.format==('JPEG' if fmt=='JPG' else 'PNG')
                extrema=im.getextrema();assert any(high>low for low,high in extrema)
                exports.append({'format':fmt,'colored':colored,'size':list(im.size),'end_to_end_seconds':time.perf_counter()-started})
        response=client.predict([handle_file(str(path)),handle_file(str(ROOT/'samples/23_portrait.jpg'))],STYLES[3],*astuple(p),'PNG','标准',api_name='/batch')
        with zipfile.ZipFile(response[0]) as archive:
            images=[name for name in archive.namelist() if name.endswith('.png')]
            assert len(images)==2
        report.update({'http_status':200,'exports':exports,'batch_png_count':2,'cancel_response':client.predict(api_name='/cancel')})
        (ROOT/'outputs/pencil_v5/http_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:demo.close()


if __name__=='__main__':main()
