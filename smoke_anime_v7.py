"""Exercise real Gradio HTTP routes for both offline portrait models."""
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
    app=build_app();report={}
    try:
        _,url,_=app.queue(default_concurrency_limit=1,max_size=8).launch(server_name='127.0.0.1',server_port=7861,prevent_thread_lock=True,inbrowser=False,quiet=True,max_file_size='40mb')
        assert requests.get(url,timeout=20).status_code==200
        client=Client(url,verbose=False);path=ROOT/'samples/03_portrait.jpg';p=Params(*PRESETS[0]);modes=[]
        for mode,fmt in [('动漫增强','PNG'),('柔和插画','JPG')]:
            selected=replace(p,anime_mode=mode);t=time.perf_counter()
            pending=client.submit(handle_file(str(path)),STYLES[0],*astuple(selected),api_name='/preview')
            stages=list(pending);response=pending.result()
            assert len(stages)>=2 and '草稿' in stages[0][3] and '最终预览' in response[3]
            preview_seconds=time.perf_counter()-t
            response=client.predict(handle_file(str(path)),STYLES[0],*astuple(selected),fmt,'精细',api_name='/export')
            with Image.open(response[2]) as im:
                assert im.size==Image.open(path).size and im.info.get('icc_profile')
                assert im.format==('JPEG' if fmt=='JPG' else 'PNG')
                assert any(hi>lo for lo,hi in im.getextrema())
                modes.append({'mode':mode,'preview_stages':len(stages),'preview_end_to_end_seconds':preview_seconds,'format':fmt,'size':list(im.size),'export_status':response[3]})
        response=client.predict([handle_file(str(ROOT/'samples/05_portrait.jpg')),handle_file(str(ROOT/'samples/23_portrait.jpg'))],STYLES[0],*astuple(p),'PNG','标准',api_name='/batch')
        with zipfile.ZipFile(response[0]) as archive:
            images=[name for name in archive.namelist() if name.endswith('.png')];assert len(images)==2
        report={'http_status':200,'modes':modes,'batch_png_count':len(images),'cancel_response':client.predict(api_name='/cancel')}
        (ROOT/'outputs/anime_v7/http_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:app.close()


if __name__=='__main__':main()
