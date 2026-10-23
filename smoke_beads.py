"""Actual HTTP/API tests of bead presets, automatic and user-supplied color cards."""
from pathlib import Path
from io import BytesIO
import csv,json,zipfile,time
import requests
from pypdf import PdfReader
from PIL import Image
from gradio_client import Client,handle_file
from app import build_app

ROOT=Path(__file__).parent

def main():
    demo=build_app();results=[]
    try:
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7862,prevent_thread_lock=True,inbrowser=False,quiet=True,max_file_size='40mb')
        assert requests.get(url,timeout=20).status_code==200;client=Client(url,verbose=False)
        for mode,palette in [('自动精简颜色',None),('使用实体色卡 CSV',handle_file(str(ROOT/'palettes/example.csv')))]:
            started=time.perf_counter();response=client.predict(handle_file(str(ROOT/'samples/05_portrait.jpg')),48,8,2,3,2,5.,True,mode,palette,api_name='/beads')
            preview,legend,pdf,archive,status=response
            with Image.open(preview) as image:assert image.width>0
            assert 'B01' in legend and '孤立单粒' in status
            reader=PdfReader(pdf);assert len(reader.pages)>1
            with zipfile.ZipFile(archive) as bundle:
                data=json.loads(bundle.read('pattern.json'));counts=list(csv.DictReader(bundle.read('colors.csv').decode('utf-8-sig').splitlines()))
                assert sum(int(r['count']) for r in counts)==data['metrics']['total_beads']
                assert data['metrics']['custom_palette']==(palette is not None)
                assert data['metrics']['used_colors']<=8
                if palette is not None:assert all(r['code'].startswith('EXAMPLE-') for r in counts)
            results.append({'mode':mode,'status':status,'pdf_pages':len(reader.pages),'http_seconds':time.perf_counter()-started})
            print('Bead API passed:',mode,flush=True)
        pending=client.submit(handle_file(str(ROOT/'samples/03_portrait.jpg')),160,32,2,3,2,5.,True,'自动精简颜色',None,api_name='/beads')
        from cartoonify.service import _JOBS,_JOBS_LOCK
        deadline=time.monotonic()+60;active=False
        while time.monotonic()<deadline:
            with _JOBS_LOCK:active=bool(_JOBS)
            if active:break
            if pending.done():raise AssertionError('Bead job finished before cancellation observation')
            time.sleep(.05)
        assert active
        cancel_response=client.predict(api_name='/beads_cancel')
        response=pending.result(timeout=60)
        assert '取消' in response[-1] and response[2] is None and response[3] is None
        print('Bead in-flight cancellation passed',flush=True)
        (ROOT/'outputs/beads/smoke_test.json').write_text(json.dumps({'http_status':200,'cases':results,'cancel_response':cancel_response,'cancellation_status':response[-1]},ensure_ascii=False,indent=2),encoding='utf-8')
    finally:demo.close()

if __name__=='__main__':main()
