"""Acquire a genuinely detailed 4K Commons reference with verified attribution."""
from pathlib import Path
import datetime
import json
from io import BytesIO
import requests
from PIL import Image

def main():
    root=Path(__file__).parent;meta=json.loads((root/'samples/metadata.json').read_text(encoding='utf-8'))
    folder=root/'samples/high_resolution';folder.mkdir(exist_ok=True)
    if (folder/'reference_4k.jpg').exists(): return
    session=requests.Session();session.headers['User-Agent']='Mozilla/5.0 (CartoonifyStudio local evaluation)'
    attempts=[]
    for name in ('03_portrait.jpg','15_landscape.jpg','17_architecture.jpg','01_portrait.jpg'):
        source=next(r for r in meta['images'] if r['filename']==name)
        try:
            r=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':source['title'],'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':3840},timeout=45);r.raise_for_status()
            info=next(iter(r.json()['query']['pages'].values()))['imageinfo'][0]
            license=info['extmetadata']['LicenseShortName']['value']
            if not license.startswith(('CC BY','CC0','Public domain','OGL 3')): raise ValueError('License not eligible')
            if info['width']<3840: raise ValueError(f'Original width {info["width"]} < 3840')
            url=info.get('thumburl',info['url']);r=session.get(url,timeout=90);r.raise_for_status()
            image=Image.open(BytesIO(r.content)).convert('RGB')
            if image.width<3840 or image.width*image.height>16_000_000: raise ValueError('Reference size unsuitable')
            image.save(folder/'reference_4k.jpg',quality=96,subsampling=0)
            record={**source,'filename':'high_resolution/reference_4k.jpg','download_url':url,'download_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_dimensions':[info['width'],info['height']],'download_dimensions':list(image.size),'modifications':'Official 3840px thumbnail converted to JPEG; original source contains at least this detail'}
            (folder/'metadata.json').write_text(json.dumps({'image':record,'attempts':attempts},ensure_ascii=False,indent=2),encoding='utf-8')
            (folder/'ATTRIBUTION.md').write_text(f'# 真实高清评估参考图\n\n{record["title"]}\n\n{record["author"]}\n\n{record["license"]} {record["license_url"]}\n\n{record["source_url"]}\n\n原始尺寸 {record["original_dimensions"]}；下载尺寸 {record["download_dimensions"]}。\n',encoding='utf-8')
            print('Real 4K reference',image.size,flush=True);return
        except Exception as exc: attempts.append({'title':source['title'],'error':str(exc)})
    (folder/'metadata.json').write_text(json.dumps({'attempts':attempts},ensure_ascii=False,indent=2),encoding='utf-8')
    raise RuntimeError('No verified real 4K reference available; see recorded failures')

if __name__=='__main__': main()
