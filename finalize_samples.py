"""Curate discovered Commons records and fetch explicitly reviewed extra portraits."""
import datetime
import json
from io import BytesIO
from pathlib import Path
import sys
import time
from PIL import Image
from download_samples import ROOT, session, clean

if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')

def main():
    metadata=json.loads((ROOT/'metadata.json').read_text(encoding='utf-8'))
    rejected=ROOT/'rejected_candidates_4'; rejected.mkdir(exist_ok=True)
    kept=[]
    for r in metadata['images']:
        if any(s in r['title'] for s in ['Yaquina','Allentown','Portrait head']):
            p=ROOT/r['filename']
            if p.exists(): p.replace(rejected/p.name)
            metadata['failures'].append({'title':r['title'],'error':'Visual content rejected: not a real human portrait photograph'})
        else: kept.append(r)
    titles=['File:Portrait Ludwig Schleritzko (2023).jpg','File:Arben Vitia - portrait 2023.jpg','File:Victoria Atkins Official Cabinet Portrait, November 2023.jpg','File:Sarah Healey official portrait 2023.jpg','File:Official portrait of first minister Humza Yousaf, 2023 (cropped).jpg','File:Richard Holden Official Cabinet Portrait, November 2023.jpg','File:James Cleverly Official Cabinet Portrait, November 2023.jpg','File:Greg Hands Official Cabinet Portrait, February 2023.jpg','File:Grant Shapps Official Cabinet Portrait August 2023.jpg']
    for i,title in enumerate(titles):
        if any(r['title']==title for r in kept): continue
        try:
            response=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':title,'prop':'imageinfo','iiprop':'url|extmetadata','iiurlwidth':640},timeout=45);response.raise_for_status()
            info=next(iter(response.json()['query']['pages'].values()))['imageinfo'][0];m=info['extmetadata']
            lic=clean(m.get('LicenseShortName',{}).get('value',''));author=clean(m.get('Artist',{}).get('value',''))
            if not lic.startswith(('CC BY','CC0','Public domain','OGL 3')) or not author: raise ValueError('Unverified license or author')
            url=info.get('thumburl',info['url']);response=session.get(url,timeout=60);response.raise_for_status()
            im=Image.open(BytesIO(response.content)).convert('RGB');filename=f'{21+i:02d}_portrait.jpg';im.save(ROOT/filename,quality=94)
            kept.append({'filename':filename,'category':'portrait','title':title,'source_url':info['descriptionurl'],'download_url':url,'author':author,'license':lic,'license_url':m.get('LicenseUrl',{}).get('value',''),'download_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modifications':'Wikimedia thumbnail resized; JPEG conversion','description':clean(m.get('ImageDescription',{}).get('value',''))})
            print(filename,title,flush=True);time.sleep(1)
        except Exception as e: metadata['failures'].append({'title':title,'error':str(e)});print(str(e),flush=True)
    metadata['images']=kept
    (ROOT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    attribution=['# 测试图片与生成结果署名\n','本目录及 outputs 的对应改编结果保留以下署名。CC BY-SA 改编图片按相同许可版本再分发；OGL 3 遵守英国开放政府许可；公有领域图片保留来源说明。人物仅作技术测试，不表示背书。\n']
    for r in kept: attribution.append(f"## {r['filename']}\n\n{r['title']}\n\n作者：{r['author']}\n\n来源：{r['source_url']}\n\n许可：{r['license']} {r['license_url']}\n\n修改：缩略图转 JPEG；outputs 中另有卡通、插画、漫画和素描改编。\n")
    (ROOT/'ATTRIBUTION.md').write_text('\n'.join(attribution),encoding='utf-8')
    print('Accepted',len(kept),'portraits',sum(r['category']=='portrait' for r in kept),flush=True)

if __name__=='__main__': main()
