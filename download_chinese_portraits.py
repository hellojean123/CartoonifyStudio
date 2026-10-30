"""Append a fixed, independently inspected Commons portrait selection; never refresh old files."""
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse
import datetime,hashlib,html,json,re,time
import requests
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
if ROOT.name=='.tools':ROOT=Path('D:/CartoonifyStudio')
SELECTION=[
 ('30_hu_ge','胡歌','File:Hu Ge 2019.jpg'),
 ('31_zhou_xun','周迅','File:Zhou Xun at 2026 Busan International Film Festival.jpg'),
 ('32_zhang_ziyi','章子怡','File:Zhang Ziyi-63357 (cropped).jpg'),
 ('33_gong_li','巩俐','File:Gong Li Cannes 2016.jpg'),
 ('34_tang_wei','汤唯','File:Tang Wei.jpg'),
 ('35_shu_qi','舒淇','File:Shu Qi at the 2025 Toronto International Film Festival. 01 (cropped).jpg'),
 ('36_tony_leung','梁朝伟','File:Tony Leung Chiu Wai (Berlin Film Festival 2013) 2 (cropped).jpg'),
 ('37_donnie_yen','甄子丹','File:Donnie Yen at MCM May 2017.jpg'),
 ('38_andy_lau','刘德华','File:Andy Lau (cropped).jpg'),
 ('39_jackie_chan','成龙','File:Jackie Chan 2024.jpg'),
 ('40_liu_yifei','刘亦菲','File:Liu Yifei Portrait1.jpg'),
 ('41_zhou_xun_group','周迅、王圣迪','File:Actresses, Wang Shengdi and Zhou Xun at 2026 Busan International Film Festival.jpg')]


def clean(value):return html.unescape(re.sub('<[^>]+>','',value)).strip()


def validate(info,categories):
    m=info['extmetadata'];license=clean(m.get('LicenseShortName',{}).get('value',''));url=clean(m.get('LicenseUrl',{}).get('value',''));author=clean(m.get('Artist',{}).get('value',''))
    if not re.fullmatch(r'CC BY(?:-SA)? (?:2\.0|2\.5|3\.0|4\.0)|CC0',license):raise ValueError('Not an accepted derivative/reuse license: '+license)
    if not author or urlparse(url).hostname not in ('creativecommons.org','www.creativecommons.org'):raise ValueError('Missing author or verifiable license URL')
    bad=('license review needed','deletion requests','copyright violations','no permission','missing permission','disputed copyright')
    if any(any(word in category.lower() for word in bad) for category in categories):raise ValueError('Pending license/deletion warning')
    if info['mime'] not in ('image/jpeg','image/png','image/webp'):raise ValueError('Not a supported photograph file')
    return author,license,url


def main():
    samples=ROOT/'samples';evidence=samples/'source_records';evidence.mkdir(exist_ok=True)
    meta=json.loads((samples/'metadata.json').read_text(encoding='utf-8'));seen={r['title'] for r in meta['images']};failure_path=samples/'download_failures_v10.json';failures=json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else [];added=[]
    session=requests.Session();session.headers['User-Agent']='CartoonifyStudio/1.0 (https://github.com/hellojean123/CartoonifyStudio; licensed photo tests)'
    for stem,person,title in SELECTION:
        if title in seen:continue
        try:
            r=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':title,'prop':'imageinfo|categories|revisions','iiprop':'url|extmetadata|size|sha1|mime','iiurlwidth':1280,'cllimit':500,'rvprop':'ids|timestamp'},timeout=(20,60));r.raise_for_status();payload=r.json();page=next(iter(payload['query']['pages'].values()));info=page['imageinfo'][0]
            categories=[c['title'] for c in page.get('categories',[])];author,license,license_url=validate(info,categories)
            error=None;image=None;source_bytes=None;actual_url=None
            for url in dict.fromkeys([info.get('thumburl',info['url']),info['url']]):
                try:
                    if urlparse(url).hostname not in ('upload.wikimedia.org','thumb.wikimedia.org'):raise ValueError('Unexpected media host')
                    data=session.get(url,timeout=(25,100));data.raise_for_status()
                    if len(data.content)>20*1024**2:raise ValueError('Downloaded source exceeds 20 MiB')
                    source=Image.open(BytesIO(data.content))
                    if source.width*source.height>30_000_000:raise ValueError('Downloaded source exceeds 30 megapixels')
                    image=ImageOps.exif_transpose(source).convert('RGB');image.thumbnail((1600,1600),Image.Resampling.LANCZOS)
                    if min(image.size)<350:raise ValueError('Photo too small for a useful portrait test')
                    source_bytes=data.content;actual_url=url;break
                except Exception as exc:error=exc;failures.append({'title':title,'url':url,'error':str(exc),'phase':'media attempt'})
            if image is None or source_bytes is None:raise ValueError('All media downloads failed: '+str(error))
            filename=stem+'.jpg';dest=samples/filename
            if dest.exists():raise ValueError('Refusing to overwrite existing file: '+filename)
            image.save(dest,quality=95,subsampling=0)
            record={'filename':filename,'category':'portrait','collection':'chinese_celebrities','person':person,'title':title,'source_url':info['descriptionurl'],
                'source_revision':page.get('revisions',[{}])[0],'download_url':actual_url,'original_url':info['url'],'author':author,'license':license,'license_url':license_url,
                'download_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification_date':'2026-10-30',
                'modifications':'EXIF orientation normalized; resized to longest edge at most 1600 pixels and converted to JPEG. Derived style outputs retain this source license.',
                'description':clean(info['extmetadata'].get('ImageDescription',{}).get('value','')),'original_credit':clean(info['extmetadata'].get('Credit',{}).get('value','')),
                'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'download_sha256':hashlib.sha256(source_bytes).hexdigest(),'dimensions':list(image.size),
                'source_evidence':'source_records/'+stem+'.commons.json'}
            if stem=='40_liu_yifei':record['license_review']='Commons review by Shizhao on 2022-05-05, confirmed on file description page; public event portrait, downloaded from Commons only.'
            (evidence/(stem+'.commons.json')).write_text(json.dumps({'retrieved_api':r.url,'retrieved_at':record['download_date'],'page':page},ensure_ascii=False,indent=2),encoding='utf-8')
            meta['images'].append(record);seen.add(title);added.append(record)
            (samples/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'file':filename,'person':person,'license':license,'size':image.size}),flush=True)
        except Exception as exc:failures.append({'title':title,'error':str(exc),'phase':'selection'});print(json.dumps({'title':title,'error':str(exc)}),flush=True)
        time.sleep(.5)
    (samples/'download_failures_v10.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    # Append rather than rewriting prior attributions or changing their licenses.
    credits='\n\n# 新增中国明星公开活动人像（2026-10-30）\n\n人物仅供图像算法演示，不表示背书。以下原图及对应派生图片保留所列授权；CC BY-SA 派生图片按该相同许可版本共享。\n'
    for record in added:
        credits+=f"\n## {record['filename']} · {record['person']}\n\n作者：{record['author']}\n\n来源：{record['source_url']}\n\n许可：{record['license']} {record['license_url']}\n\n修改：{record['modifications']}\n"
    if added:
        with (samples/'ATTRIBUTION.md').open('a',encoding='utf-8') as f:f.write(credits)
    print(json.dumps({'added':len(added),'total':len(meta['images']),'failed_attempts':len(failures)}),flush=True)


if __name__=='__main__':main()
