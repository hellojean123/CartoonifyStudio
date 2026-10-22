"""Download only Commons files whose actual file metadata permits derivatives."""
from pathlib import Path
import datetime
import html
import json
import re
import time
import sys
import requests
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).parent/'samples'
ROOT.mkdir(exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (compatible; CartoonifyStudio/1.0)'

def clean(s):
    return html.unescape(re.sub('<[^>]+>', '', s))

def main():
    existing = ROOT/'metadata.json'
    if existing.exists():
        saved = json.loads(existing.read_text(encoding='utf-8'))
        if len(saved.get('images', [])) >= 20:
            for record in saved['images']:
                try:
                    if not record['license'].startswith(('CC BY', 'CC0', 'Public domain', 'OGL 3')):
                        raise ValueError('Unsupported license')
                    response = session.get(record['download_url'], timeout=60)
                    response.raise_for_status()
                    im = Image.open(BytesIO(response.content)).convert('RGB')
                    im.save(ROOT/record['filename'], quality=94)
                    record['download_date'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    print(record['filename'], 'refreshed', flush=True)
                    time.sleep(1)
                except Exception as exc:
                    saved['failures'].append({'title':record['title'], 'error':str(exc)})
            existing.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
            return
    records, failures = [], []
    queries = [('portrait', 'intitle:portrait intitle:India'), ('portrait', 'intitle:portrait intitle:2023'), ('portrait', 'intitle:woman intitle:portrait intitle:2019'), ('portrait', 'intitle:group intitle:people intitle:2018'), ('landscape', 'intitle:mountain intitle:landscape intitle:2020'), ('architecture', 'intitle:building intitle:2020'), ('animal', 'intitle:cat intitle:2020')]
    seen = set()
    for query_index, (category, query) in enumerate(queries):
        portrait_limit = [5, 9, 12, 15][min(query_index, 3)]
        target = 15 if category == 'portrait' else 2 if category in ('landscape', 'architecture') else 1
        if category == 'portrait' and sum(r['category']=='portrait' for r in records)>=15:
            continue
        try:
            response = session.get('https://commons.wikimedia.org/w/api.php', params={'action':'query', 'format':'json', 'generator':'search', 'gsrsearch':query, 'gsrnamespace':6, 'gsrlimit':40, 'prop':'imageinfo', 'iiprop':'url|extmetadata|size', 'iiurlwidth':1280}, timeout=50)
            response.raise_for_status()
            pages = sorted(response.json()['query']['pages'].values(), key=lambda p:p.get('index', 999))
        except Exception as e:
            failures.append({'query':query, 'error':str(e)})
            continue
        count = 0
        for page in pages:
            if (category=='portrait' and sum(r['category']=='portrait' for r in records)>=portrait_limit) or (category!='portrait' and count>=target):
                break
            title = page['title']
            if title in seen:
                continue
            seen.add(title)
            try:
                info = page['imageinfo'][0]
                meta = info['extmetadata']
                description = clean(meta.get('ImageDescription', {}).get('value', ''))
                if any(word in (title+' '+description).lower() for word in ['painting', 'oil on canvas', 'watercolor', 'drawing', 'lithograph', 'sculpture', 'museum', 'rembrandt', 'bridge']):
                    continue
                lic = clean(meta.get('LicenseShortName', {}).get('value', ''))
                author = clean(meta.get('Artist', {}).get('value', ''))
                if not (lic.startswith(('CC BY', 'CC0', 'Public domain', 'OGL 3'))) or not author or 'NC' in lic or 'ND' in lic:
                    failures.append({'title':title, 'error':'License or author not eligible: '+lic})
                    continue
                url = info.get('thumburl', info['url'])
                data = session.get(url, timeout=60)
                data.raise_for_status()
                im = Image.open(BytesIO(data.content)).convert('RGB')
                if min(im.size)<200:
                    continue
                filename = f'{len(records)+1:02d}_{category}.jpg'
                im.save(ROOT/filename, quality=94)
                records.append({'filename':filename, 'category':category, 'title':title, 'source_url':info['descriptionurl'], 'download_url':url, 'author':author, 'license':lic, 'license_url':meta.get('LicenseUrl', {}).get('value',''), 'download_date':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'modifications':'Resized by Wikimedia thumbnail service; converted to JPEG for local tests', 'description':clean(meta.get('ImageDescription',{}).get('value',''))})
                count += 1
                print(filename, title, lic, flush=True)
                time.sleep(.5)
            except Exception as e:
                failures.append({'title':title, 'error':str(e)})
        (ROOT/'metadata.json').write_text(json.dumps({'images':records,'failures':failures},ensure_ascii=False,indent=2), encoding='utf-8')
    (ROOT/'metadata.json').write_text(json.dumps({'images':records,'failures':failures},ensure_ascii=False,indent=2), encoding='utf-8')
    print('Downloaded:', len(records))

if __name__=='__main__':
    main()
