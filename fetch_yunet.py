from pathlib import Path
from datetime import datetime,timezone
import requests,json,hashlib
root=Path(__file__).resolve().parent/'models';root.mkdir(exist_ok=True)
s=requests.Session()
commit=s.get('https://api.github.com/repos/opencv/opencv_zoo/commits/main',timeout=45);commit.raise_for_status();sha=commit.json()['sha']
base=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{sha}/models/face_detection_yunet/'
name='face_detection_yunet_2023mar.onnx'
r=s.get(base+name,timeout=90);r.raise_for_status()
assert len(r.content)>100000 and not r.content.startswith(b'version https')
license_url=f'https://raw.githubusercontent.com/opencv/opencv_zoo/{sha}/models/face_detection_yunet/LICENSE'
lic=s.get(license_url,timeout=45);lic.raise_for_status();assert 'MIT' in lic.text
(root/name).write_bytes(r.content);(root/'LICENSE-YuNet.txt').write_text(lic.text,encoding='utf-8')
(root/'metadata.json').write_text(json.dumps({'name':name,'source':base+name,'upstream_commit':sha,'license':'MIT','license_url':license_url,'sha256':hashlib.sha256(r.content).hexdigest(),'downloaded_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Pretrained CPU-only face detection; no training, no face recognition'},indent=2),encoding='utf-8')
print('Verified official YuNet:',len(r.content),'bytes',sha)
