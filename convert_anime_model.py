"""Optional development-only CPU conversion of pinned author FacePaint weights.

Runtime users need only the bundled ONNX file. See README for CPU-only torch
and ONNX installation. Downloads are verified before importing upstream code;
torch weights are loaded with weights_only=True, never arbitrary checkpoint code.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,tempfile
import requests
import numpy as np

COMMIT='25d7b017267208dfaf34026aa3425e518372aa2f'
FILES={
    'model.py':('model.py','2047cdb37764e7fc3b8e2e4b1b93bced06ee86e705e34995ca5747cdf0d0bceb'),
    'LICENSE.txt':('LICENSE','0722b5505349ac7ccc5a446da532e81f0962a07a671c6cc1f2f38a4a09c80c45'),
    'face_paint_512_v2.pt':('weights/face_paint_512_v2.pt','06b88a204eb230889444ad868ee5608f4fce5d4ff7b7738acaa4209c2b8fdca7')}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-cache',type=Path,help='Optional local upstream files; every file is still SHA256 checked')
    parser.add_argument('--output-dir',type=Path,help='Optional conversion destination, defaults to project models/')
    args=parser.parse_args()
    import torch,onnxruntime as ort
    if torch.version.cuda is not None:raise RuntimeError('Install the CPU-only torch build described in README')
    torch.set_num_threads(2);root=args.output_dir or Path(__file__).resolve().parent/'models';root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cartoonify-convert-') as folder:
        temp=Path(folder);sources={}
        for local,(remote,digest) in FILES.items():
            url=f'https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/{COMMIT}/{remote}'
            if args.source_cache:data=(args.source_cache/local).read_bytes()
            else:
                r=requests.get(url,timeout=(30,120));r.raise_for_status();data=r.content
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Upstream checksum mismatch: '+local)
            (temp/local).write_bytes(data);sources[local]={'url':url,'sha256':digest,'bytes':len(data)}
        spec=importlib.util.spec_from_file_location('pinned_facepaint',temp/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.Generator().cpu().eval()
        model.load_state_dict(torch.load(temp/'face_paint_512_v2.pt',map_location='cpu',weights_only=True))
        dest=temp/'face_paint_v2.onnx'
        with torch.inference_mode():
            torch.onnx.export(model,torch.zeros(1,3,256,256),str(dest),opset_version=17,input_names=['image'],output_names=['anime'],
                dynamic_axes={'image':{2:'height',3:'width'},'anime':{2:'height',3:'width'}},dynamo=False)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(dest),sess_options=options,providers=['CPUExecutionProvider'])
        errors=[]
        for height,width in ((192,256),(64,100),(256,192)):
            x=np.random.default_rng(12).uniform(-1,1,(1,3,height,width)).astype(np.float32)
            with torch.inference_mode():expected=model(torch.from_numpy(x)).numpy()
            actual=session.run(None,{'image':x})[0];errors.append(float(np.max(np.abs(actual-expected))))
        if max(errors)>=.001:raise RuntimeError('ONNX conversion exceeds numerical tolerance: '+str(errors))
        previous=root/'anime_metadata.json'
        metadata=json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
        metadata.update({'name':'FacePaint v2','author':'Bryan Lee','license':'MIT (pinned repository LICENSE)','commit':COMMIT,'files':sources,
            'torch':torch.__version__,'cuda':None,'onnxruntime':ort.__version__,'providers':session.get_providers(),'max_abs_error':max(errors),
            'onnx_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'model_limit':640,
            'conversion':'PyTorch CPU weights_only=True -> ONNX opset 17; conversion tool is development-only'})
        # Publish only after all shape and numerical checks pass.
        (root/'face_paint_v2.onnx').write_bytes(dest.read_bytes())
        (root/'LICENSE-FacePaint.txt').write_bytes((temp/'LICENSE.txt').read_bytes())
        previous.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'providers':session.get_providers(),'shape_errors':errors,'sha256':metadata['onnx_sha256']}))

if __name__=='__main__':main()
