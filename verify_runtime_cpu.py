"""Check the saved anime result can be reproduced using runtime-only CPU deps."""
from pathlib import Path
import importlib.util,json
import numpy as np
from PIL import Image
from cartoonify.core import process,load_image,STYLES
from cartoonify.anime import session
ROOT=Path(__file__).resolve().parent

def main():
    image=load_image(ROOT/'samples/23_portrait.jpg')
    result=process(image,STYLES[0],cache=None)
    with Image.open(ROOT/'outputs/1_经典卡通/23_portrait.png') as stored:expected=np.array(stored.convert('RGB'))
    difference=int(np.max(np.abs(result.astype(np.int16)-expected.astype(np.int16))))
    assert difference==0 and session().get_providers()==['CPUExecutionProvider']
    record={'providers':session().get_providers(),'torch_installed':importlib.util.find_spec('torch') is not None,
        'onnx_development_package_installed':importlib.util.find_spec('onnx') is not None,'pixel_max_difference_from_saved_result':difference,
        'source':'samples/23_portrait.jpg','output':'outputs/1_经典卡通/23_portrait.png'}
    (ROOT/'outputs/runtime_cpu_v4.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record))

if __name__=='__main__':main()
