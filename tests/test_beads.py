from dataclasses import replace
from io import BytesIO, StringIO
import csv
import math
import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader
from cartoonify.beads import (BeadParams,make_pattern,read_palette,cleanup_regions,region_metrics,
    color_rows,export_files,package_files,normalized)

def noisy_scene():
    im=np.full((120,160,3),(215,170,125),np.uint8);im[:,:55]=(45,65,100);im[70:,80:]=(70,120,65)
    rng=np.random.default_rng(9)
    for y,x in rng.integers([2,2],[118,158],size=(160,2)):im[y,x]=(240,240,240)
    return im

def test_simplification_discrete_counts_and_determinism():
    p=BeadParams(80,8,0,5,3)
    a=make_pattern(noisy_scene(),p);b=make_pattern(noisy_scene(),p)
    assert np.array_equal(a.labels,b.labels) and np.array_equal(a.palette,b.palette)
    assert a.labels.shape==(60,80) and len(a.palette)<=8
    assert sum(row['count'] for row in color_rows(a))==4800
    assert a.metrics['after']['isolated_beads']<a.metrics['before']['isolated_beads']
    assert len(np.unique(a.palette[a.labels].reshape(-1,3),axis=0))==len(a.palette)

def test_four_connected_regions_and_merging():
    diagonal=np.array([[0,1],[1,0]],np.uint8)
    assert region_metrics(diagonal)['isolated_beads']==4
    labels=np.zeros((20,30),np.uint8);labels[:,15:]=1;labels[5,5]=1
    out=cleanup_regions(labels,np.array([[0,0,0],[100,0,0]],np.float32),4)
    assert out[5,5]==0 and np.all(out[:,15:]==1)
    protected=np.zeros(labels.shape,bool);protected[5,5]=True
    preserved=cleanup_regions(labels,np.array([[0,0,0],[100,0,0]],np.float32),4,protected=protected)
    assert preserved[5,5]==1

def test_custom_palette_exact_no_fabricated_codes(tmp_path):
    path=tmp_path/'colors.csv';path.write_text('code,r,g,b,name\nMY-BLACK,0,0,0,黑豆\nMY-WHITE,255,255,255,白豆\nMY-TAN,215,170,125,肤色\n',encoding='utf-8-sig')
    palette=read_palette(path);pattern=make_pattern(noisy_scene(),BeadParams(colors=2),palette)
    assert len(pattern.palette)<=2
    assert set(pattern.codes).issubset(set(palette[1]))
    assert all(any(np.array_equal(color,available) for available in palette[0]) for color in pattern.palette)

@pytest.mark.parametrize('body',['r,g,b\n1,2,3','code,r,g,b\nx,256,0,0','code,r,g,b\nx,1,2,3\nx,4,5,6','code,r,g,b\n'])
def test_bad_palette(tmp_path,body):
    path=tmp_path/'bad.csv';path.write_text(body,encoding='utf-8')
    with pytest.raises(ValueError):read_palette(path)

def test_exports_match_grid_and_pdf_page_ranges():
    pattern=make_pattern(noisy_scene(),BeadParams(64,6,2,4,2,5))
    files=export_files(pattern);h,w=pattern.labels.shape
    pdf=PdfReader(BytesIO(files['pattern.pdf']))
    assert len(pdf.pages)==1+math.ceil(h/29)*math.ceil(w/29)
    assert abs(float(pdf.pages[0].mediabox.width)-595.2756)<.1
    assert 'Columns 59-64' in pdf.pages[-1].extract_text()
    assert '50 mm' in pdf.pages[-1].extract_text()
    text='\n'.join(p.extract_text() for p in pdf.pages)
    assert 'Print at 100%' in text and 'B01' in text
    counts=list(csv.DictReader(StringIO(files['colors.csv'].decode('utf-8-sig'))))
    assert sum(int(row['count']) for row in counts)==h*w
    data=json.loads(files['pattern.json']);grid=list(csv.reader(StringIO(files['grid.csv'].decode('utf-8-sig'))))
    assert len(grid)==h+1 and len(grid[0])==w+1
    assert [row[1:] for row in grid[1:]]==data['grid']
    ET.fromstring(files['chart.svg'])
    image=Image.open(BytesIO(files['chart.png']));image.verify()
    with zipfile.ZipFile(BytesIO(package_files(files))) as archive:assert set(archive.namelist())==set(files)

def test_csv_formula_protection_and_unicode_code():
    palette=(np.array([[0,0,0]],np.uint8),['=unsafe'],['测试'])
    result=make_pattern(np.zeros((10,20,3),np.uint8),BeadParams(16,2),palette)
    files=export_files(result)
    assert "'=unsafe" in files['colors.csv'].decode('utf-8-sig')
    assert json.loads(files['pattern.json'])['palette'][0]['code']=='=unsafe'

@pytest.mark.parametrize('shape',[(1,1,3),(9,30,3),(40,40)])
def test_edge_sizes(shape):
    pattern=make_pattern(np.full(shape,90,np.uint8))
    assert len(pattern.palette)==1 and pattern.labels.size>0 and max(pattern.labels.shape)==72

def test_validation_cancel_and_clamps():
    with pytest.raises(ValueError):normalized(replace(BeadParams(),cohesion=float('nan')))
    with pytest.raises(ValueError):make_pattern(np.empty((0,2,3),np.uint8))
    with pytest.raises(InterruptedError):make_pattern(noisy_scene(),cancel=lambda:True)
    p=normalized(BeadParams(999,999,-1,999,999,999))
    assert (p.grid,p.colors,p.smooth,p.min_region,p.cohesion,p.pitch_mm)==(160,32,0,12,4,5)

def test_beads_share_ui_queue():
    from app import build_app
    demo=build_app();handlers=[fn for fn in demo.fns.values() if fn.api_name=='beads']
    assert len(handlers)==1 and handlers[0].concurrency_id=='cpu'

@pytest.mark.parametrize('pitch',[2.6,5.])
def test_maximum_grid_pdf_last_tile_and_pitch(pitch):
    from cartoonify.beads import pattern_pdf
    pattern=make_pattern(np.full((32,32,3),120,np.uint8),BeadParams(160,32,pitch_mm=pitch))
    reader=PdfReader(BytesIO(pattern_pdf(pattern)))
    assert len(reader.pages)==37
    text=reader.pages[-1].extract_text()
    assert 'Columns 146-160' in text and 'Rows 146-160' in text and f'{pitch:g} mm cells' in text

def test_failed_disk_export_cleans_only_its_own_files(tmp_path,monkeypatch):
    from pathlib import Path
    from cartoonify import bead_service
    from cartoonify.service import _JOBS
    source=tmp_path/'source.png';Image.new('RGB',(20,20),'tan').save(source)
    exports=tmp_path/'exports';exports.mkdir();sentinel=exports/'keep.txt';sentinel.write_text('keep')
    monkeypatch.setattr(bead_service,'EXPORTS',exports)
    original=Path.write_bytes
    def disk_error(path,data):
        if path.name.endswith('.zip.part'):raise OSError('Simulated disk write failure')
        return original(path,data)
    monkeypatch.setattr(Path,'write_bytes',disk_error)
    with pytest.raises(OSError):bead_service.create_export(source,16,2,0,2,0,5,False,'自动精简颜色',None,'disk-test')
    assert list(exports.iterdir())==[sentinel] and sentinel.read_text()=='keep'
    assert 'disk-test' not in _JOBS

def test_long_color_codes_fit_pdf_but_remain_complete_in_data():
    from reportlab.pdfbase.pdfmetrics import stringWidth
    code='W'*48
    pattern=make_pattern(np.zeros((20,20,3),np.uint8),BeadParams(16,2,preserve_features=False),
                         (np.array([[0,0,0]],np.uint8),[code],['physical color']))
    files=export_files(pattern);reader=PdfReader(BytesIO(files['pattern.pdf']))
    label=next(line for line in reader.pages[0].extract_text().splitlines() if line.startswith('B01 '))
    assert '...' in label and stringWidth(label,'Helvetica',9)<=230
    assert json.loads(files['pattern.json'])['palette'][0]['code']==code
