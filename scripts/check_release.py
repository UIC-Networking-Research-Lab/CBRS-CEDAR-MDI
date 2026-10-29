"""Check the installed fixed PNG split, checkpoint set and optional raw renderer."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import platform
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.release_assets import digest


def check(root=ROOT, weights='jetson', raw=False, data_only=False, max_samples=None):
    root=Path(root)
    errors=[]
    rows=list(csv.DictReader((root/'splits/cbrs_split.csv').open(encoding='utf-8')))
    expected={s:{r['stem'] for r in rows if r['split']==s} for s in ['train','val','test']}
    for s,stems in expected.items():
        for folder,extension in [('images','.png'),('labels','.txt')]:
            found={p.stem for p in (root/'yolo_split'/s/folder).glob('*'+extension)}
            if found != stems:
                errors.append(f'{s}/{folder}: missing={len(stems-found)}, extra={len(found-stems)}')
        for p in (root/'yolo_split'/s/'images').glob('*.png'):
            with p.open('rb') as f: header=f.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',header[16:24]) != (640,640):
                errors.append(f'Expected 640x640 PNG: {p}')
    if not data_only:
        manifest=json.loads((root/'configs/release_checkpoints.json').read_text(encoding='utf-8'))
        sets=['original_yolo', 'cedar_jetson' if weights=='jetson' else 'cedar_local_latest']
        if weights=='latest':
            core={f'sig_yolo_v8{s}_f20t20_c100' for s in 'nsx'}
            present={x['model'] for x in manifest['sets']['cedar_local_latest']}
            if core-present:
                errors.append(f'Later local checkpoints still pending: {sorted(core-present)}')
        for group in sets:
            records=manifest['sets'][group]
            if not records:
                errors.append(f'Checkpoint set unavailable: {group}')
            for rec in records:
                p=root/rec['path']
                if not p.is_file(): errors.append(f'Missing checkpoint: {p}')
                elif digest(p)!=rec['sha256']: errors.append(f'Wrong checkpoint edition: {p}')
    versions={}
    for name in ['torch','ultralytics','numpy','scipy','matplotlib','Pillow','flask','requests']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='missing'
    for name in ['torch','ultralytics','numpy','Pillow']:
        if versions[name]=='missing': errors.append(f'Missing dependency: {name}; install requirements.txt and PyTorch for your platform')
    if raw:
        raw_dir=root/'raw_sigmf_split/test'
        metas={p.stem for p in raw_dir.glob('*.sigmf-meta')}
        recordings={p.stem for p in raw_dir.glob('*.sigmf-data')}
        if metas != recordings or not metas.issubset(expected['test']):
            errors.append('Raw test split has incomplete pairs or sample IDs outside the fixed test split')
        required=len(expected['test']) if max_samples is None else max_samples
        if len(metas)<required:
            errors.append(f'Raw test split needs {required} complete frames; found {len(metas)}. Download the official I/Q dataset and run --task raw-split first.')
        scipy_version='1.17.1' if platform.machine().lower()=='aarch64' else '1.14.1'
        pinned={'numpy':'2.2.4','scipy':scipy_version,'matplotlib':'3.10.8','Pillow':'11.2.1'}
        for k,v in pinned.items():
            if versions[k]!=v: errors.append(f'Raw renderer requires {k}=={v}; found {versions[k]}')
    print(json.dumps({'split_counts':{k:len(v) for k,v in expected.items()},'weights':weights,'environment':versions,'errors':errors},indent=2))
    if errors: raise SystemExit(1)
    return versions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',choices=['jetson','latest'],default='jetson')
    p.add_argument('--raw',action='store_true')
    p.add_argument('--data-only',action='store_true')
    a=p.parse_args()
    check(weights=a.weights,raw=a.raw,data_only=a.data_only)
