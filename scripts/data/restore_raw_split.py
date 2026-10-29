"""Restore the fixed paper split from a separately downloaded SigMF dataset."""
from __future__ import annotations
import argparse
import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def restore(source, output, manifest=ROOT/'splits/cbrs_split.csv', splits=('train','val','test'), method='copy', limit=None):
    rows = [r for r in csv.DictReader(Path(manifest).open(encoding='utf-8')) if r['split'] in splits]
    if limit is not None:
        rows = rows[:limit]
    lookup = {}
    for p in Path(source).rglob('*.sigmf-meta'):
        if p.stem in lookup:
            raise ValueError(f'Duplicate sample in raw source: {p.stem}')
        lookup[p.stem] = p
    plan = []
    for row in rows:
        meta = lookup.get(row['stem'])
        if meta is None:
            raise FileNotFoundError(f"Missing downloaded sample: {row['stem']}")
        raw_split = 'validation' if row['split'] == 'val' else row['split']
        for src in [meta, meta.with_suffix('.sigmf-data')]:
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = Path(output)/raw_split/src.name
            if dst.exists() and dst.stat().st_size != src.stat().st_size:
                raise FileExistsError(f'Different existing file: {dst}')
            plan.append((src,dst))
    for i,(src,dst) in enumerate(plan):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            if method == 'hardlink':
                os.link(src,dst)
            else:
                shutil.copy2(src,dst)
        if i % 1000 == 0:
            print(f'Restoring {i}/{len(plan)} SigMF files',flush=True)
    print(f'Restored {len(rows)} fixed-split frames into {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Downloaded IEEE_UPLOAD or its parent')
    p.add_argument('--output',type=Path,default=ROOT/'raw_sigmf_split')
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test'])
    p.add_argument('--method',choices=['copy','hardlink'],default='copy')
    p.add_argument('--limit',type=int,help='Functional check only; omit for paper experiments')
    a=p.parse_args()
    restore(a.source,a.output,splits=a.splits,method=a.method,limit=a.limit)


if __name__ == '__main__':
    main()
