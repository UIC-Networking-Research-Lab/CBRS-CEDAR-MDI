"""Export this repository's installed data and weights as a simple standalone download."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from release_assets import digest, safe_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT, help='Repository containing installed data and weights')
    p.add_argument('--output', type=Path, default=ROOT.parent / (ROOT.name + '-ASSETS'))
    a = p.parse_args()
    catalog = json.loads((ROOT / 'configs/release_assets.json').read_text(encoding='utf-8'))
    for i, row in enumerate(catalog['files']):
        src = safe_path(a.source, row['destination'])
        dst = safe_path(a.output, row['source'])
        if src.stat().st_size != row['bytes'] or digest(src) != row['sha256']:
            raise ValueError(f'Source does not match the reference asset: {src}')
        if dst.exists():
            if digest(dst) != row['sha256']:
                raise FileExistsError(f'Different file already exists: {dst}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if digest(dst) != row['sha256']:
                raise ValueError(f'Copy mismatch: {dst}')
        if i % 2000 == 0:
            print(f'Exported {i}/{len(catalog["files"])} assets', flush=True)
    if catalog['project'] == 'mdi':
        for name in catalog['expected_latest_cedar_models']:
            safe_path(a.output, f'weights/cedar_local_latest/{name}/weights').mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'docs/download_bundle.md', a.output / 'README.md')
    print(f'Download ready: {a.output.resolve()}')


if __name__ == '__main__':
    main()
