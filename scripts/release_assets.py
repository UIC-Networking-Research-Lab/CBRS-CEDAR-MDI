"""Install this project's data/model bundle. Verification metadata stays in the repository."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "mdi"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Invalid relative path: {relative}')
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside destination: {relative}')
    return target


def install(bundle, destination, project=PROJECT, verify_only=False, *, catalog_path=None):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    catalog_path = Path(catalog_path) if catalog_path else ROOT / 'configs/release_assets.json'
    data = json.loads(catalog_path.read_text(encoding='utf-8'))
    if project != PROJECT or data.get('schema') != 2 or data.get('project') != project:
        raise ValueError(f'Expected the {PROJECT.upper()} repository and its project-specific asset catalog')
    checked = []
    # Check the entire package and existing destinations before writing any file.
    for i, row in enumerate(data['files']):
        src = safe_path(bundle, row['source'])
        dst = safe_path(destination, row['destination'])
        if src.stat().st_size != row['bytes'] or digest(src) != row['sha256']:
            raise ValueError(f'Asset checksum mismatch: {src}')
        if not verify_only and dst.exists() and digest(dst) != row['sha256']:
            raise FileExistsError(f'Different file already exists: {dst}')
        checked.append((src, dst, row['sha256']))
        if i % 2000 == 0:
            print(f'Verifying {project}: {i}/{len(data["files"])} files', flush=True)
    if verify_only:
        print(f'Verified {len(checked)} {project} assets; no files installed.')
        return data
    for src, dst, sha in checked:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if digest(dst) != sha:
                raise ValueError(f'Installed checksum mismatch: {dst}')
    image_root = destination / ('yolo_split' if project == 'mdi' else 'data/yolo_split')
    image_root.mkdir(parents=True, exist_ok=True)
    yaml = ('path: ' + json.dumps(image_root.as_posix()) + '\n'
            'train: train/images\nval: val/images\ntest: test/images\n'
            'nc: 2\nnames: [LTE, Radar]\n')
    (image_root / 'data.yaml').write_text(yaml, encoding='utf-8')
    if project == 'mdi':
        groups = {k: [] for k in ['original_yolo', 'cedar_jetson', 'cedar_local_latest']}
        for row in data['files']:
            parts = PurePosixPath(row['source']).parts
            if len(parts) >= 3 and parts[0] == 'weights' and parts[1] in groups:
                groups[parts[1]].append({'model': parts[2], 'path': row['destination'], 'sha256': row['sha256']})
        expected = data.get('expected_latest_cedar_models', [])
        found = {r['model'] for r in groups['cedar_local_latest']}
        status = 'complete' if expected and found == set(expected) else ('partial' if found else 'reserved_for_owner_upload')
        catalog = {'schema': 1, 'sets': groups, 'expected_latest_cedar_models': expected, 'latest_status': status}
        (destination / 'configs').mkdir(parents=True, exist_ok=True)
        (destination / 'configs/release_checkpoints.json').write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
        (destination / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for group in groups.values() for r in group), encoding='utf-8')
        for name in expected:
            safe_path(destination, f'trained_models/{name}/weights').mkdir(parents=True, exist_ok=True)
    else:
        records = [{'model_id': PurePosixPath(r['source']).stem, 'path': r['destination'],
                    'sha256': r['sha256'], 'source': r['source']}
                   for r in data['files'] if r['source'].startswith('weights/')]
        (destination / 'artifacts/weights').mkdir(parents=True, exist_ok=True)
        (destination / 'artifacts/weights/manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
        (destination / 'artifacts/dataset.json').write_text(json.dumps(data['dataset'], indent=2) + '\n', encoding='utf-8')
    print(f'Installed {len(checked)} files. Data: {image_root}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--destination', type=Path, default=ROOT)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    install(a.bundle, a.destination, verify_only=a.verify_only)


if __name__ == '__main__':
    main()
