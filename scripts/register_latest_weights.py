"""Register uploaded CEDAR local/sweep weights in the MDI repository's catalogs."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from release_assets import digest

ROOT = Path(__file__).resolve().parents[1]


def expected_models():
    return [f'sig_yolo_v8{s}_f{f}t{t}_c{c}' for s in 'nsx' for f in [20, 40, 80]
            for t in [20, 40, 80] for c in [25, 50, 100, 150]]


def register(bundle, mdi_repo=ROOT, require_complete=False):
    bundle, mdi_repo = Path(bundle).resolve(), Path(mdi_repo).resolve()
    catalog_path = mdi_repo / 'configs/release_assets.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if catalog.get('project') != 'mdi' or catalog.get('schema') != 2:
        raise ValueError('Expected the MDI repository asset catalog')
    existing = {r['source']: r for r in catalog['files']}
    prefix = 'weights/cedar_local_latest/'
    expected = expected_models()
    found = []
    for name in expected:
        path = bundle / prefix / name / 'weights/best.pt'
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            raise ValueError(f'Empty checkpoint: {path}')
        sha = digest(path)
        rel = path.relative_to(bundle).as_posix()
        if rel in existing and existing[rel]['sha256'] != sha:
            raise ValueError(f'Previously registered checkpoint changed: {rel}')
        found.append({'source': rel, 'destination': f'trained_models/{name}/weights/best.pt',
                      'bytes': path.stat().st_size, 'sha256': sha})
    missing = sorted(set(expected) - {Path(r['source']).parts[2] for r in found})
    if require_complete and missing:
        raise ValueError(f'{len(missing)} of {len(expected)} later checkpoints are still absent; no catalogs updated.')
    previous = {s for s in existing if s.startswith(prefix)}
    if previous - {r['source'] for r in found}:
        raise FileNotFoundError('A previously registered checkpoint is missing')
    catalog['files'] = [r for r in catalog['files'] if not r['source'].startswith(prefix)] + found
    catalog['expected_latest_cedar_models'] = expected
    model_path = mdi_repo / 'configs/release_checkpoints.json'
    models = json.loads(model_path.read_text(encoding='utf-8'))
    models['sets']['cedar_local_latest'] = [
        {'model': Path(r['source']).parts[2], 'path': r['destination'], 'sha256': r['sha256']} for r in found]
    models['expected_latest_cedar_models'] = expected
    models['latest_status'] = 'complete' if not missing else ('partial' if found else 'reserved_for_owner_upload')
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
    model_path.write_text(json.dumps(models, indent=2) + '\n', encoding='utf-8')
    (mdi_repo / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n"
        for group in models['sets'].values() for r in group), encoding='utf-8')
    print(json.dumps({'registered': len(found), 'expected': len(expected), 'missing': len(missing)}))
    return found, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT.parent / 'CBRS-CEDAR-MDI-ASSETS')
    p.add_argument('--mdi-repo', type=Path, default=ROOT)
    p.add_argument('--source', type=Path, help='Uploaded trained_models directory; copy its best.pt files before registration')
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()
    if a.source:
        plan = []
        for name in expected_models():
            src = a.source / name / 'weights/best.pt'
            if src.is_file():
                dst = a.bundle / 'weights/cedar_local_latest' / name / 'weights/best.pt'
                if dst.exists() and digest(dst) != digest(src):
                    raise ValueError(f'Different checkpoint already exists: {dst}')
                plan.append((src, dst))
        if a.require_complete and len(plan) != len(expected_models()):
            raise ValueError(f'Source has {len(plan)}/{len(expected_models())} expected best.pt files; none copied')
        for src, dst in plan:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                if digest(src) != digest(dst):
                    raise ValueError(f'Copy checksum mismatch: {dst}')
    register(a.bundle, a.mdi_repo, a.require_complete)


if __name__ == '__main__':
    main()
