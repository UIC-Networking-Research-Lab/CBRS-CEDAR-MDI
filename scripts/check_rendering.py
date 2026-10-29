"""Compare rendered raw-I/Q pixels against the supplied reference PNGs."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-source',type=Path,required=True)
    p.add_argument('--png-root',type=Path,default=ROOT/'yolo_split')
    p.add_argument('--output',type=Path,default=ROOT/'experiment_reruns/rendering_check.json')
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    from cbrs.sigmf_preprocessor import SigMFSpectrogramPreprocessor
    rows=list(csv.DictReader((ROOT/'splits/cbrs_split.csv').open()))
    chosen={}
    for r in rows: chosen.setdefault((r['split'],r['band']),r)
    sources={q.stem:q for q in a.raw_source.rglob('*.sigmf-meta')}
    pre=SigMFSpectrogramPreprocessor()
    results=[]
    for row in chosen.values():
        meta=sources[row['stem']]
        got=pre.raw_to_image_array(meta,meta.with_suffix('.sigmf-data'))
        ref=np.asarray(Image.open(a.png_root/row['split']/'images'/(row['stem']+'.png')).convert('RGB'))
        diff=np.abs(got.astype(np.int16)-ref.astype(np.int16))
        results.append({'split':row['split'],'band':row['band'],'stem':row['stem'],
                        'max_channel_difference':int(diff.max()),'changed_channels':int(np.count_nonzero(diff)),
                        'mean_absolute_difference':float(diff.mean())})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
    if any(r['max_channel_difference']>1 or r['mean_absolute_difference']>0.001 for r in results):
        raise SystemExit('Renderer differs from reference PNGs. Use supplied PNGs or retrain with your rendering pipeline.')


if __name__=='__main__':
    main()
