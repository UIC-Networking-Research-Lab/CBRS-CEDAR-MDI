"""Predict one reference-style PNG with original YOLO or CEDAR-YOLO."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=ROOT/'experiment_reruns/prediction.json')
    a=p.parse_args()
    import torch
    import numpy as np
    from PIL import Image
    from cbrs.training_state import device_name
    from scripts.cedar.test_cedar_yolo import build_model_from_ckpt, overlap_from_detections
    image=Image.open(a.image).convert('RGB')
    if image.size != (640,640): raise ValueError('Expected supplied 640x640 spectrogram; input is not resized.')
    device=torch.device(device_name(a.device))
    if 'sig_yolo_' in str(a.weights):
        ck=torch.load(a.weights,map_location='cpu',weights_only=False)
        run=a.weights.parent.parent.name
        built=build_model_from_ckpt(ck,run_name=run)
        if built is None: raise ValueError('Unable to construct CEDAR-YOLO from checkpoint')
        model,_,nc=built
        model.to(device).eval()
        tensor=torch.from_numpy(np.array(image).transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
        with torch.inference_mode():
            boxes=model.predict(tensor,conf_thres=0.25,iou_thres=0.45)[0].cpu()
    else:
        from ultralytics import YOLO
        out=YOLO(a.weights).predict(source=np.asarray(image)[:,:,::-1].copy(),device=str(device),conf=0.25,iou=0.45,verbose=False)[0]
        boxes=out.boxes.data.cpu()
    overlap,ratios=overlap_from_detections(boxes)
    result={'image':a.image.name,'checkpoint':a.weights.name,'classes':{'0':'LTE','1':'Radar'},
            'boxes_xyxy_conf_class':boxes.tolist(),'frequency_overlap':overlap,'overlap_ratios':ratios}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
