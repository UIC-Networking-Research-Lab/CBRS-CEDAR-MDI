"""One-command entry points for the four paper experiment pipelines."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def commands(a):
    models_dir='trained_models_mdi' if a.weights=='jetson' else 'trained_models'
    py=sys.executable
    def cmd(path,*args): return [py,'-B',path,*map(str,args)]
    if a.task=='offline':
        return [cmd('scripts/mdi/build_cedar_partition_profiles.py')]
    if a.task=='raw-split':
        if a.raw_source is None: raise ValueError('--raw-source must point to the separately downloaded dataset')
        return [cmd('scripts/data/restore_raw_split.py','--source',a.raw_source)]
    if a.task=='png':
        yolo=a.models or [f'cbrs_yolo{v}{s}' for v in [8,11,26] for s in 'nsx']
        cedar=[n for n in yolo if n.startswith('sig_yolo_')]
        if not a.models: cedar=[f'sig_yolo_v8{s}_f20t20_c100' for s in 'nsx']
        out=[cmd('scripts/yolo/test_yolo.py','--models',f'trained_models/{n}','--device',a.device,'--no-predict') for n in yolo if n.startswith('cbrs_')]
        if cedar:
            out.append(cmd('scripts/cedar/test_cedar_yolo.py','--models_dir',models_dir,'--models',*cedar,
                           '--device',a.device,'--no_viz','--no_sweep_plots','--out_dir',f'experiment_reruns/png_{a.weights}'))
        return out
    if a.task in ('train-yolo','train-cedar'):
        return [cmd('scripts/yolo/train_yolo.py' if a.task=='train-yolo' else 'scripts/cedar/train_cedar_yolo.py')]
    if a.task in ('local-yolo','local-cedar'):
        path='scripts/yolo/local_sigmf_stage_cost_test.py' if a.task=='local-yolo' else 'scripts/cedar/local_cedar_stage_cost_test.py'
        opts=['--sweep','--sweep_devices',a.device]
        if a.models:
            opts=['--no-sweep','--model_name',a.models[0],'--device',a.device]
        if a.task=='local-cedar': opts+=['--models_dir',models_dir]
        if a.max_samples: opts+=['--max_samples',str(a.max_samples)]
        return [cmd(path,*opts)]
    if a.task in ('yolo-mdi','cedar-mdi'):
        path='scripts/mdi/yolo_mdi_compressed.py' if a.task=='yolo-mdi' else 'scripts/mdi/cedar_mdi.py'
        opts=['--device',a.device]
        if a.models: opts+=['--model_name',a.models[0]]
        if a.task=='cedar-mdi': opts+=['--models_dir',models_dir]
        if a.max_samples: opts+=['--max_samples',str(a.max_samples)]
        if a.local_nodes:
            opts+=['--compressor','none','--sweep_rates_mbps','150']
            opts+=['--output_dir','experiment_reruns/smoke/'+a.task]
            if a.task=='cedar-mdi': opts+=['--partition_top_k','1']
            return [cmd('scripts/mdi/run_local_4node.py',path,*opts,'--run_location','local')]
        if not a.node or not a.hosts:
            raise ValueError('Set --node A/B/C/D and --hosts <A> <B> <C> <D>, or --local-nodes for a functional check')
        for node,host in zip('abcd',a.hosts): opts+=['--ip_'+node,host]
        return [cmd(path,'--node',a.node,*opts)]
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['install','check','png','offline','raw-split','train-yolo','train-cedar','local-yolo','local-cedar','yolo-mdi','cedar-mdi'],default='check')
    p.add_argument('--assets',type=Path,help='Extracted CBRS-CEDAR-MDI-ASSETS folder; verify and copy assets first')
    p.add_argument('--weights',choices=['latest','jetson'],default='jetson',
                   help='Checkpoint edition (default: supplied Jetson weights; latest requires the later upload)')
    p.add_argument('--models',nargs='+')
    p.add_argument('--device',default='cpu')
    p.add_argument('--raw-source',type=Path)
    p.add_argument('--node',choices=list('ABCD'))
    p.add_argument('--hosts',nargs=4,metavar=('A','B','C','D'))
    p.add_argument('--local-nodes',action='store_true')
    p.add_argument('--max-samples',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    plan=commands(a)
    if a.dry_run:
        if a.assets: print('Install checked assets from',a.assets)
        for c in plan: print(subprocess.list2cmdline(c))
        return
    if a.assets:
        from scripts.release_assets import install
        install(a.assets,ROOT,'mdi')
    if a.task=='install':
        if not a.assets: p.error('--task install requires --assets')
        return
    if a.task not in ('offline','raw-split') and (not a.node or a.node=='A'):
        from scripts.check_release import check
        check(weights=a.weights,raw=a.task in ('local-yolo','local-cedar','yolo-mdi','cedar-mdi'),data_only=a.task.startswith('train-'),max_samples=a.max_samples)
    for c in plan:
        subprocess.run(c,cwd=ROOT,check=True)


if __name__=='__main__':
    main()
