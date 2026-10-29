import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cbrs.training_state import completed, advance, device_name
from scripts.release_assets import digest, install, safe_path
from scripts.mdi.build_cedar_partition_profiles import link_bandwidth_mbps


class ReleaseTests(unittest.TestCase):
    def test_default_check_uses_available_jetson_weights(self):
        import reproduce
        with patch.object(sys, 'argv', ['reproduce.py']), patch('scripts.check_release.check') as check:
            reproduce.main()
        self.assertEqual(check.call_args.kwargs['weights'], 'jetson')

    def test_png_forwards_selected_device_to_both_detectors(self):
        import reproduce
        args=SimpleNamespace(task='png', weights='jetson', device='cuda:0',
                             models=['cbrs_yolo8n','sig_yolo_v8n_f20t20_c100'])
        commands=reproduce.commands(args)
        self.assertEqual(len(commands),2)
        for command in commands:
            self.assertEqual(command[command.index('--device')+1],'cuda:0')

    def test_interrupted_epoch25_is_not_finished(self):
        self.assertFalse(completed({'epoch':24},100))
        self.assertFalse(completed({'epoch':24,'training_complete':False,'target_epochs':100},100))

    def test_early_stop_is_finished(self):
        self.assertTrue(completed({'epoch':20,'training_complete':True,'target_epochs':100},100))
        self.assertFalse(completed({'epoch':20,'training_complete':True,'target_epochs':100},200))

    def test_flat_metric_exhausts_patience(self):
        best, stale = 0.99, 0
        for _ in range(20): best,stale=advance(0.99,best,stale)
        self.assertEqual(stale,20)
        self.assertEqual(advance(0.995,best,stale),(0.995,0))

    def test_device_names(self):
        for value, expected in [('0','cuda:0'),('cuda','cuda'),('cuda:1','cuda:1'),('cpu','cpu')]:
            self.assertEqual(device_name(value),expected)

    def test_missing_bandwidth_is_not_nominal_rate(self):
        with self.assertRaises(ValueError): link_bandwidth_mbps(150,'A_to_B',{})

    def test_installer_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relative in ['../x','/x','C:/x','x\\y']:
                with self.assertRaises(ValueError): safe_path(Path(tmp),relative)

    def test_installer_validates_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            (bundle/'a.txt').write_text('original')
            manifest={'schema':2,'project':'mdi','files':[{'source':'a.txt','destination':'yolo_split/a.txt',
                      'bytes':8,'sha256':digest(bundle/'a.txt')}]}
            catalog=root/'repository_assets.json'
            catalog.write_text(json.dumps(manifest))
            (bundle/'a.txt').write_text('modified')
            with self.assertRaises(ValueError): install(bundle,root/'repo','mdi',catalog_path=catalog)
            self.assertFalse((root/'repo/yolo_split/a.txt').exists())

    def test_installer_imports_new_checkpoint_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            model='sig_yolo_v8n_f20t20_c100'
            source=f'weights/cedar_local_latest/{model}/weights/best.pt'
            weight=bundle/source;weight.parent.mkdir(parents=True);weight.write_bytes(b'test checkpoint')
            png=bundle/'sample.png';png.write_bytes(b'test image bytes')
            mappings=[{'source':source,'destination':f'trained_models/{model}/weights/best.pt'},
                      {'source':'sample.png','destination':'yolo_split/train/images/sample.png'}]
            files=[dict(mapping,bytes=p.stat().st_size,sha256=digest(p)) for mapping,p in zip(mappings,[weight,png])]
            manifest={'schema':2,'project':'mdi','files':files,
                      'expected_latest_cedar_models':[model,'sig_yolo_v8s_f20t20_c100']}
            metadata=root/'repository_assets.json'
            metadata.write_text(json.dumps(manifest))
            install(bundle,root/'repo','mdi',catalog_path=metadata)
            catalog=json.loads((root/'repo/configs/release_checkpoints.json').read_text())
            self.assertEqual(catalog['sets']['cedar_local_latest'][0]['sha256'],digest(weight))
            self.assertEqual(catalog['latest_status'],'partial')
            self.assertEqual(catalog['sets']['cedar_jetson'],[])
            # Reinstalling identical assets is safe; a different existing weight is rejected.
            install(bundle,root/'repo','mdi',catalog_path=metadata)
            (root/'repo'/mappings[0]['destination']).write_bytes(b'different checkpoint')
            with self.assertRaises(FileExistsError):install(bundle,root/'repo','mdi',catalog_path=metadata)

    def test_installer_rejects_other_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            catalog=root/'catalog.json'
            catalog.write_text(json.dumps({'schema':2,'project':'rt','files':[]}))
            with self.assertRaises(ValueError):install(root,root/'repo','mdi',catalog_path=catalog)

    def test_installer_verify_only_keeps_bundle_and_destination_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            (bundle/'splits.csv').write_text('split,band,stem\ntrain,10M,sample\n')
            catalog=root/'catalog.json'
            catalog.write_text(json.dumps({'schema':2,'project':'mdi','files':[{
                'source':'splits.csv','destination':'splits/cbrs_split.csv',
                'bytes':(bundle/'splits.csv').stat().st_size,'sha256':digest(bundle/'splits.csv')}]}))
            install(bundle,root/'repo','mdi',True,catalog_path=catalog)
            self.assertFalse((root/'repo').exists())
            self.assertEqual([p.name for p in bundle.iterdir()],['splits.csv'])


if __name__=='__main__': unittest.main()
