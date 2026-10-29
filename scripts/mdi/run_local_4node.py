"""
Run a four-node MDI script on one machine (127.0.0.1) as a functional test.

Starts worker nodes B, C and D in the background, then runs node A in the
foreground and stops the workers when A finishes. Every argument after the
script name is passed to all four nodes.

This checks that the distributed pipeline works end to end; timings measured on
one machine do not reproduce the Jetson testbed results.

Usage (from the repository root):
    python scripts/mdi/run_local_4node.py scripts/mdi/yolo_mdi_compressed.py \
        --model_name cbrs_yolo8n --compressor quantization --compression_param int8 \
        --sweep_rates_mbps 150 --max_samples 10 --output_dir mdi_local_test

    python scripts/mdi/run_local_4node.py scripts/mdi/cedar_mdi.py \
        --model_name sig_yolo_v8n_f20t20_c100 --compressor none \
        --sweep_rates_mbps 150 --max_samples 10 --output_dir mdi_local_test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCALHOST_ARGS = [
    "--ip_a", "127.0.0.1", "--ip_b", "127.0.0.1", "--ip_c", "127.0.0.1", "--ip_d", "127.0.0.1",
]
WORKER_NODES = ("B", "C", "D")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", type=Path, help="MDI script, e.g. scripts/mdi/cedar_mdi.py")
    parser.add_argument("--startup_delay", type=float, default=10.0,
                        help="Seconds to wait for the workers to bind their ports before starting A.")
    args, node_args = parser.parse_known_args()

    script = args.script if args.script.is_absolute() else PROJECT_ROOT / args.script
    workers = []
    try:
        for node in WORKER_NODES:
            cmd = [sys.executable, str(script), "--node", node, *LOCALHOST_ARGS, *node_args]
            workers.append(subprocess.Popen(cmd, cwd=PROJECT_ROOT))
        time.sleep(args.startup_delay)
        for node, proc in zip(WORKER_NODES, workers):
            if proc.poll() is not None:
                print(f"Node {node} exited early with code {proc.returncode}")
                return 1
        cmd = [sys.executable, str(script), "--node", "A", *LOCALHOST_ARGS, *node_args]
        return subprocess.call(cmd, cwd=PROJECT_ROOT)
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
        for proc in workers:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
