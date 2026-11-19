"""Write a slim copy of cubist_side.npz (drops the 120-PC score matrices, which exceed the
30 MB transfer limit and are fully reproducible with 09a_p2_cubist.py).

python scripts/11_slim_p2_arrays.py [--root P2|V_visnir] SCENARIO [SCENARIO ...]
"""
import argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="P2")
ap.add_argument("scenarios", nargs="+")
args = ap.parse_args()
for s in args.scenarios:
    base = ROOT / "results" / args.root / s
    d = dict(np.load(base / "cubist_side.npz"))
    keep = {k: d[k] for k in ["tr", "tg", "pool", "test", "ytr", "ytg", "cub_tg", "r_tr", "Ztr", "Ztg", "mbl_tg", "keep"]}
    np.savez_compressed(base / "cubist_side_slim.npz", **keep)
    print(s, round((base / "cubist_side_slim.npz").stat().st_size / 1e6, 1), "MB")
