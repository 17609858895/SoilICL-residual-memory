"""Retrieval provenance for leave-source-out scenarios (exploratory, descriptive).

For each held-out scenario: which training sources supply the k = 20 retrieval neighbours of the held-out
spectra, in the PLS-20 space (Cubist side) and the CNN embedding space, and the enrichment
= neighbour share / share of the training library.

python scripts/13_retrieval_provenance.py --data mir
Writes results/<root>/provenance_enrichment.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import DATASETS, SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="mir", choices=list(DATASETS))
args = ap.parse_args()
_, mfile, _, root = DATASETS[args.data]
meta = pd.read_parquet(ROOT / "data" / mfile)
rows = []
for s in SCENARIOS[args.data]:
    d = ROOT / "results" / root / s
    if not (d / "cubist_side.npz").exists() or not (d / "cnn_side.npz").exists():
        print("missing", s); continue
    C, N = np.load(d / "cubist_side.npz"), np.load(d / "cnn_side.npz")
    src = meta["source"].to_numpy()[C["tr"]]
    lib = pd.Series(src).value_counts(normalize=True)
    for space, (A, B) in {"PLS20": (C["Ztr"], C["Ztg"]), "CNN": (N["z_tr"], N["z_tg"])}.items():
        _, idx = NearestNeighbors(n_neighbors=20).fit(A).kneighbors(B)
        share = pd.Series(src[idx.ravel()]).value_counts(normalize=True)
        for so in lib.index:
            sh = float(share.get(so, 0.0))
            rows.append({"scenario": s, "space": space, "neighbour_source": so, "library_share": float(lib[so]),
                         "neighbour_share": sh, "enrichment": sh / float(lib[so]), "n_query": int(len(B))})
    print("done", s, flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / root / "provenance_enrichment.csv", index=False)
print(out.sort_values(["scenario", "space", "neighbour_share"], ascending=[True, True, False])
      .round(3).to_string(index=False))
