"""Exploratory (post hoc, no retraining): does each backbone need its own retrieval view?

VisNIR P1 folds 0-4. For each backbone (Cubist, CNN) the residual memory is retrieved either in its own view
(Cubist: standardized PLS-20; CNN: 128-d embedding) or in the other backbone's view (swap).
Also reports the correlation between the two backbones' errors.
Writes results/V_visnir/summary/view_ablation.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import bootstrap_groups
meta = pd.read_parquet(ROOT / "data/visnir_soc_meta.parquet")
K = 20


def corr(Mtr, Mtg, r_tr):
    d, i = NearestNeighbors(n_neighbors=K).fit(Mtr).kneighbors(Mtg)
    w = 1.0 / (d + 1e-6)
    return (w * r_tr[i]).sum(1) / w.sum(1)


rows = []
for f in range(5):
    base = ROOT / "results/V_visnir" / f"fold{f}"
    C, N = np.load(base / "cubist_side.npz"), np.load(base / "cnn_side.npz")
    assert (C["tg"] == N["tg"]).all()
    y = C["ytg"]
    cub, cnn = C["cub_tg"], N["cnn_tg"]
    out = pd.DataFrame({"y": y, "Cubist": cub, "CNN": cnn,
                        "Cubist+mem(own PLS view)": cub + corr(C["Ztr"], C["Ztg"], C["r_tr"]),
                        "Cubist+mem(CNN view)": cub + corr(N["z_tr"], N["z_tg"], C["r_tr"]),
                        "CNN+mem(own CNN view)": cnn + corr(N["z_tr"], N["z_tg"], N["r_tr"]),
                        "CNN+mem(PLS view)": cnn + corr(C["Ztr"], C["Ztg"], N["r_tr"])})
    out["group"] = bootstrap_groups(meta, C["tg"], f"fold{f}", "visnir")
    rows.append(out)
    print("fold", f, "done", flush=True)
P = pd.concat(rows, ignore_index=True)
P["Average (no memory)"] = (P.Cubist + P.CNN) / 2
P["Dual view, matched (SoilICL-lite)"] = (P["Cubist+mem(own PLS view)"] + P["CNN+mem(own CNN view)"]) / 2
P["Dual view, swapped"] = (P["Cubist+mem(CNN view)"] + P["CNN+mem(PLS view)"]) / 2
P["Single view: both in PLS view"] = (P["Cubist+mem(own PLS view)"] + P["CNN+mem(PLS view)"]) / 2
P["Single view: both in CNN view"] = (P["Cubist+mem(CNN view)"] + P["CNN+mem(own CNN view)"]) / 2
models = [c for c in P.columns if c not in ("y", "group")]

ug, inv = np.unique(P.group.to_numpy(), return_inverse=True)
cnt = np.bincount(inv, minlength=len(ug))
se = {m: np.bincount(inv, weights=(P[m] - P.y).to_numpy() ** 2, minlength=len(ug)) for m in models}
rng = np.random.default_rng(0); B = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)); n = (w * cnt).sum()
    r0 = np.sqrt((w * se["Cubist"]).sum() / n)
    B.append({m: 100 * (np.sqrt((w * se[m]).sum() / n) / r0 - 1) for m in models})
B = pd.DataFrame(B)
r0 = np.sqrt(np.mean((P.Cubist - P.y) ** 2))
res = pd.DataFrame([{"model": m, "RMSE_log1p": np.sqrt(np.mean((P[m] - P.y) ** 2)),
                     "rel_vs_Cubist_pct": 100 * (np.sqrt(np.mean((P[m] - P.y) ** 2)) / r0 - 1),
                     "ci2.5": B[m].quantile(0.025), "ci97.5": B[m].quantile(0.975)} for m in models])
ec = np.corrcoef(P.Cubist - P.y, P.CNN - P.y)[0, 1]
res.loc[len(res)] = {"model": "error correlation Cubist vs CNN (Pearson r)", "RMSE_log1p": ec,
                     "rel_vs_Cubist_pct": np.nan, "ci2.5": np.nan, "ci97.5": np.nan}
res.to_csv(ROOT / "results/V_visnir/summary/view_ablation.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
