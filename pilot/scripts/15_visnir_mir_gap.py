"""Exploratory, post hoc: on soils that have both a VisNIR and a MIR spectrum, how much of the
VisNIR-to-MIR accuracy gap does the frozen SoilICL-lite recipe close? No model is trained.

Uses existing out-of-fold spatial-block (P1) predictions:
  MIR    results/summary_stage2/all_predictions_log1p.parquet  (fold 1 = stage-2 development fold, reported separately)
  VisNIR results/V_visnir/summary/P1_all_predictions_log1p.parquet (all five folds confirmatory)
Join key: (source, id.layer_uuid_txt). Same SOC label for both rows (checked).
gap closure = (RMSE_VisNIR_Cubist - RMSE_VisNIR_SoilICL) / (RMSE_VisNIR_Cubist - RMSE_MIR_Cubist)
CI: 2,000-draw bootstrap over VisNIR spatial groups.
Writes results/V_visnir/summary/visnir_mir_gap_closure.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mm = pd.read_parquet(ROOT / "data/mir_soc_meta.parquet")[["source", "id.layer_uuid_txt"]]
vm = pd.read_parquet(ROOT / "data/visnir_soc_meta.parquet")[["source", "id.layer_uuid_txt"]]
M = pd.read_parquet(ROOT / "results/summary_stage2/all_predictions_log1p.parquet")
V = pd.read_parquet(ROOT / "results/V_visnir/summary/P1_all_predictions_log1p.parquet")
M = M.drop(columns="source").join(mm, on="row")[["source", "id.layer_uuid_txt", "y_true", "fold", "M0_OSSL_Cubist", "S2_primary_SoilICL_lite"]]
M = M.rename(columns={"y_true": "y_mir", "fold": "fold_mir", "M0_OSSL_Cubist": "MIR_Cubist", "S2_primary_SoilICL_lite": "MIR_SoilICL"})
V = V.drop(columns="source").join(vm, on="row")[["source", "id.layer_uuid_txt", "y_true", "group", "OSSL_Cubist", "SoilICL_lite"]]
V = V.rename(columns={"y_true": "y", "OSSL_Cubist": "VIS_Cubist", "SoilICL_lite": "VIS_SoilICL"})
J = V.merge(M, on=["source", "id.layer_uuid_txt"], how="inner")
assert np.allclose(J.y, J.y_mir, atol=1e-9), "SOC labels differ between modalities"
models = ["VIS_Cubist", "VIS_SoilICL", "MIR_Cubist", "MIR_SoilICL"]


def summarise(d, label):
    ug, inv = np.unique(d.group.to_numpy(), return_inverse=True)
    cnt = np.bincount(inv, minlength=len(ug))
    se = {m: np.bincount(inv, weights=(d[m] - d.y).to_numpy() ** 2, minlength=len(ug)) for m in models}
    def stats(w):
        n = (w * cnt).sum(); r = {m: np.sqrt((w * se[m]).sum() / n) for m in models}
        return r, (r["VIS_Cubist"] - r["VIS_SoilICL"]) / (r["VIS_Cubist"] - r["MIR_Cubist"])
    r, g = stats(np.ones(len(ug)))
    rng = np.random.default_rng(0); G = []
    for _ in range(2000):
        G.append(stats(np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)))[1])
    row = {"subset": label, "n": len(d), "n_groups": len(ug), **{f"RMSE_log1p_{m}": r[m] for m in models},
           "gap_closure_pct": 100 * g, "ci2.5": 100 * np.quantile(G, 0.025), "ci97.5": 100 * np.quantile(G, 0.975)}
    return row


rows = [summarise(J, "all paired soils"), summarise(J[J.fold_mir != 1], "paired, excluding MIR development fold 1")]
for s, d in J.groupby("source"):
    rows.append(summarise(d, f"source {s}"))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/V_visnir/summary/visnir_mir_gap_closure.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
