"""Combine stage-1 (pre-registered) and stage-2 (fixed after fold-1 development) predictions.

Reports confirmation folds (0, 2, 3, 4) and the development fold (1) separately, plus all five folds.
python scripts/05_summarize_stage2.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "summary_stage2"; OUT.mkdir(parents=True, exist_ok=True)
DEV, CONFIRM = [1], [0, 2, 3, 4]


def metrics(y, p):
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ccc = float(2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2))
    q1, q3 = np.percentile(y, [25, 75])
    return {"n": int(len(y)), "RMSE": rmse, "MAE": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), "CCC": ccc, "RPIQ": float((q3 - q1) / rmse)}


frames = []
for k in range(5):
    s1 = pd.read_parquet(ROOT / "results" / "P1" / f"fold{k}_pred.parquet")
    s2 = pd.read_parquet(ROOT / "results" / "P1_stage2" / f"fold{k}_pred.parquet")
    cn = pd.read_parquet(ROOT / "results" / "P1_stage2b" / f"fold{k}_pred.parquet")
    assert (s1.row.values == s2.row.values).all() and (s1.row.values == cn.row.values).all()
    assert np.max(np.abs(s1.M0_OSSL_Cubist.values - s2.cubist_plain.values)) < 1e-9
    f = s1[["row", "fold", "y_true", "source", "group"]].copy()
    for c in ["M0_OSSL_Cubist", "M1_PLSR", "M5_LightGBM", "M2_MBL", "M3a_RA_localPLS", "M3b_SoilICL_pilot"]:
        f[c] = s1[c]
    f["Cubist_nn9"] = s2["cubist_nn9"]
    f["Cubist+memory"] = s2["rrm_pls_in_k20"]
    f["CNN"] = cn["cnn"]
    f["CNN+memory"] = cn["cnn_rrm_k20"]
    f["S2_primary_SoilICL_lite"] = (s2["rrm_pls_in_k20"] + cn["cnn_rrm_k20"]) / 2
    f["S2_secondary"] = (s2["cubist_nn9"] + cn["cnn_rrm_k20"]) / 2
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df.to_parquet(OUT / "all_predictions_log1p.parquet", index=False)
models = [c for c in df.columns if c not in ["row", "fold", "y_true", "source", "group"]]

rows = []
for setname, folds in [("confirmation_folds_0_2_3_4", CONFIRM), ("development_fold_1", DEV), ("all_five_folds", list(range(5)))]:
    g = df[df.fold.isin(folds)]
    base = None
    for m in models:
        for scale in ["log1p", "wt%"]:
            y, p = g["y_true"].to_numpy(), g[m].to_numpy()
            if scale == "wt%":
                y, p = np.expm1(y), np.expm1(p)
            r = metrics(y, p)
            rows.append({"set": setname, "model": m, "scale": scale, **r})
res = pd.DataFrame(rows)
b = res[res.model == "M0_OSSL_Cubist"].set_index(["set", "scale"])["RMSE"]
res["rel_RMSE_vs_M0_pct"] = [100 * (r.RMSE - b[(r.set, r.scale)]) / b[(r.set, r.scale)] for r in res.itertuples()]
res.to_csv(OUT / "metrics_by_set.csv", index=False)

# spatial-group bootstrap on the confirmation folds (log1p)
g = df[df.fold.isin(CONFIRM)]
ug, inv = np.unique(g["group"].to_numpy(), return_inverse=True)
cnt = np.bincount(inv, minlength=len(ug))
se = {m: np.bincount(inv, weights=(g[m] - g["y_true"]).to_numpy() ** 2, minlength=len(ug)) for m in models}
rng = np.random.default_rng(0); boot = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)); n = (w * cnt).sum()
    r = {m: np.sqrt((w * se[m]).sum() / n) for m in models}
    boot.append({m: 100 * (r[m] - r["M0_OSSL_Cubist"]) / r["M0_OSSL_Cubist"] for m in models})
boot = pd.DataFrame(boot)
ci = pd.DataFrame({"ci2.5": boot.quantile(0.025), "ci97.5": boot.quantile(0.975)})
ci.to_csv(OUT / "confirmation_rel_RMSE_groupboot_ci.csv")

per_fold = df.groupby("fold").apply(lambda h: pd.Series({m: np.sqrt(np.mean((h[m] - h.y_true) ** 2)) for m in models}))
per_fold.to_csv(OUT / "rmse_log1p_per_fold.csv")
per_src = []
for s, h in df[df.fold.isin(CONFIRM)].groupby("source"):
    for m in models:
        per_src.append({"source": s, "model": m, **metrics(h.y_true.to_numpy(), h[m].to_numpy())})
pd.DataFrame(per_src).to_csv(OUT / "confirmation_metrics_per_source_log1p.csv", index=False)

view = res[(res.scale == "log1p")].pivot(index="model", columns="set", values="rel_RMSE_vs_M0_pct").round(2)
view["confirm_ci2.5"] = ci["ci2.5"].round(2); view["confirm_ci97.5"] = ci["ci97.5"].round(2)
print(view.sort_values("confirmation_folds_0_2_3_4").to_string())
print(per_fold.round(4).T.to_string())
