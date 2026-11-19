"""VisNIR confirmation summary (PILOT_PROTOCOL.md amendment 4). Re-runnable as scenarios finish.

P1: pool the five spatial-block folds (no development fold on VisNIR: every recipe was frozen from MIR),
    relative RMSE (log1p) vs OSSL-Cubist with a 2,000-draw bootstrap over spatial groups; per-source CIs.
P2: zero-shot table per held-out scenario. P3: adaptation at k = 20 and the three pre-stated criteria.
Writes results/V_visnir/summary/*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "V_visnir"
OUT = R / "summary"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["OSSL_Cubist", "MBL", "Cubist+memory", "CNN", "CNN+memory", "SoilICL_lite"]


def boot_rel(df, models, n_boot=2000, seed=0):
    """Relative RMSE change vs OSSL_Cubist (%), bootstrap over df.group."""
    ug, inv = np.unique(df.group.to_numpy(), return_inverse=True)
    cnt = np.bincount(inv, minlength=len(ug))
    se = {m: np.bincount(inv, weights=(df[m] - df.y_true).to_numpy() ** 2, minlength=len(ug)) for m in models}
    rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)); n = (w * cnt).sum()
        r0 = np.sqrt((w * se["OSSL_Cubist"]).sum() / n)
        B.append({m: 100 * (np.sqrt((w * se[m]).sum() / n) / r0 - 1) for m in models})
    B = pd.DataFrame(B)
    y = df.y_true.to_numpy(); r0 = np.sqrt(np.mean((df.OSSL_Cubist - y) ** 2))
    rows = []
    for m in models:
        e = df[m].to_numpy() - y; r = np.sqrt(np.mean(e ** 2))
        rows.append({"model": m, "n": len(df), "n_groups": len(ug), "RMSE_log1p": r, "bias_log1p": e.mean(),
                     "RMSE_wt%": np.sqrt(np.mean((np.expm1(df[m]) - np.expm1(y)) ** 2)),
                     "rel_RMSE_pct": 100 * (r / r0 - 1), "ci2.5": B[m].quantile(0.025), "ci97.5": B[m].quantile(0.975)})
    return pd.DataFrame(rows)


# ---------- P1 ----------
folds = [f"fold{k}" for k in range(5)]
done = [f for f in folds if (R / f / "zero_shot_predictions.parquet").exists()]
P = pd.concat([pd.read_parquet(R / f / "zero_shot_predictions.parquet").assign(fold=f) for f in done], ignore_index=True)
p1 = boot_rel(P, MODELS); p1.insert(0, "set", f"P1 pooled ({len(done)} folds)")
p1.to_csv(OUT / "P1_pooled_rel_RMSE_ci.csv", index=False)
perfold = P.groupby("fold").apply(lambda d: pd.Series({m: np.sqrt(np.mean((d[m] - d.y_true) ** 2)) for m in MODELS}),
                                  include_groups=False)
perfold.to_csv(OUT / "P1_rmse_log1p_per_fold.csv")
src_rows = []
for s, d in P.groupby("source"):
    t = boot_rel(d, ["OSSL_Cubist", "SoilICL_lite"]); t = t[t.model == "SoilICL_lite"]; t.insert(0, "source", s)
    src_rows.append(t)
pd.concat(src_rows).to_csv(OUT / "P1_per_source_rel_RMSE_ci.csv", index=False)
P.to_parquet(OUT / "P1_all_predictions_log1p.parquet", index=False)

# ---------- P2 zero-shot ----------
z = []
for s in ["Europe", "ICRAF", "KSSL"]:
    f = R / s / "zero_shot.csv"
    if f.exists():
        z.append(pd.read_csv(f))
Z = pd.concat(z, ignore_index=True) if z else pd.DataFrame()
Z.to_csv(OUT / "P2_zero_shot_all.csv", index=False)

# ---------- P3 adaptation ----------
a = []
for s in ["Europe", "ICRAF", "KSSL"]:
    f = R / s / "adaptation_summary.csv"
    if f.exists():
        a.append(pd.read_csv(f).assign(scenario=s))
A = pd.concat(a, ignore_index=True) if a else pd.DataFrame()
A.to_csv(OUT / "P3_adaptation_all.csv", index=False)

# ---------- criteria ----------
crit = []
sl = p1.set_index("model").loc["SoilICL_lite"]
crit.append({"criterion": "1 P1: SoilICL-lite < OSSL-Cubist, CI excludes 0", "value": f"{sl.rel_RMSE_pct:.1f}% [{sl['ci2.5']:.1f}, {sl['ci97.5']:.1f}]",
             "pass": bool(sl["ci97.5"] < 0) if len(done) == 5 else "pending", "basis": f"{len(done)}/5 folds"})
if len(Z):
    zz = Z[Z.model == "SoilICL_lite"]
    wins = int((zz["ci97.5"] < 0).sum())
    crit.append({"criterion": "2 P2: SoilICL-lite better in >= 2 of 3 scenarios (CI excludes 0)", "value": f"{wins} of {len(zz)} finished",
                 "pass": (wins >= 2) if len(zz) == 3 else ("pass (already >= 2)" if wins >= 2 else "pending"),
                 "basis": ", ".join(zz.scenario)})
if len(A):
    k20 = A[A.k == 20].pivot(index="scenario", columns="method", values="rel_vs_zero_shot_cubist_pct")
    k20.to_csv(OUT / "P3_k20_table.csv")
    lf = k20["SoilICL_lite_local_first"]
    beats = int(((lf < k20["Cubist+bias_correction"]) & (lf < k20["MBL+spiking"])).sum())
    k0 = A[A.k == 0].set_index("scenario")
    crit.append({"criterion": "3a P3: local-first at k=20 >= 10% lower than zero-shot OSSL-Cubist (mean of 3)",
                 "value": f"mean {lf.mean():.1f}% over {len(lf)} scenario(s)", "pass": (lf.mean() <= -10) if len(lf) == 3 else "pending",
                 "basis": ", ".join(lf.index)})
    crit.append({"criterion": "3b P3: local-first beats bias correction and MBL spiking in >= 2 of 3",
                 "value": f"{beats} of {len(lf)}", "pass": (beats >= 2) if len(lf) == 3 else ("pass (already >= 2)" if beats >= 2 else "pending"),
                 "basis": ", ".join(lf.index)})
C = pd.DataFrame(crit); C.to_csv(OUT / "criteria_status.csv", index=False)

pd.set_option("display.width", 200)
print(p1.round(4).to_string(index=False)); print(perfold.round(4).to_string())
print(pd.concat(src_rows)[["source", "n", "n_groups", "rel_RMSE_pct", "ci2.5", "ci97.5"]].round(2).to_string(index=False))
if len(Z):
    print(Z[Z.model.isin(["OSSL_Cubist", "CNN", "SoilICL_lite"])][["scenario", "model", "n", "RMSE_log1p", "bias_log1p", "rel_RMSE_pct", "ci2.5", "ci97.5"]].round(3).to_string(index=False))
if len(A):
    print(k20.round(1).to_string())
print(C.to_string(index=False))
