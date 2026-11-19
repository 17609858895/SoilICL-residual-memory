"""P2 combine: zero-shot transfer metrics and plug-in adaptation curves.

python scripts/09c_p2_combine.py --scenario ICRAF [--smoke]
Writes results/P2/<scenario>/zero_shot.csv, adaptation_runs.csv
"""
import argparse, json, time, sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import SEED, DATASETS, bootstrap_groups
spec = importlib.util.spec_from_file_location("pilot", ROOT / "scripts" / "01_pilot_mir_soc.py")
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
K_MEM, K_MBL = 20, 1000
KS, REPS = [5, 10, 20, 50, 100], 10


def topk(Mz, Qz, k):
    d, i = NearestNeighbors(n_neighbors=k).fit(Mz).kneighbors(Qz)
    return d, i


def merged_correction(d_tr, r_tr_nb, Q, L, r_loc, k=K_MEM):
    """Inverse-distance mean residual of the k nearest among (pre-retrieved training neighbours + local samples)."""
    if L is None or len(L) == 0:
        d, r = d_tr[:, :k], r_tr_nb[:, :k]
    else:
        dl = cdist(Q, L)
        d = np.hstack([d_tr[:, :k], dl]); r = np.hstack([r_tr_nb[:, :k], np.broadcast_to(r_loc, dl.shape)])
        o = np.argsort(d, axis=1)[:, :k]
        d, r = np.take_along_axis(d, o, 1), np.take_along_axis(r, o, 1)
    w = 1.0 / (d + 1e-6)
    return (w * r).sum(1) / w.sum(1)


def local_first_correction(Q, L, r_loc, k_loc=10):
    """Amendment 4: inverse-distance mean residual of the min(k, 10) nearest inserted local samples only."""
    dl = cdist(Q, L)
    m = min(k_loc, L.shape[0])
    o = np.argsort(dl, axis=1)[:, :m]
    d = np.take_along_axis(dl, o, 1); r = r_loc[o]
    w = 1.0 / (d + 1e-6)
    return (w * r).sum(1) / w.sum(1)


def mbl_spike(C, q_idx, loc_idx, d_tr, i_tr, Sq, Sloc):
    """Local WAPLS with library = training neighbours merged with spiked local samples."""
    out = np.empty(len(q_idx))
    for n, qi in enumerate(q_idx):
        if loc_idx is None or len(loc_idx) == 0:
            lib_X, lib_y = C["Ptr"][i_tr[n]], C["ytr"][i_tr[n]]
        else:
            dl = np.sqrt(((Sloc - Sq[n]) ** 2).sum(1))
            d = np.concatenate([d_tr[n], dl]); o = np.argsort(d)[:K_MBL]
            Xcat = np.vstack([C["Ptr"][i_tr[n]], C["Ptg"][loc_idx]])
            ycat = np.concatenate([C["ytr"][i_tr[n]], C["ytg"][loc_idx]])
            lib_X, lib_y = Xcat[o], ycat[o]
        out[n] = pilot.wapls_predict(lib_X.astype(np.float64), lib_y, C["Ptg"][qi].astype(np.float64))
    return out


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="mir", choices=list(DATASETS))
    args = ap.parse_args()
    _, mfile, _, root = DATASETS[args.data]
    base = ROOT / "results" / (root + "_smoke" if args.smoke else root) / args.scenario
    C = dict(np.load(base / "cubist_side.npz"))
    N = dict(np.load(base / "cnn_side.npz"))
    assert (C["tg"] == N["tg"]).all() and (C["tr"] == N["tr"]).all()
    meta = pd.read_parquet(ROOT / "data" / mfile)
    tg = C["tg"]; pos = {r: j for j, r in enumerate(tg)}
    ytg = C["ytg"]

    # ---------- zero-shot on the whole held-out source ----------
    d_pls, i_pls = topk(C["Ztr"], C["Ztg"], K_MEM)
    d_cnn, i_cnn = topk(N["z_tr"], N["z_tg"], K_MEM)
    corr_pls = merged_correction(d_pls, C["r_tr"][i_pls], C["Ztg"], None, None)
    corr_cnn = merged_correction(d_cnn, N["r_tr"][i_cnn], N["z_tg"], None, None)
    zs = pd.DataFrame({"row": tg, "y_true": ytg, "OSSL_Cubist": C["cub_tg"], "MBL": C["mbl_tg"],
                       "Cubist+memory": C["cub_tg"] + corr_pls, "CNN": N["cnn_tg"],
                       "CNN+memory": N["cnn_tg"] + corr_cnn})
    zs["SoilICL_lite"] = (zs["Cubist+memory"] + zs["CNN+memory"]) / 2
    zs["group"] = bootstrap_groups(meta, tg, args.scenario, args.data)
    zs["source"] = meta["source"].values[tg]
    zs.to_parquet(base / "zero_shot_predictions.parquet", index=False)
    models = ["OSSL_Cubist", "MBL", "Cubist+memory", "CNN", "CNN+memory", "SoilICL_lite"]
    ug, inv = np.unique(zs.group.to_numpy(), return_inverse=True); cnt = np.bincount(inv, minlength=len(ug))
    se = {m: np.bincount(inv, weights=(zs[m] - zs.y_true).to_numpy() ** 2, minlength=len(ug)) for m in models}
    rng = np.random.default_rng(0); B = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)); n = (w * cnt).sum()
        r0 = np.sqrt((w * se["OSSL_Cubist"]).sum() / n)
        B.append({m: 100 * (np.sqrt((w * se[m]).sum() / n) / r0 - 1) for m in models})
    B = pd.DataFrame(B)
    y = zs.y_true.to_numpy(); r0 = rmse(zs.OSSL_Cubist.to_numpy(), y)
    zrows = []
    for m in models:
        p = zs[m].to_numpy(); e = p - y
        zrows.append({"scenario": args.scenario, "model": m, "n": len(y), "n_groups": len(ug), "RMSE_log1p": rmse(p, y),
                      "bias_log1p": float(e.mean()), "R2_log1p": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                      "RMSE_wt%": rmse(np.expm1(p), np.expm1(y)), "rel_RMSE_pct": 100 * (rmse(p, y) / r0 - 1),
                      "ci2.5": float(B[m].quantile(0.025)), "ci97.5": float(B[m].quantile(0.975))})
    pd.DataFrame(zrows).to_csv(base / "zero_shot.csv", index=False)
    print(pd.DataFrame(zrows)[["model", "RMSE_log1p", "rel_RMSE_pct", "ci2.5", "ci97.5", "bias_log1p"]].round(4).to_string(index=False), flush=True)

    # ---------- plug-in adaptation ----------
    if len(C["pool"]) == 0:
        print("no adaptation for", args.scenario); return
    pool, test = C["pool"], C["test"]
    rng = np.random.default_rng(SEED)
    if len(test) > 3000:
        test = np.sort(rng.choice(test, 3000, replace=False))
    tp = np.array([pos[r] for r in test]); pp = np.array([pos[r] for r in pool])
    yt = ytg[tp]
    dP, iP = d_pls[tp], i_pls[tp]; dN, iN = d_cnn[tp], i_cnn[tp]
    Ssc = C["Ptg"][:, :int(C["keep"])] / C["psd"]
    d_mbl, i_mbl = topk(C["Ptr"][:, :int(C["keep"])] / C["psd"], Ssc[tp], K_MBL)
    cub_t, cnn_t = C["cub_tg"][tp], N["cnn_tg"][tp]
    rows = []
    t0 = time.time()
    for k in [0] + [k for k in KS if k <= len(pool)]:
        for rep in range(1 if k == 0 else REPS):
            loc = np.array([], int) if k == 0 else rng.choice(pp, k, replace=False)
            r_loc_cub = ytg[loc] - C["cub_tg"][loc]; r_loc_cnn = ytg[loc] - N["cnn_tg"][loc]
            cub_mem = cub_t + merged_correction(dP, C["r_tr"][iP], C["Ztg"][tp], C["Ztg"][loc] if k else None, r_loc_cub)
            cnn_mem = cnn_t + merged_correction(dN, N["r_tr"][iN], N["z_tg"][tp], N["z_tg"][loc] if k else None, r_loc_cnn)
            if k:
                cub_lf = cub_t + local_first_correction(C["Ztg"][tp], C["Ztg"][loc], r_loc_cub)
                cnn_lf = cnn_t + local_first_correction(N["z_tg"][tp], N["z_tg"][loc], r_loc_cnn)
            else:
                cub_lf, cnn_lf = cub_mem, cnn_mem
            preds = {"OSSL_Cubist_zero_shot": cub_t,
                     "Cubist+bias_correction": cub_t + (r_loc_cub.mean() if k else 0.0),
                     "MBL+spiking": mbl_spike(C, tp, loc if k else None, d_mbl, i_mbl, Ssc[tp], Ssc[loc] if k else None),
                     "Cubist+memory_insertion": cub_mem, "CNN+memory_insertion": cnn_mem,
                     "SoilICL_lite_insertion": (cub_mem + cnn_mem) / 2,
                     "SoilICL_lite_local_first": (cub_lf + cnn_lf) / 2}
            for m, p in preds.items():
                rows.append({"scenario": args.scenario, "k": k, "rep": rep, "method": m, "n_local_test": len(tp),
                             "RMSE_log1p": rmse(p, yt), "bias_log1p": float(np.mean(p - yt))})
        print("k", k, "done", round(time.time() - t0), "s", flush=True)
    ad = pd.DataFrame(rows); ad.to_csv(base / "adaptation_runs.csv", index=False)
    base0 = ad[(ad.k == 0) & (ad.method == "OSSL_Cubist_zero_shot")].RMSE_log1p.iloc[0]
    s = ad.groupby(["k", "method"]).RMSE_log1p.agg(["mean", "std"]).reset_index()
    s["rel_vs_zero_shot_cubist_pct"] = 100 * (s["mean"] / base0 - 1)
    s.to_csv(base / "adaptation_summary.csv", index=False)
    print(s.pivot(index="k", columns="method", values="rel_vs_zero_shot_cubist_pct").round(1).to_string())


if __name__ == "__main__":
    main()
