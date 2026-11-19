"""Stage 2 (exploratory, development on outer fold 1 only, confirmation on folds 0,2,3,4).

Global Cubist backbone (identical to M0) + retrieval-augmented residual memory:
    y_hat(x) = Cubist(x) + sum_k w_k r_k / sum_k w_k
where r_k are residuals of the k training samples retrieved for x in the
standardised 20-LV PLS space (same retrieval space as M3a/M3b), and w_k = 1/(d_k + eps).
Also records Quinlan's composite (Cubist instance correction, nn = 1, 5, 9), which the
original OSSL models did not use.

python scripts/03_stage2_hybrid.py --fold 1 [--oof]
"""
import argparse, json, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from cubist import Cubist

warnings.filterwarnings("ignore", message="X does not have valid feature names")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
import importlib.util
spec = importlib.util.spec_from_file_location("pilot", ROOT / "scripts" / "01_pilot_mir_soc.py")
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)


def set_header(model_str, insts, nn=None, maxd=None):
    lines = model_str.split("\n")
    h = lines[1]
    h = re.sub(r' insts="\d+"( nn="\d+")?( maxd="[^"]+")?', "", h)
    add = f' insts="{insts}"' + (f' nn="{nn}" maxd="{maxd}"' if insts == 1 else "")
    h = h.replace(' ceiling=', add + ' ceiling=', 1)
    lines[1] = h
    return "\n".join(lines)


def cubist_predict_with(model, X, insts, nn=None, maxd=None):
    orig = model.model_
    model.model_ = set_header(orig, insts, nn, maxd)
    try:
        return model.predict(X)
    finally:
        model.model_ = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--oof", action="store_true", help="also compute 2-fold out-of-fold residuals")
    ap.add_argument("--nn", type=int, nargs="+", default=[1, 5, 9], help="Cubist instance-correction neighbours to evaluate")
    args = ap.parse_args()
    out = ROOT / "results" / "P1_stage2"; out.mkdir(parents=True, exist_ok=True)
    info = {"fold": args.fold, "timing_s": {}}
    T0 = time.time()

    X = np.load(ROOT / "data" / "mir_soc_X.npy")
    meta = pd.read_parquet(ROOT / "data" / "mir_soc_meta.parquet")
    X = pilot.snv(X)
    y = np.log1p(meta["soc"].to_numpy(np.float64))
    fold = pd.read_parquet(ROOT / "results" / "P1" / "fold_assignment.parquet")["fold"].to_numpy()
    te = np.where(fold == args.fold)[0]; tr = np.where(fold != args.fold)[0]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    del X
    best_c = json.load(open(ROOT / "results" / "P1" / f"fold{args.fold}_info.json"))["cubist_committees"]

    pca = PCA(n_components=120, svd_solver="randomized", random_state=SEED).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)

    # PLS-20 retrieval space (identical construction to stage 1)
    xm, ym = Xtr.mean(0), ytr.mean()
    W, P, q = pilot.pls1_fit(Xtr - xm, ytr - ym, 20)
    R20 = pilot.pls_rotations(W, P)[-1]
    Ttr, Tte = (Xtr - xm) @ R20, (Xte - xm) @ R20
    sd = Ttr.std(0); Ztr, Zte = Ttr / sd, Tte / sd
    del Xtr, Xte
    info["timing_s"]["features"] = time.time() - T0

    t = time.time()
    cub = Cubist(n_rules=100, n_committees=best_c, neighbors=5, random_state=SEED).fit(Ptr, ytr)
    maxd = re.search(r'maxd="([^"]+)"', cub.model_.split("\n")[1]).group(1)
    info["timing_s"]["cubist_fit"] = time.time() - t

    pred = pd.DataFrame({"row": te, "fold": args.fold, "y_true": yte,
                         "source": meta["source"].values[te], "group": meta["group_block"].values[te]})
    t = time.time()
    pred["cubist_plain"] = cubist_predict_with(cub, Pte, 0)
    for nn in args.nn:
        pred[f"cubist_nn{nn}"] = cubist_predict_with(cub, Pte, 1, nn, maxd)
    info["timing_s"]["cubist_predict_test"] = time.time() - t

    # sanity: plain predictions must reproduce stage-1 M0
    m0 = pd.read_parquet(ROOT / "results" / "P1" / f"fold{args.fold}_pred.parquet")["M0_OSSL_Cubist"].to_numpy()
    info["max_abs_diff_vs_stage1_M0"] = float(np.max(np.abs(m0 - pred["cubist_plain"].to_numpy())))

    t = time.time()
    r_in = ytr - cubist_predict_with(cub, Ptr, 0)
    info["timing_s"]["cubist_predict_train"] = time.time() - t
    resid = {"in": r_in}
    if args.oof:
        t = time.time()
        r_oof = np.empty(len(tr))
        for a, b in GroupKFold(2, shuffle=True, random_state=SEED).split(Ptr, groups=meta["group_block"].values[tr]):
            mh = Cubist(n_rules=100, n_committees=best_c, neighbors=None, random_state=SEED).fit(Ptr[a], ytr[a])
            r_oof[b] = ytr[b] - mh.predict(Ptr[b])
        resid["oof"] = r_oof
        info["timing_s"]["oof"] = time.time() - t

    # retrieval-augmented residual memory in PLS space (and PCA space for comparison)
    keep = max(5, int((pca.explained_variance_ratio_ >= 0.01).sum()))
    psd = Ptr[:, :keep].std(0)
    spaces = {"pls": (Ztr, Zte), "pca": (Ptr[:, :keep] / psd, Pte[:, :keep] / psd)}
    for sname, (Str, Ste) in spaces.items():
        dist, idx = NearestNeighbors(n_neighbors=50).fit(Str).kneighbors(Ste)
        for rname, r in resid.items():
            for k in (5, 10, 20, 50):
                w = 1.0 / (dist[:, :k] + 1e-6)
                corr = (w * r[idx[:, :k]]).sum(1) / w.sum(1)
                pred[f"rrm_{sname}_{rname}_k{k}"] = pred["cubist_plain"] + corr
    info["timing_s"]["total"] = time.time() - T0
    pred.to_parquet(out / f"fold{args.fold}_pred.parquet", index=False)
    json.dump(info, open(out / f"fold{args.fold}_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
