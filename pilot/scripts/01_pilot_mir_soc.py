"""SoilICL pilot: MIR soil organic carbon, one outer fold per call.

python scripts/01_pilot_mir_soc.py --protocol P1 --fold 0 [--subsample N]

Writes results/<protocol>/fold<k>_pred.parquet and fold<k>_info.json.
See PILOT_PROTOCOL.md for the fixed design.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from cubist import Cubist
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
K_LOCAL = 1000
N_ANCHORS = 1000
N_LV_RETRIEVAL = 20
N_HP_ANCHORS = 20
N_HP_SUBSET = 300
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def snv(X):
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True)
    return X


def pls1_fit(X, y, A):
    """NIPALS PLS1 on centred X (n,p), centred y. Returns W, P, q (A comps)."""
    Xk = X.copy()
    yk = y.astype(np.float64).copy()
    p = X.shape[1]
    W = np.zeros((p, A)); P = np.zeros((p, A)); q = np.zeros(A)
    for a in range(A):
        w = Xk.T @ yk
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            W, P, q = W[:, :a], P[:, :a], q[:a]
            break
        w /= nw
        t = Xk @ w
        tt = t @ t
        pa = Xk.T @ t / tt
        qa = yk @ t / tt
        Xk -= np.outer(t, pa).astype(Xk.dtype)
        yk -= qa * t
        W[:, a], P[:, a], q[a] = w, pa, qa
    return W, P, q


def pls_rotations(W, P):
    """R_a = W_a (P_a' W_a)^-1 for every a, returned as list."""
    return [W[:, :a] @ np.linalg.inv(P[:, :a].T @ W[:, :a]) for a in range(1, W.shape[1] + 1)]


def wapls_predict(Xn, yn, xq, amin=3, amax=15):
    """Local weighted-average PLS (Shenk-style weights) for one query."""
    xm, ym = Xn.mean(0), yn.mean()
    Xc, yc, xqc = Xn - xm, yn - ym, xq - xm
    W, P, q = pls1_fit(Xc, yc, amax)
    A = W.shape[1]
    preds, wts = [], []
    for a in range(amin, A + 1):
        Wa, Pa = W[:, :a], P[:, :a]
        R = Wa @ np.linalg.inv(Pa.T @ Wa)
        b = R @ q[:a]
        tq = xqc @ R
        resid_x = xqc - tq @ Pa.T
        s = np.sqrt(np.mean(resid_x ** 2)) + 1e-12
        g = np.sqrt(np.mean(b ** 2)) + 1e-12
        preds.append(ym + xqc @ b)
        wts.append(1.0 / (s * g))
    wts = np.asarray(wts)
    return float(np.dot(wts, preds) / wts.sum())


def local_wapls(feat_tr, y_tr, feat_te, space_tr, space_te, k):
    nn = NearestNeighbors(n_neighbors=k).fit(space_tr)
    _, idx = nn.kneighbors(space_te)
    out = np.empty(len(feat_te))
    for i in range(len(feat_te)):
        nb = idx[i]
        out[i] = wapls_predict(feat_tr[nb], y_tr[nb], feat_te[i])
    return out


def make_folds(meta, protocol):
    n = len(meta)
    fold = np.full(n, -1)
    if protocol == "P1":
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = gkf.split(np.zeros(n), groups=meta["group_block"].values)
    else:
        splits = KFold(n_splits=10, shuffle=True, random_state=SEED).split(np.zeros(n))
    for k, (_, te) in enumerate(splits):
        fold[te] = k
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P1", choices=["P0", "P1"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--subsample", type=int, default=0, help="smoke test only")
    args = ap.parse_args()

    out_dir = ROOT / "results" / (args.protocol + ("_smoke" if args.subsample else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"protocol": args.protocol, "fold": args.fold, "timing_s": {}}
    T0 = time.time()

    X = np.load(ROOT / "data" / "mir_soc_X.npy")
    meta = pd.read_parquet(ROOT / "data" / "mir_soc_meta.parquet")
    if args.subsample:
        sel = np.random.default_rng(0).choice(len(meta), args.subsample, replace=False)
        sel.sort()
        X, meta = X[sel], meta.iloc[sel].reset_index(drop=True)
    X = snv(X)
    y = np.log1p(meta["soc"].to_numpy(np.float64))

    fold = make_folds(meta, args.protocol)
    if args.fold == 0 and not args.subsample:
        pd.DataFrame({"row": np.arange(len(meta)), "fold": fold}).to_parquet(out_dir / "fold_assignment.parquet")
    te = np.where(fold == args.fold)[0]
    tr = np.where(fold != args.fold)[0]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    del X
    info["n_train"], info["n_test"] = int(len(tr)), int(len(te))

    # inner group hold-out (10%) inside the training fold, used only for model selection
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    itr, iva = next(gss.split(np.zeros(len(tr)), groups=meta["group_block"].values[tr]))

    pred = pd.DataFrame({"row": te, "fold": args.fold, "y_true": yte,
                         "source": meta["source"].values[te], "group": meta["group_block"].values[te]})

    # ---------------- PCA(120) ----------------
    t = time.time()
    pca = PCA(n_components=120, svd_solver="randomized", random_state=SEED).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    info["pca_explained_120"] = float(pca.explained_variance_ratio_.sum())
    info["timing_s"]["pca"] = time.time() - t

    # ---------------- M0 OSSL-Cubist ----------------
    t = time.time()
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr), min(2000, len(tr)), replace=False)
    cv_rmse = {}
    for c in [1, 5, 10, 15, 20]:
        errs = []
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(sub):
            m = Cubist(n_rules=100, n_committees=c, neighbors=None, random_state=SEED).fit(Ptr[sub[a]], ytr[sub[a]])
            errs.append(np.mean((m.predict(Ptr[sub[b]]) - ytr[sub[b]]) ** 2))
        cv_rmse[c] = float(np.sqrt(np.mean(errs)))
    best_c = min(cv_rmse, key=cv_rmse.get)
    m0 = Cubist(n_rules=100, n_committees=best_c, neighbors=None, random_state=SEED).fit(Ptr, ytr)
    pred["M0_OSSL_Cubist"] = m0.predict(Pte)
    info["cubist_cv_rmse"], info["cubist_committees"] = cv_rmse, best_c
    info["timing_s"]["M0"] = time.time() - t
    print("M0 done", info["timing_s"]["M0"], flush=True)

    # ---------------- M1 global PLSR + PLS retrieval space ----------------
    t = time.time()
    xm_i, ym_i = Xtr[itr].mean(0), ytr[itr].mean()
    W, P, q = pls1_fit(Xtr[itr] - xm_i, ytr[itr] - ym_i, 30)
    Rs = pls_rotations(W, P)
    Xva_c = Xtr[iva] - xm_i
    inner = [float(np.sqrt(np.mean((ym_i + Xva_c @ (R @ q[:R.shape[1]]) - ytr[iva]) ** 2))) for R in Rs]
    a_star = int(np.argmin(inner) + 1)
    xm, ym = Xtr.mean(0), ytr.mean()
    Xtr_c = Xtr - xm
    Xte_c = Xte - xm
    W, P, q = pls1_fit(Xtr_c, ytr - ym, 30)
    Rs = pls_rotations(W, P)
    pred["M1_PLSR"] = ym + Xte_c @ (Rs[a_star - 1] @ q[:a_star])
    R20 = Rs[N_LV_RETRIEVAL - 1]
    Ttr, Tte = Xtr_c @ R20, Xte_c @ R20
    tsd = Ttr.std(0)
    Ztr, Zte = Ttr / tsd, Tte / tsd
    del Xtr_c, Xte_c
    info["pls_inner_rmse"], info["pls_ncomp"] = inner, a_star
    info["timing_s"]["M1"] = time.time() - t
    print("M1 done", info["timing_s"]["M1"], flush=True)

    # ---------------- M5 LightGBM on PCA scores ----------------
    t = time.time()
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=1,
                  seed=SEED, verbose=-1)
    dtr = lgb.Dataset(Ptr[itr], ytr[itr]); dva = lgb.Dataset(Ptr[iva], ytr[iva])
    b = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best_it = int(b.best_iteration or 3000)
    m5 = lgb.train(params, lgb.Dataset(Ptr, ytr), best_it)
    pred["M5_LightGBM"] = m5.predict(Pte)
    info["lgb_best_iter"] = best_it
    info["timing_s"]["M5"] = time.time() - t
    print("M5 done", info["timing_s"]["M5"], flush=True)

    # ---------------- M2 MBL (PCA-Mahalanobis + local WAPLS) ----------------
    t = time.time()
    keep = max(5, int((pca.explained_variance_ratio_ >= 0.01).sum()))
    psd = Ptr[:, :keep].std(0)
    pred["M2_MBL"] = local_wapls(Ptr, ytr, Pte, Ptr[:, :keep] / psd, Pte[:, :keep] / psd, K_LOCAL)
    info["mbl_n_pc_distance"] = keep
    info["timing_s"]["M2"] = time.time() - t
    print("M2 done", info["timing_s"]["M2"], flush=True)

    # ---------------- M3a RA-localPLS (PLS-space retrieval + local WAPLS) ----------------
    t = time.time()
    pred["M3a_RA_localPLS"] = local_wapls(Ptr, ytr, Pte, Ztr, Zte, K_LOCAL)
    info["timing_s"]["M3a"] = time.time() - t
    print("M3a done", info["timing_s"]["M3a"], flush=True)

    # ---------------- M3b SoilICL-pilot (anchors + fixed-prior GP) ----------------
    t = time.time()
    km = MiniBatchKMeans(n_clusters=min(N_ANCHORS, len(tr) // 20), random_state=SEED, n_init=3,
                         batch_size=4096).fit(Ztr)
    centres = km.cluster_centers_
    nn = NearestNeighbors(n_neighbors=K_LOCAL).fit(Ztr)
    _, ctx_idx = nn.kneighbors(centres)
    q_anchor = NearestNeighbors(n_neighbors=1).fit(centres).kneighbors(Zte, return_distance=False)[:, 0]

    d = Ztr.shape[1]
    kernel0 = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.full(d, 3.0), (1e-1, 1e3))
               + ConstantKernel(0.1, (1e-4, 1e2)) * DotProduct(1.0, (1e-3, 1e2))
               + WhiteKernel(0.1, (1e-5, 1.0)))
    hp_rng = np.random.default_rng(SEED)
    hp_anchor = hp_rng.choice(len(centres), N_HP_ANCHORS, replace=False)
    thetas = []
    for a in hp_anchor:
        # hyperparameters learned on a random 300-point subset of the anchor context
        # (gradient cost of ARD kernels scales with n^2 x n_params); then frozen
        idx = hp_rng.choice(ctx_idx[a], N_HP_SUBSET, replace=False)
        g = GaussianProcessRegressor(kernel0, normalize_y=True, n_restarts_optimizer=0, random_state=SEED)
        g.fit(Ztr[idx], ytr[idx])
        thetas.append(g.kernel_.theta)
    theta = np.median(np.vstack(thetas), axis=0)
    kernel_fixed = kernel0.clone_with_theta(theta)
    info["gp_kernel_fixed"] = str(kernel_fixed)
    t_hp = time.time() - t

    mu = np.empty(len(te)); sd = np.empty(len(te))
    for a in np.unique(q_anchor):
        qi = np.where(q_anchor == a)[0]
        idx = ctx_idx[a]
        g = GaussianProcessRegressor(kernel_fixed, optimizer=None, normalize_y=True).fit(Ztr[idx], ytr[idx])
        m, s = g.predict(Zte[qi], return_std=True)
        mu[qi], sd[qi] = m, s
    pred["M3b_SoilICL_pilot"] = mu
    pred["M3b_sd"] = sd
    info["n_anchors"] = int(len(centres)); info["n_anchors_used"] = int(len(np.unique(q_anchor)))
    info["timing_s"]["M3b_hyperparam"] = t_hp
    info["timing_s"]["M3b_total"] = time.time() - t
    print("M3b done", info["timing_s"]["M3b_total"], flush=True)

    info["timing_s"]["total"] = time.time() - T0
    pred.to_parquet(out_dir / f"fold{args.fold}_pred.parquet", index=False)
    with open(out_dir / f"fold{args.fold}_info.json", "w") as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps({k: info[k] for k in ["n_train", "n_test", "cubist_committees", "pls_ncomp"]}))


if __name__ == "__main__":
    main()
