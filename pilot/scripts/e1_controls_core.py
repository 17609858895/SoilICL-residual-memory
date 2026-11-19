"""Training-only spectral transforms and paired metrics for E1 strong controls.

PLS1 uses the same centred, unscaled NIPALS construction as 01_pilot_mir_soc.py,
without importing its optional GP/LightGBM dependencies or executing that runner.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def snv_copy(x):
    x = np.array(x, dtype=np.float32, copy=True)
    sd = x.std(axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(sd <= 1e-12):
        raise ValueError("Non-finite or constant spectrum; do not silently drop an original fold row.")
    x -= x.mean(axis=1, keepdims=True)
    x /= sd
    return x


def fit_pls1(x, y, max_components):
    xm, ym = x.mean(axis=0), float(y.mean())
    xk = np.array(x-xm, dtype=np.float32, copy=True)
    yk = np.array(y-ym, dtype=np.float64, copy=True)
    maximum = min(int(max_components), x.shape[1], len(x)-1)
    W = np.zeros((x.shape[1], maximum))
    P = np.zeros_like(W)
    q = np.zeros(maximum)
    actual = 0
    for a in range(maximum):
        w = xk.T @ yk
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        w /= norm
        t = xk @ w
        tt = float(t @ t)
        if tt <= 1e-12:
            break
        pa = xk.T @ t / tt
        qa = float(yk @ t / tt)
        xk -= np.outer(t, pa).astype(xk.dtype)
        yk -= qa*t
        W[:, a], P[:, a], q[a] = w, pa, qa
        actual = a+1
    if actual < 1:
        raise ValueError("Training PLS has zero usable components")
    W, P, q = W[:, :actual], P[:, :actual], q[:actual]
    rotations = {}
    for a in range(1, actual+1):
        rotations[a] = W[:, :a] @ np.linalg.inv(P[:, :a].T @ W[:, :a])
    return {"x_mean": xm, "y_mean": ym, "W": W, "P": P, "q": q,
            "actual_components": actual, "rotations": rotations}


def scores(x, basis, components):
    if components not in basis["rotations"]:
        raise ValueError(f"Unavailable PLS rank: {components}")
    return np.asarray((x-basis["x_mean"]) @ basis["rotations"][components], dtype=np.float32)


def standardize_scores(train_scores, query_scores):
    sd = train_scores.std(axis=0)
    if np.any(sd <= 1e-10):
        raise ValueError("Degenerate training score component")
    # Centred PLS scores already have zero training mean; follow the original T/sd rule.
    return np.asarray(train_scores/sd, dtype=np.float32), np.asarray(query_scores/sd, dtype=np.float32), sd


def memory_correction(z_train, residuals, z_query, k=20, batch_size=512):
    if len(z_train) == 0:
        return np.zeros(len(z_query))
    index = NearestNeighbors(n_neighbors=min(k,len(z_train))).fit(z_train)
    corrections = np.empty(len(z_query))
    for start in range(0,len(z_query),batch_size):
        stop = min(start+batch_size,len(z_query))
        d, i = index.kneighbors(z_query[start:stop])
        w = 1.0/(d+1e-6)
        corrections[start:stop] = (w*residuals[i]).sum(axis=1)/w.sum(axis=1)
    return corrections


def metrics(y, prediction):
    y, p = np.asarray(y,dtype=float), np.asarray(prediction,dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid prediction/target shape or non-finite values")
    def one(truth,estimate):
        error=estimate-truth
        sst=np.sum((truth-truth.mean())**2)
        return {"RMSE":float(np.sqrt(np.mean(error**2))),"MAE":float(np.mean(np.abs(error))),
                "bias":float(error.mean()),"R2":float(1-np.sum(error**2)/sst) if sst>0 else None}
    return {"log1p":one(y,p),"native_wt_pct":one(np.expm1(y),np.expm1(p))}


def paired_gain_ci(y, method, reference, groups, reps=2000, seed=20261011):
    _, inv = np.unique(groups,return_inverse=True)
    n_groups = int(inv.max()+1)
    counts = np.bincount(inv,minlength=n_groups)
    sm = np.bincount(inv,weights=(method-y)**2,minlength=n_groups)
    sr = np.bincount(inv,weights=(reference-y)**2,minlength=n_groups)
    gain = 100*(1-np.sqrt(sm.sum()/sr.sum())) if sr.sum()>0 else float("nan")
    if n_groups<5:
        return {"gain_pct":float(gain),"ci_low":None,"ci_high":None,"groups":n_groups,"scope":"descriptive"}
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(reps):
        w=np.bincount(rng.integers(n_groups,size=n_groups),minlength=n_groups)
        denominator=float(w@sr)
        if denominator>0:
            draws.append(100*(1-np.sqrt((w@sm)/denominator)))
    lo,hi=np.quantile(draws,[.025,.975])
    return {"gain_pct":float(gain),"ci_low":float(lo),"ci_high":float(hi),"groups":n_groups,"scope":"paired group bootstrap; marginal 95% CI, no simultaneous significance claim"}
