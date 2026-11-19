"""P2 leave-source-out, rule-based side: OSSL-Cubist, MBL, and the arrays needed for residual memory.

python scripts/09a_p2_cubist.py --scenario ICRAF [--subsample_train 5000]
Writes results/P2/<scenario>/cubist_side.npz and cubist_info.json
"""
import argparse, json, time, warnings
from pathlib import Path
import importlib.util, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from cubist import Cubist

warnings.filterwarnings("ignore", message="X does not have valid feature names")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import get_split, SEED, DATASETS
spec = importlib.util.spec_from_file_location("pilot", ROOT / "scripts" / "01_pilot_mir_soc.py")
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--subsample_train", type=int, default=0)
    ap.add_argument("--data", default="mir", choices=list(DATASETS))
    args = ap.parse_args()
    xfile, mfile, _, root = DATASETS[args.data]
    out = ROOT / "results" / (root + "_smoke" if args.subsample_train else root) / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); info = {"scenario": args.scenario, "data": args.data, "timing_s": {}}

    X = np.load(ROOT / "data" / xfile)
    meta = pd.read_parquet(ROOT / "data" / mfile)
    X = pilot.snv(X)
    y = np.log1p(meta["soc"].to_numpy(np.float64))
    tr, tg, pool, test = get_split(meta, args.scenario, args.data)
    if args.subsample_train:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.subsample_train, replace=False))
    Xtr, Xtg, ytr, ytg = X[tr], X[tg], y[tr], y[tg]
    del X
    info.update({"n_train": int(len(tr)), "n_target": int(len(tg)),
                 "n_pool": None if pool is None else int(len(pool)), "n_local_test": None if test is None else int(len(test))})

    t = time.time()
    pca = PCA(n_components=120, svd_solver="randomized", random_state=SEED).fit(Xtr)
    Ptr, Ptg = pca.transform(Xtr).astype(np.float32), pca.transform(Xtg).astype(np.float32)
    keep = max(5, int((pca.explained_variance_ratio_ >= 0.01).sum()))
    info["timing_s"]["pca"] = time.time() - t

    t = time.time()
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr), min(2000, len(tr)), replace=False)
    cv = {}
    for c in [1, 5, 10, 15, 20]:
        errs = []
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(sub):
            m = Cubist(n_rules=100, n_committees=c, neighbors=None, random_state=SEED).fit(Ptr[sub[a]], ytr[sub[a]])
            errs.append(np.mean((m.predict(Ptr[sub[b]]) - ytr[sub[b]]) ** 2))
        cv[c] = float(np.sqrt(np.mean(errs)))
    best_c = min(cv, key=cv.get)
    cub = Cubist(n_rules=100, n_committees=best_c, neighbors=None, random_state=SEED).fit(Ptr, ytr)
    cub_tg = cub.predict(Ptg)
    r_tr = ytr - cub.predict(Ptr)
    info.update({"cubist_cv_rmse": cv, "cubist_committees": best_c})
    info["timing_s"]["cubist"] = time.time() - t
    print("cubist done", info["timing_s"]["cubist"], flush=True)

    t = time.time()
    xm, ym = Xtr.mean(0), ytr.mean()
    W, P, q = pilot.pls1_fit(Xtr - xm, ytr - ym, 20)
    R20 = pilot.pls_rotations(W, P)[-1]
    Ttr, Ttg = (Xtr - xm) @ R20, (Xtg - xm) @ R20
    sd = Ttr.std(0)
    Ztr, Ztg = (Ttr / sd).astype(np.float32), (Ttg / sd).astype(np.float32)
    info["timing_s"]["pls"] = time.time() - t

    t = time.time()
    psd = Ptr[:, :keep].std(0)
    mbl_tg = pilot.local_wapls(Ptr, ytr, Ptg, Ptr[:, :keep] / psd, Ptg[:, :keep] / psd, 1000)
    info["timing_s"]["mbl"] = time.time() - t
    print("mbl done", info["timing_s"]["mbl"], flush=True)

    np.savez_compressed(out / "cubist_side.npz", tr=tr, tg=tg, pool=pool if pool is not None else np.array([], int),
                        test=test if test is not None else np.array([], int), ytr=ytr, ytg=ytg,
                        cub_tg=cub_tg, r_tr=r_tr, Ztr=Ztr, Ztg=Ztg, Ptr=Ptr, Ptg=Ptg, psd=psd, keep=keep, mbl_tg=mbl_tg)
    info["timing_s"]["total"] = time.time() - T0
    json.dump(info, open(out / "cubist_info.json", "w"), indent=1, default=float)
    print(json.dumps({k: info[k] for k in ["n_train", "n_target", "n_pool", "n_local_test", "cubist_committees"]}))


if __name__ == "__main__":
    main()
