"""P2 leave-source-out, learned-encoder side: same 1D-CNN recipe as scripts/04_cnn_rrm.py.

python scripts/09b_p2_cnn.py --scenario ICRAF [--subsample_train 5000]
Writes results/P2/<scenario>/cnn_side.npz, cnn_state.pt, cnn_info.json
"""
import argparse, json, time, sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import get_split, SEED, DATASETS
spec = importlib.util.spec_from_file_location("cnnmod", ROOT / "scripts" / "04_cnn_rrm.py")
cnnmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cnnmod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--subsample_train", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--data", default="mir", choices=list(DATASETS))
    args = ap.parse_args()
    xfile, mfile, bw, root = DATASETS[args.data]
    torch.manual_seed(SEED); np.random.seed(SEED)
    out = ROOT / "results" / (root + "_smoke" if args.subsample_train else root) / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); info = {"scenario": args.scenario, "data": args.data, "cnn_bin_width": bw}

    X = np.load(ROOT / "data" / xfile)
    meta = pd.read_parquet(ROOT / "data" / mfile)
    X = cnnmod.pilot.snv(X)
    L = (X.shape[1] // bw) * bw
    X = X[:, :L].reshape(len(X), L // bw, bw).mean(2).astype(np.float32)
    y = np.log1p(meta["soc"].to_numpy(np.float64))
    tr, tg, pool, test = get_split(meta, args.scenario, args.data)
    if args.subsample_train:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.subsample_train, replace=False))
    itr, iva = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
                    .split(np.zeros(len(tr)), groups=meta["group_block"].values[tr]))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32); del X
    ym, ys = y[tr].mean(), y[tr].std()
    yz = ((y - ym) / ys).astype(np.float32)

    Xa, ya = torch.from_numpy(Xs[tr[itr]]).unsqueeze(1), torch.from_numpy(yz[tr[itr]])
    Xv, yv = Xs[tr[iva]], yz[tr[iva]]
    model = cnnmod.SpecCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.MSELoss(); g = torch.Generator().manual_seed(SEED)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(args.max_epochs):
        model.train(); t = time.time()
        perm = torch.randperm(len(ya), generator=g)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad(); loss = lossf(model(Xa[b]), ya[b]); loss.backward(); opt.step()
        pv, _ = cnnmod.predict(model, Xv)
        vr = float(np.sqrt(np.mean((pv - yv) ** 2)) * ys)
        sched.step(vr); hist.append(vr)
        print(f"epoch {ep} val_rmse_log1p {vr:.4f} {time.time()-t:.0f}s", flush=True)
        if vr < best - 1e-5:
            best, wait = vr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "cnn_state.pt")
    p_tr, z_tr = cnnmod.predict(model, Xs[tr]); p_tg, z_tg = cnnmod.predict(model, Xs[tg])
    p_tr, p_tg = p_tr * ys + ym, p_tg * ys + ym
    np.savez_compressed(out / "cnn_side.npz", tr=tr, tg=tg, cnn_tg=p_tg, r_tr=y[tr] - p_tr,
                        z_tr=z_tr.astype(np.float32), z_tg=z_tg.astype(np.float32))
    info.update({"epochs_run": len(hist), "best_inner_val_rmse": best, "val_history": hist,
                 "timing_s": time.time() - T0, "n_train": int(len(tr)), "n_target": int(len(tg))})
    json.dump(info, open(out / "cnn_info.json", "w"), indent=1)
    print(json.dumps({k: info[k] for k in ["epochs_run", "best_inner_val_rmse", "timing_s"]}))


if __name__ == "__main__":
    main()
