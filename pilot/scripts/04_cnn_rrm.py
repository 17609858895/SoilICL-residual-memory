"""Stage 2b: learned spectral encoder (1D CNN) + retrieval-augmented residual memory.

Fixed design (literature-based, not tuned on test folds):
  input  : SNV MIR spectra, 1,701 points averaged in bins of 8 -> 212 points (16 cm-1),
           band-wise standardisation with training-fold statistics
  model  : 4 conv blocks (16/32/64/64 channels, kernels 9/7/5/3, BN, ReLU, max-pool 2)
           -> adaptive avg-pool(8) -> dropout 0.2 -> dense 128 (embedding) -> 1 output
  target : standardised log1p(SOC)
  train  : AdamW (lr 1e-3, wd 1e-4), batch 512, <= 40 epochs, ReduceLROnPlateau,
           early stopping (patience 6) on the same 10% group hold-out used in stage 1
  memory : 128-d embedding of every training sample + its in-sample residual;
           test prediction = CNN(x) + inverse-distance mean residual of k retrieved neighbours
python scripts/04_cnn_rrm.py --fold 1
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
spec = importlib.util.spec_from_file_location("pilot", ROOT / "scripts" / "01_pilot_mir_soc.py")
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)


class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, 16, 9), block(16, 32, 7), block(32, 64, 5), block(64, 64, 3),
                                      nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(0.2))
        self.embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.head = nn.Linear(128, 1)

    def forward(self, x, return_embedding=False):
        z = self.embed(self.features(x))
        out = self.head(z).squeeze(-1)
        return (out, z) if return_embedding else out


def predict(model, X, bs=2048):
    model.eval(); outs, embs = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o, z = model(torch.from_numpy(X[i:i + bs]).unsqueeze(1), return_embedding=True)
            outs.append(o.numpy()); embs.append(z.numpy())
    return np.concatenate(outs), np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max_epochs", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    out = ROOT / "results" / "P1_stage2b"; out.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); info = {"fold": args.fold}

    X = np.load(ROOT / "data" / "mir_soc_X.npy")
    meta = pd.read_parquet(ROOT / "data" / "mir_soc_meta.parquet")
    X = pilot.snv(X)
    X = X[:, :1696].reshape(len(X), 212, 8).mean(2).astype(np.float32)
    y = np.log1p(meta["soc"].to_numpy(np.float64))
    fold = pd.read_parquet(ROOT / "results" / "P1" / "fold_assignment.parquet")["fold"].to_numpy()
    te = np.where(fold == args.fold)[0]; tr = np.where(fold != args.fold)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    itr, iva = next(gss.split(np.zeros(len(tr)), groups=meta["group_block"].values[tr]))

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32); del X
    ym, ys = y[tr].mean(), y[tr].std()
    yz = ((y - ym) / ys).astype(np.float32)

    Xa, ya = torch.from_numpy(Xs[tr[itr]]).unsqueeze(1), torch.from_numpy(yz[tr[itr]])
    Xv, yv = Xs[tr[iva]], yz[tr[iva]]
    model = SpecCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.MSELoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(SEED)
    for ep in range(args.max_epochs):
        model.train(); t = time.time()
        perm = torch.randperm(len(ya), generator=g)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad(); loss = lossf(model(Xa[b]), ya[b]); loss.backward(); opt.step()
        pv, _ = predict(model, Xv)
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
    torch.save(best_state, out / f"fold{args.fold}_cnn_state.pt")
    info.update({"epochs_run": len(hist), "best_inner_val_rmse": best, "val_history": hist})

    p_tr, z_tr = predict(model, Xs[tr]); p_te, z_te = predict(model, Xs[te])
    p_tr, p_te = p_tr * ys + ym, p_te * ys + ym
    r_in = y[tr] - p_tr
    pred = pd.DataFrame({"row": te, "fold": args.fold, "y_true": y[te], "source": meta["source"].values[te],
                         "group": meta["group_block"].values[te], "cnn": p_te})
    dist, idx = NearestNeighbors(n_neighbors=50).fit(z_tr).kneighbors(z_te)
    for k in (5, 10, 20, 50):
        w = 1.0 / (dist[:, :k] + 1e-6)
        pred[f"cnn_rrm_k{k}"] = p_te + (w * r_in[idx[:, :k]]).sum(1) / w.sum(1)
    pred.to_parquet(out / f"fold{args.fold}_pred.parquet", index=False)
    info["timing_s"] = time.time() - T0
    json.dump(info, open(out / f"fold{args.fold}_info.json", "w"), indent=1)
    print(json.dumps({k: info[k] for k in ["epochs_run", "best_inner_val_rmse", "timing_s"]}))


if __name__ == "__main__":
    main()
