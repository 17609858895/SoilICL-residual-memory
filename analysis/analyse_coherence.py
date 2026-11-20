"""Coherence of the applied residual correction, its expected benefit and the locality of
residual information. Readouts on the fixed LRM predictions; no predictor is refitted.

Applied correction delta = (delta_C + delta_N) / 2, where each branch correction is the
inverse-distance weighted mean of k = 20 reference residuals. Its sampling variance is
tau^2 = (tau_C^2 + tau_N^2) / 4 with tau_m^2 = s_m^2 / k_eff,m (independent branch noise).
Coherence c = delta^2 / tau^2. If delta estimates the local mean error mu, Eq. (2) gives an
expected reduction mu^2 - tau^2, estimated without the query label by b = delta^2 - 2 tau^2,
which is positive when c > 2.

Query SOC is used only to evaluate the readouts. Inputs: VisNIR fold side arrays and the
source-holdout side arrays (VisNIR ICRAF and KSSL; MIR AfSIS, ICRAF, KSSL, LUCAS-Woodwell,
Schiedung, Serbia). Paired group bootstrap: 2,000 resamples of existing spatial/site groups.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import argparse
ARGS = None
PIL = EVD = SRC = None
B, SEED, K = 2000, 20261120, 20
SHORT = {'LUCAS.SSL': 'LUCAS', 'KSSL.SSL': 'KSSL', 'ICRAF.ISRIC': 'ICRAF–ISRIC', 'LUCAS.WOODWELL.SSL': 'LUCAS–Woodwell'}
HOLD = [('VisNIR', 'V_visnir', 'ICRAF', 'VisNIR ICRAF–ISRIC'), ('VisNIR', 'V_visnir', 'KSSL', 'VisNIR KSSL'),
        ('MIR', 'P2', 'AfSIS', 'MIR AfSIS'), ('MIR', 'P2', 'ICRAF', 'MIR ICRAF–ISRIC'), ('MIR', 'P2', 'KSSL', 'MIR KSSL'),
        ('MIR', 'P2', 'LUCAS_Woodwell', 'MIR LUCAS–Woodwell'), ('MIR', 'P2', 'SCHIEDUNG', 'MIR Schiedung'), ('MIR', 'P2', 'SERBIA', 'MIR Serbia')]
BANDS = [(0, 20), (20, 40), (40, 60), (80, 100), (180, 200)]
GRID = np.linspace(0, 1, 101)


# ---------- helpers ----------
def boot_w(G):
    return np.random.default_rng([SEED, G]).multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)


def wts(d):
    w = 1.0 / (d + 1e-6)
    return w / w.sum(1, keepdims=True)


def memory(d, r):
    """Weighted residual mean and its sampling variance tau^2 = s^2 / k_eff."""
    w = wts(d)
    dl = (w * r).sum(1)
    s2 = (w * (r - dl[:, None]) ** 2).sum(1)
    return dl, s2 * (w ** 2).sum(1), np.sqrt(s2)


def gsum(group, cols):
    _, gi = np.unique(group, return_inverse=True)
    G = gi.max() + 1
    return np.column_stack([np.bincount(gi, c, G) for c in cols]), G


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ratio_ci(group, num, den):
    S, G = gsum(group, [num, den])
    bs = boot_w(G) @ S
    return 100 * num.sum() / den.sum(), ci(100 * bs[:, 0] / bs[:, 1]), G


def curve(gain, score):
    o = np.argsort(-score)
    c = np.cumsum(gain[o]) / gain.sum()
    x = np.arange(1, len(o) + 1) / len(o)
    return np.interp(GRID, x, c, left=0)


def split(group, e, d, keep):
    g = e ** 2 - (e - d) ** 2
    S, G = gsum(group, [g * keep, g * ~keep, e ** 2 * keep, e ** 2 * ~keep])
    bs = boot_w(G) @ S
    T = S.sum(0)
    return dict(groups=G, soils_pct=100 * keep.mean(),
                gain_share_pct=100 * T[0] / (T[0] + T[1]), gain_share_ci=ci(100 * bs[:, 0] / (bs[:, 0] + bs[:, 1])),
                gain_in_pct=100 * T[0] / T[2], gain_in_ci=ci(100 * bs[:, 0] / bs[:, 2]),
                gain_out_pct=100 * T[1] / T[3], gain_out_ci=ci(100 * bs[:, 1] / bs[:, 3]),
                improved_in_pct=100 * float((g[keep] > 0).mean()), improved_out_pct=100 * float((g[~keep] > 0).mean()))


# ---------- data ----------
def visnir():
    rows, loc = [], []
    for f in range(5):
        C = np.load(PIL / f'V_visnir/fold{f}/cubist_side_slim.npz')
        N = np.load(PIL / f'V_visnir/fold{f}/cnn_side.npz')
        P = pd.read_parquet(PIL / f'V_visnir/fold{f}/zero_shot_predictions.parquet')
        s_tr, s_tg = SRC[C['tr']], SRC[C['tg']]
        assert (s_tg == P.source.values).all()
        o = dict(fold=f, row=P.row.values, group=P.group.values, source=s_tg, y=C['ytg'], cub=C['cub_tg'], cnn=N['cnn_tg'])
        for tag, Ztr, Ztg, r in [('C', C['Ztr'], C['Ztg'], C['r_tr']), ('N', N['z_tr'], N['z_tg'], N['r_tr'])]:
            d, i = NearestNeighbors(n_neighbors=K).fit(Ztr).kneighbors(Ztg)
            o[f'd{tag}'], o[f't{tag}'], o[f's{tag}'] = memory(d, r[i])
            o[f'dist{tag}'] = d.mean(1)
            o[f'same{tag}'] = (s_tr[i] == s_tg[:, None]).mean(1)
            # locality: same-source neighbour-rank bands and other-source nearest references
            for lab, same, bands in [('same', True, BANDS), ('other', False, BANDS[:1])]:
                for b0, _ in bands:
                    o[f'{tag}_{lab}_{b0}'] = np.full(len(s_tg), np.nan)
                    o[f'r{tag}_{lab}_{b0}'] = np.full(len(s_tg), np.nan)
                kmax = bands[-1][1]
                for s in np.unique(s_tg):
                    q = s_tg == s
                    ref = (s_tr == s) if same else (s_tr != s)
                    if ref.sum() < kmax:
                        continue
                    dd, ii = NearestNeighbors(n_neighbors=kmax).fit(Ztr[ref]).kneighbors(Ztg[q])
                    rr = r[ref][ii]
                    for b0, b1 in bands:
                        o[f'{tag}_{lab}_{b0}'][q] = memory(dd[:, b0:b1], rr[:, b0:b1])[0]
                        o[f'r{tag}_{lab}_{b0}'][q] = dd[:, b0:b1].mean(1) / d.mean(1)[q]
        df = pd.DataFrame(o)
        assert np.abs((df.cub + df.dC + df.cnn + df.dN) / 2 - P['SoilICL_lite'].values).max() < 1e-10
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def holdouts():
    out = []
    for mod, root, s, lab in HOLD:
        C = np.load(PIL / f'{root}/{s}/cubist_side_slim.npz')
        N = np.load(PIL / f'{root}/{s}/cnn_side.npz')
        P = pd.read_parquet(PIL / f'{root}/{s}/zero_shot_predictions.parquet')
        dC, iC = NearestNeighbors(n_neighbors=K).fit(C['Ztr']).kneighbors(C['Ztg'])
        dN, iN = NearestNeighbors(n_neighbors=K).fit(N['z_tr']).kneighbors(N['z_tg'])
        deC, tC, _ = memory(dC, C['r_tr'][iC])
        deN, tN, _ = memory(dN, N['r_tr'][iN])
        base = (C['cub_tg'] + N['cnn_tg']) / 2
        assert np.abs(base + (deC + deN) / 2 - P['SoilICL_lite'].values).max() < 1e-10
        out.append(pd.DataFrame(dict(series=lab, modality=mod, group=P.group.values, y=C['ytg'], base=base,
                                     dC=deC, dN=deN, tC=tC, tN=tN)))
    return pd.concat(out, ignore_index=True)


def applied(df):
    d = ((df.dC + df.dN) / 2).to_numpy()
    tau2 = ((df.tC + df.tN) / 4).to_numpy()
    return d, tau2, d ** 2 / tau2


# ---------- readouts ----------
def main():
    V = visnir()
    V.to_parquet(EVD / 'visnir_coherence.parquet', index=False)
    base = ((V.cub + V.cnn) / 2).to_numpy()
    y = V.y.to_numpy()
    e = y - base
    d, tau2, c = applied(V)
    b = d ** 2 - 2 * tau2
    g = e ** 2 - (e - d) ** 2
    grp = V.group.to_numpy()
    summary = dict(models_fitted=0, bootstrap_repeats=B, bootstrap_seed=SEED, k=K)

    # deciles of coherence and of correction magnitude
    dec_rows = []
    for name, sc in [('Coherence', c), ('Correction magnitude', np.abs(d))]:
        dec = pd.qcut(sc, 10, labels=False)
        for k in range(10):
            m = dec == k
            val, cint, G = ratio_ci(grp[m], g[m], e[m] ** 2)
            imp = (g[m] > 0).astype(float)
            S, _ = gsum(grp[m], [imp, np.ones(m.sum())])
            bs = boot_w(G) @ S
            dec_rows.append(dict(ranking=name, decile=k + 1, n=int(m.sum()), groups=G, key_median=float(np.median(sc[m])),
                                 gain_pct=val, gain_lo=cint[0], gain_hi=cint[1], improved_pct=100 * imp.mean(),
                                 improved_lo=ci(100 * bs[:, 0] / bs[:, 1])[0], improved_hi=ci(100 * bs[:, 0] / bs[:, 1])[1],
                                 base_mse=float((e[m] ** 2).mean()), correction_energy=float((d[m] ** 2).mean()),
                                 lam_star=float((e[m] * d[m]).sum() / (d[m] ** 2).sum()),
                                 share_of_total_gain_pct=100 * g[m].sum() / g.sum()))
    pd.DataFrame(dec_rows).to_csv(EVD / 'coherence_deciles.csv', index=False)

    # label-free rankings
    rk = lambda s: pd.Series(s).groupby(V.fold.values).rank(pct=True).to_numpy()
    scores = {'Coherence': c, 'Expected benefit': b, 'Correction magnitude': np.abs(d),
              'Neighbour closeness': (rk(-V.distC.to_numpy()) + rk(-V.distN.to_numpy())) / 2,
              'Low neighbour spread': (rk(-V.sC.to_numpy()) + rk(-V.sN.to_numpy())) / 2}
    rank_rows, curves = [], []
    for name, sc in scores.items():
        top = (sc >= np.quantile(sc, 0.7)).astype(float)
        val, cint, _ = ratio_ci(grp, g * top, g)
        rank_rows.append(dict(ranking=name, top30_share_pct=val, lo=cint[0], hi=cint[1],
                              top30_base_mse_ratio=float((e[top > 0] ** 2).mean() / (e ** 2).mean())))
        curves += [dict(series='VisNIR spatial folds', ranking=name, fraction=x, cum_share=v) for x, v in zip(GRID, curve(g, sc))]
    pd.DataFrame(rank_rows).to_csv(EVD / 'coherence_rankings.csv', index=False)

    # coherence within deciles of correction magnitude
    md = np.minimum((rk(np.abs(d)) * 10).astype(int), 9)
    cq = np.zeros(len(V), int)
    for k in range(10):
        m = md == k
        cq[m] = np.minimum((pd.Series(c[m]).rank(pct=True).to_numpy() * 5).astype(int), 4)
    wm = []
    for k in range(10):
        for q in range(5):
            m = (md == k) & (cq == q)
            wm.append(dict(magnitude_decile=k + 1, coherence_quintile=q + 1, n=int(m.sum()),
                           gain_pct=100 * g[m].sum() / (e[m] ** 2).sum(), improved_pct=100 * float((g[m] > 0).mean())))
    pd.DataFrame(wm).to_csv(EVD / 'coherence_within_magnitude.csv', index=False)
    summary['within_magnitude'] = {}
    for lab, m in [('top', cq == 4), ('bottom', cq == 0)]:
        val, cint, G = ratio_ci(grp[m], g[m], e[m] ** 2)
        summary['within_magnitude'][lab] = dict(gain_pct=val, ci=cint, groups=G, improved_pct=100 * float((g[m] > 0).mean()))

    # expected-benefit condition c > 2
    keep = c > 2
    summary['visnir_c_gt_2'] = split(grp, e, d, keep)
    summary['visnir_c_gt_2_by_source'] = {SHORT[s]: split(grp[V.source.values == s], e[V.source.values == s],
                                                          d[V.source.values == s], keep[V.source.values == s]) for s in SHORT}

    # holdouts: concentration and c > 2
    H = holdouts()
    hrows = []
    for lab, t in H.groupby('series', sort=False):
        eh = (t.y - t.base).to_numpy()
        dh, _, ch = applied(t)
        gh = eh ** 2 - (eh - dh) ** 2
        cur = curve(gh, ch)
        curves += [dict(series=lab, ranking='Coherence', fraction=x, cum_share=v) for x, v in zip(GRID, cur)]
        qi = np.minimum((pd.Series(ch).rank(pct=True).to_numpy() * 5).astype(int), 4)
        tot, tot_ci, G = ratio_ci(t.group.to_numpy(), gh, eh ** 2)
        row = dict(series=lab, modality=t.modality.iloc[0], n=len(t), groups=G, total_gain_pct=tot, total_lo=tot_ci[0], total_hi=tot_ci[1],
                   top30_share_pct=100 * cur[30], top10_share_pct=100 * cur[10],
                   **{f'q{k + 1}_gain_pct': 100 * gh[qi == k].sum() / (eh[qi == k] ** 2).sum() for k in range(5)})
        sp = split(t.group.to_numpy(), eh, dh, ch > 2)
        row.update({f'c2_{k}': v for k, v in sp.items()})
        hrows.append(row)
    pd.DataFrame(hrows).to_csv(EVD / 'coherence_holdouts.csv', index=False)
    pd.DataFrame(curves).to_csv(EVD / 'coherence_curves.csv', index=False)
    qi = np.minimum((pd.Series(c).groupby(V.fold.values).rank(pct=True).to_numpy() * 5).astype(int), 4)
    summary['visnir_quintiles'] = [100 * g[qi == k].sum() / (e[qi == k] ** 2).sum() for k in range(5)]
    summary['visnir_top30_share_pct'] = float(100 * curve(g, c)[30])
    summary['visnir_top10_share_pct'] = float(100 * curve(g, c)[10])

    # locality and source restriction
    loc = []
    for lab, b0 in [('Same source', bb[0]) for bb in BANDS] + [('Other sources', 0)]:
        tag = 'same' if lab == 'Same source' else 'other'
        dd = ((V[f'C_{tag}_{b0}'] + V[f'N_{tag}_{b0}']) / 2).to_numpy()
        ok = np.isfinite(dd)
        gg = e[ok] ** 2 - (e[ok] - dd[ok]) ** 2
        val, cint, G = ratio_ci(grp[ok], gg, e[ok] ** 2)
        loc.append(dict(references=lab, ranks=f'{b0 + 1}–{b0 + 20}', n=int(ok.sum()), groups=G, net_gain_pct=val, lo=cint[0], hi=cint[1],
                        rel_distance=float(np.nanmedian((V[f'rC_{tag}_{b0}'] + V[f'rN_{tag}_{b0}']).to_numpy()[ok] / 2))))
    pd.DataFrame(loc).to_csv(EVD / 'residual_locality.csv', index=False)
    same = ((V.sameC + V.sameN) / 2).to_numpy()
    rest = []
    dsame = ((V.C_same_0 + V.N_same_0) / 2).to_numpy()
    dother = ((V.C_other_0 + V.N_other_0) / 2).to_numpy()
    for s in ['All'] + list(SHORT):
        m = np.ones(len(V), bool) if s == 'All' else (V.source.values == s)
        for lab, dd in [('All references', d), ('Same-source references', dsame), ('Other-source references', dother)]:
            ok = m & np.isfinite(dd)
            gg = e[ok] ** 2 - (e[ok] - dd[ok]) ** 2
            val, cint, G = ratio_ci(grp[ok], gg, e[ok] ** 2)
            rest.append(dict(source=SHORT.get(s, s), memory=lab, n=int(ok.sum()), groups=G, net_gain_pct=val, lo=cint[0], hi=cint[1],
                             same_source_neighbour_pct=100 * same[m].mean()))
    pd.DataFrame(rest).to_csv(EVD / 'restricted_memory_gains.csv', index=False)
    summary['same_source_neighbour_pct'] = float(100 * same.mean())
    json.dump(summary, open(EVD / 'coherence_summary.json', 'w'), indent=2, default=float)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(dec_rows).round(3).to_string())
    print(pd.DataFrame(rank_rows).round(2).to_string())
    print(json.dumps(summary, indent=1, default=float))
    print(pd.DataFrame(hrows).round(2).T.to_string())
    print(pd.DataFrame(loc).round(3).to_string())
    print(pd.DataFrame(rest).round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-results', type=Path, default=Path('pilot/results'))
    ap.add_argument('--visnir-meta', type=Path, required=True, help='VisNIR metadata parquet with a source column in modelling-row order')
    ap.add_argument('--output', type=Path, required=True)
    ARGS = ap.parse_args()
    PIL, EVD = ARGS.pilot_results, ARGS.output
    EVD.mkdir(parents=True, exist_ok=True)
    SRC = pd.read_parquet(ARGS.visnir_meta).source.to_numpy()
    main()
