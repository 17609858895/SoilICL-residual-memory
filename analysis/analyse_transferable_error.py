"""Transferable-error readouts of the frozen LRM predictions (no model is refitted).

Inputs are the frozen fold side arrays (stored training residuals, PLS-20 and CNN
representations, outer-test branch predictions) and the frozen MIR confirmation
predictions. LRM is first reproduced exactly from the side arrays; every readout
below is then a function of those reproduced corrections.

Readouts
  1. Correction scale: G(lam) = 2 lam A - lam^2 M per branch and for LRM, with
     lam* = A/M (retrospective) and a leave-one-fold-out scale fitted on the other
     folds' outer-test predictions (cross-fitted; the held-out fold's labels never
     enter its own scale).
  2. Agreement of the two branch corrections (sign of delta_C and delta_N).
  3. Neighbour distance and neighbour-residual coherence (label-free at query time).
  4. Neighbourhood size k (VisNIR recomputed; MIR from stored k5..k50 outputs).
Paired group bootstrap: 2,000 resamples of existing spatial/site groups.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

B, SEED, KS = 2000, 20261104, [5, 10, 20, 50]


def weights(d):
    w = 1.0 / (d + 1e-6)
    return w / w.sum(1, keepdims=True)


def visnir_rows(pilot_results):
    rows = []
    for f in range(5):
        C = np.load(pilot_results / f'V_visnir/fold{f}/cubist_side_slim.npz')
        N = np.load(pilot_results / f'V_visnir/fold{f}/cnn_side.npz')
        P = pd.read_parquet(pilot_results / f'V_visnir/fold{f}/zero_shot_predictions.parquet')
        dC, iC = NearestNeighbors(n_neighbors=50).fit(C['Ztr']).kneighbors(C['Ztg'])
        dN, iN = NearestNeighbors(n_neighbors=50).fit(N['z_tr']).kneighbors(N['z_tg'])
        # reference-to-reference nearest-neighbour scale (training side only) for distance normalisation
        sub = np.random.default_rng(f).choice(len(C['Ztr']), 3000, replace=False)
        refC = NearestNeighbors(n_neighbors=21).fit(C['Ztr']).kneighbors(C['Ztr'][sub])[0][:, 1:].mean(1)
        refN = NearestNeighbors(n_neighbors=21).fit(N['z_tr']).kneighbors(N['z_tr'][sub])[0][:, 1:].mean(1)
        out = dict(fold=f, row=P.row.values, group=P.group.values, source=P.source.values,
                   y=C['ytg'], cub=C['cub_tg'], cnn=N['cnn_tg'])
        for k in KS:
            for tag, d, i, r in [('C', dC, iC, C['r_tr']), ('N', dN, iN, N['r_tr'])]:
                w = weights(d[:, :k]); rr = r[i[:, :k]]
                dl = (w * rr).sum(1)
                out[f'd{tag}_k{k}'] = dl
                if k == 20:
                    s2 = (w * (rr - dl[:, None]) ** 2).sum(1)
                    keff = 1.0 / (w ** 2).sum(1)
                    out[f'snr{tag}'] = dl ** 2 / (s2 / keff + 1e-12)
                    out[f'dist{tag}'] = d[:, :k].mean(1) / np.median(refC if tag == 'C' else refN)
        df = pd.DataFrame(out)
        lrm = ((df.cub + df.dC_k20) + (df.cnn + df.dN_k20)) / 2
        err = float(np.abs(lrm - P['SoilICL_lite'].values).max())
        assert err < 1e-10, err
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def mir_rows(pilot_results, mir_phase):
    a = pd.concat([pd.read_parquet(pilot_results / f'P1_stage2/fold{f}_pred.parquet') for f in [0, 2, 3, 4]])
    b = pd.concat([pd.read_parquet(pilot_results / f'P1_stage2b/fold{f}_pred.parquet') for f in [0, 2, 3, 4]])
    m = a.merge(b[['row', 'cnn'] + [f'cnn_rrm_k{k}' for k in KS]], on='row')
    df = pd.DataFrame(dict(fold=m.fold.values, row=m.row.values, group=m.group.values, source=m.source.values,
                           y=m.y_true.values, cub=m.cubist_plain.values, cnn=m.cnn.values))
    for k in KS:
        df[f'dC_k{k}'] = m[f'rrm_pls_in_k{k}'].values - m.cubist_plain.values
        df[f'dN_k{k}'] = m[f'cnn_rrm_k{k}'].values - m.cnn.values
    cp = pd.read_parquet(mir_phase)
    chk = df.merge(cp, on='row')
    assert np.abs((chk.dC_k20 + chk.dN_k20) / 2 - chk.delta).max() < 1e-10
    return df


def group_sums(df, cols):
    g = df.groupby('group', sort=False)[cols].sum()
    return g.to_numpy(), len(g)


def boot_w(G):
    # Seeded by (SEED, number of groups): results do not depend on call order, and
    # readouts on the same group set share the same resamples (paired comparisons).
    return np.random.default_rng([SEED, G]).multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)


def scale_table(df, modality):
    """Branch and LRM scale readouts; MSE shares are relative to the matching base."""
    rows, curves = [], []
    base = (df.cub + df.cnn) / 2
    specs = {'Cubist branch': (df.y - df.cub, df.dC_k20),
             'CNN branch': (df.y - df.cnn, df.dN_k20),
             'LRM': (df.y - base, (df.dC_k20 + df.dN_k20) / 2)}
    for name, (e, d) in specs.items():
        t = pd.DataFrame(dict(group=df.group, ed=e * d, dd=d * d, ee=e * e))
        S, G = group_sums(t, ['ed', 'dd', 'ee'])
        W = boot_w(G); bs = W @ S
        lam_b = bs[:, 0] / bs[:, 1]
        gain1_b = 100 * (2 * bs[:, 0] - bs[:, 1]) / bs[:, 2]
        gopt_b = 100 * (bs[:, 0] ** 2 / bs[:, 1]) / bs[:, 2]
        A, M, E = S.sum(0)
        rows.append(dict(modality=modality, component=name, n=len(df), groups=G,
                         lam_star=A / M, lam_lo=np.percentile(lam_b, 2.5), lam_hi=np.percentile(lam_b, 97.5),
                         align_pct=100 * 2 * A / E, magnitude_pct=100 * M / E,
                         gain_unit_pct=100 * (2 * A - M) / E,
                         gain_unit_lo=np.percentile(gain1_b, 2.5), gain_unit_hi=np.percentile(gain1_b, 97.5),
                         gain_opt_pct=100 * A * A / M / E,
                         gain_opt_lo=np.percentile(gopt_b, 2.5), gain_opt_hi=np.percentile(gopt_b, 97.5)))
        for lam in np.linspace(0, 2.4, 121):
            curves.append(dict(modality=modality, component=name, lam=lam, gain_pct=100 * (2 * lam * A - lam * lam * M) / E))
        for f, s in t.assign(fold=df.fold.values).groupby('fold'):
            rows.append(dict(modality=modality, component=name + ' fold', fold=int(f), n=len(s),
                             lam_star=s.ed.sum() / s.dd.sum(), gain_unit_pct=100 * (2 * s.ed.sum() - s.dd.sum()) / s.ee.sum()))
    return rows, curves


def crossfit(df, modality):
    """Two-parameter branch scale fitted on the other folds' outer-test predictions."""
    base = (df.cub + df.cnn) / 2
    e = (df.y - base).to_numpy(); X = np.c_[df.dC_k20, df.dN_k20] / 2
    pred = np.zeros(len(df)); lams = []
    for f in sorted(df.fold.unique()):
        tr, te = (df.fold != f).to_numpy(), (df.fold == f).to_numpy()
        lam = np.linalg.lstsq(X[tr], e[tr], rcond=None)[0]
        pred[te] = X[te] @ lam; lams.append(dict(modality=modality, fold=int(f), lam_C=lam[0], lam_N=lam[1]))
    y = df.y.to_numpy(); bz = base.to_numpy()
    out = {}
    for name, z in [('Average', bz), ('LRM', bz + X.sum(1)), ('Cross-fitted scale', bz + pred)]:
        out[name] = z
    t = pd.DataFrame(dict(group=df.group))
    for name, z in out.items():
        t[name + '_sq_log'] = (z - y) ** 2
        t[name + '_sq_nat'] = (np.expm1(z) - np.expm1(y)) ** 2
    cols = [c for c in t.columns if c != 'group']
    S, G = group_sums(t, cols); W = boot_w(G); bs = W @ S
    idx = {c: i for i, c in enumerate(cols)}
    res = []
    for scale in ['log', 'nat']:
        for ref in ['Average', 'LRM']:
            a = np.sqrt(bs[:, idx[f'{ref}_sq_{scale}']]); c = np.sqrt(bs[:, idx[f'Cross-fitted scale_sq_{scale}']])
            pct = 100 * (a - c) / a
            pa = np.sqrt(S.sum(0)[idx[f'{ref}_sq_{scale}']] / len(df)); pc = np.sqrt(S.sum(0)[idx[f'Cross-fitted scale_sq_{scale}']] / len(df))
            res.append(dict(modality=modality, scale=scale, reference=ref, rmse_ref=pa, rmse_crossfit=pc,
                            gain_pct=100 * (pa - pc) / pa, lo=np.percentile(pct, 2.5), hi=np.percentile(pct, 97.5)))
    harm = {}
    for name in ['LRM', 'Cross-fitted scale']:
        dl = out[name] - bz; e2 = (y - bz) ** 2 - (y - out[name]) ** 2
        harm[name] = dict(improved_pct=100 * float((e2 > 0).mean()), worsened_pct=100 * float((e2 < 0).mean()))
    return res, lams, harm


def binned(df, key, q, modality, label):
    base = (df.cub + df.cnn) / 2
    e = df.y - base; d = (df.dC_k20 + df.dN_k20) / 2
    E = (e ** 2).sum()
    t = pd.DataFrame(dict(group=df.group, ed=e * d, dd=d * d, ee=e * e, imp=((e ** 2 - (e - d) ** 2) > 0).astype(float), one=1.0))
    t['bin'] = pd.qcut(key, q, labels=False, duplicates='drop')
    rows = []
    for b, s in t.groupby('bin'):
        Eb = s.ee.sum()
        S, G = group_sums(s, ['ed', 'dd', 'ee', 'imp', 'one']); W = boot_w(G); bs = W @ S
        gb = 100 * (2 * bs[:, 0] - bs[:, 1]) / bs[:, 2]
        rows.append(dict(modality=modality, variable=label, bin=int(b), n=len(s), groups=G,
                         key_median=float(np.median(key[s.index])),
                         align_pct=100 * 2 * s.ed.sum() / Eb, magnitude_pct=100 * s.dd.sum() / Eb,
                         gain_pct=100 * (2 * s.ed.sum() - s.dd.sum()) / Eb,
                         gain_lo=np.percentile(gb, 2.5), gain_hi=np.percentile(gb, 97.5),
                         improved_pct=100 * s.imp.mean(),
                         share_of_total_gain_pct=100 * (2 * s.ed.sum() - s.dd.sum()) / (2 * (e * d).sum() - (d * d).sum()),
                         share_of_base_mse_pct=100 * Eb / E))
    return rows


def agreement(df, modality):
    base = (df.cub + df.cnn) / 2
    e = df.y - base; d = (df.dC_k20 + df.dN_k20) / 2
    agree = np.sign(df.dC_k20) == np.sign(df.dN_k20)
    rows = []
    for lab, m in [('Agree', agree), ('Disagree', ~agree)]:
        s = pd.DataFrame(dict(group=df.group[m], ed=(e * d)[m], dd=(d * d)[m], ee=(e * e)[m],
                              imp=(((e ** 2 - (e - d) ** 2) > 0)[m]).astype(float), one=1.0))
        S, G = group_sums(s, ['ed', 'dd', 'ee', 'imp', 'one']); W = boot_w(G); bs = W @ S
        gb = 100 * (2 * bs[:, 0] - bs[:, 1]) / bs[:, 2]; ib = 100 * bs[:, 3] / bs[:, 4]
        Etot = (e * e).sum()
        rows.append(dict(modality=modality, subset=lab, n=int(m.sum()), share_pct=100 * m.mean(), groups=G,
                         gain_pct=100 * (2 * s.ed.sum() - s.dd.sum()) / s.ee.sum(), gain_lo=np.percentile(gb, 2.5), gain_hi=np.percentile(gb, 97.5),
                         improved_pct=100 * s.imp.mean(), improved_lo=np.percentile(ib, 2.5), improved_hi=np.percentile(ib, 97.5),
                         contribution_to_total_gain_pct=100 * (2 * s.ed.sum() - s.dd.sum()) / (2 * (e * d).sum() - (d * d).sum()),
                         align_pct_of_total_mse=100 * 2 * s.ed.sum() / Etot, magnitude_pct_of_total_mse=100 * s.dd.sum() / Etot))
    return rows


def k_table(df, modality):
    base = (df.cub + df.cnn) / 2; e = df.y - base; rows = []
    for k in KS:
        d = (df[f'dC_k{k}'] + df[f'dN_k{k}']) / 2
        E = (e * e).sum()
        rows.append(dict(modality=modality, k=k, align_pct=100 * 2 * (e * d).sum() / E, magnitude_pct=100 * (d * d).sum() / E,
                         gain_pct=100 * (2 * (e * d).sum() - (d * d).sum()) / E,
                         lam_star=float((e * d).sum() / (d * d).sum()),
                         improved_pct=100 * float(((e ** 2 - (e - d) ** 2) > 0).mean())))
    return rows


def plane(df, modality, lim, n=34):
    """Mean squared-error reduction on a grid of the two branch corrections."""
    base = (df.cub + df.cnn) / 2; e = df.y - base; d = (df.dC_k20 + df.dN_k20) / 2
    g = (e ** 2 - (e - d) ** 2) / (e ** 2).mean()
    edges = np.linspace(-lim, lim, n + 1)
    ix = np.clip(np.digitize(df.dC_k20, edges) - 1, 0, n - 1); iy = np.clip(np.digitize(df.dN_k20, edges) - 1, 0, n - 1)
    t = pd.DataFrame(dict(ix=ix, iy=iy, g=g, one=1)).groupby(['ix', 'iy']).agg(g=('g', 'mean'), n=('one', 'sum')).reset_index()
    t['modality'] = modality; t['x'] = (edges[t.ix] + edges[t.ix + 1]) / 2; t['y'] = (edges[t.iy] + edges[t.iy + 1]) / 2
    return t


def coherence(d, i, r, k=20):
    w = weights(d[:, :k]); rr = r[i[:, :k]]; dl = (w * rr).sum(1)
    s2 = (w * (rr - dl[:, None]) ** 2).sum(1); keff = 1.0 / (w ** 2).sum(1)
    return dl, dl ** 2 / (s2 / keff + 1e-12)


def holdout_rows(pilot_results):
    """Same label-free coherence in the frozen leave-source-out runs (all target rows)."""
    out = []
    for mod, root, scen in [('VisNIR', 'V_visnir', ['ICRAF', 'KSSL']),
                            ('MIR', 'P2', ['AfSIS', 'ICRAF', 'KSSL', 'LUCAS_Woodwell', 'SCHIEDUNG', 'SERBIA'])]:
        for s in scen:
            C = np.load(pilot_results / f'{root}/{s}/cubist_side_slim.npz'); N = np.load(pilot_results / f'{root}/{s}/cnn_side.npz')
            P = pd.read_parquet(pilot_results / f'{root}/{s}/zero_shot_predictions.parquet')
            dC, iC = NearestNeighbors(n_neighbors=20).fit(C['Ztr']).kneighbors(C['Ztg'])
            dN, iN = NearestNeighbors(n_neighbors=20).fit(N['z_tr']).kneighbors(N['z_tg'])
            deC, sC = coherence(dC, iC, C['r_tr']); deN, sN = coherence(dN, iN, N['r_tr'])
            base = (C['cub_tg'] + N['cnn_tg']) / 2
            assert np.abs(base + (deC + deN) / 2 - P['SoilICL_lite'].values).max() < 1e-10
            q = (pd.Series(sC).rank(pct=True) + pd.Series(sN).rank(pct=True)).to_numpy() / 2
            e = C['ytg'] - base; d = (deC + deN) / 2
            out.append(pd.DataFrame(dict(modality=mod, scenario=s, group=P.group.values, coherence=q,
                                         gain=e ** 2 - (e - d) ** 2, ee=e ** 2, agree=np.sign(deC) == np.sign(deN))))
    return pd.concat(out, ignore_index=True)


def concentration(gain, q, grid=np.linspace(0, 1, 101)):
    """Cumulative share of total squared-error reduction, queries ranked by decreasing coherence."""
    o = np.argsort(-q); c = np.cumsum(gain[o]) / gain.sum()
    x = np.arange(1, len(o) + 1) / len(o)
    return np.interp(grid, x, c, left=0)


def fold_source_ci(path):
    """Paired group-bootstrap native-scale gains of LRM by fold and by source (Fig. 2)."""
    S = pd.read_parquet(path); rows = []
    nat = lambda c: np.expm1(S[c].to_numpy())
    y = np.expm1(S.y_true.to_numpy())
    for ref in ['Frozen_plain_average', 'PLS_Cubist_selected']:
        sq = pd.DataFrame(dict(group=S.group, fold=S.fold, source=S.source,
                               a=(nat(ref) - y) ** 2, b=(nat('Frozen_SoilICL_lite') - y) ** 2))
        for kind in ['fold', 'source']:
            for key, s in sq.groupby(kind):
                G, ng = group_sums(s, ['a', 'b']); W = boot_w(ng); bs = W @ G
                pct = 100 * (1 - np.sqrt(bs[:, 1] / bs[:, 0]))
                rows.append(dict(kind=kind, subset=str(key), reference=ref, n=len(s), groups=ng,
                                 gain_pct=100 * (1 - np.sqrt(s.b.sum() / s.a.sum())),
                                 lo=np.percentile(pct, 2.5), hi=np.percentile(pct, 97.5)))
    return pd.DataFrame(rows)


def paired_soils(paired_predictions, output):
    """Per-soil comparison of VisNIR LRM and MIR Cubist on the confirmation pairs (Fig. 3)."""
    P = pd.read_parquet(paired_predictions)
    P = P[P.fold_mir != 1]
    a = np.abs(P.VIS_LRM - P.y_true); b = np.abs(P.MIR_Cubist - P.y_mir); c = np.abs(P.VIS_Cubist - P.y_true)
    rows = []
    for src, m in [('All paired', np.ones(len(P), bool))] + [(s, (P.source == s).to_numpy()) for s in sorted(P.source.unique())]:
        t = pd.DataFrame(dict(group=P.group[m], w1=(a[m] < b[m]).astype(float), w0=(c[m] < b[m]).astype(float), one=1.0))
        G, ng = group_sums(t, ['w1', 'w0', 'one']); W = boot_w(ng); bs = W @ G
        rows.append(dict(source=src, n=int(m.sum()), groups=ng,
                         visnir_lrm_beats_mir_pct=100 * t.w1.mean(), lo=np.percentile(100 * bs[:, 0] / bs[:, 2], 2.5), hi=np.percentile(100 * bs[:, 0] / bs[:, 2], 97.5),
                         visnir_cubist_beats_mir_pct=100 * t.w0.mean(),
                         lo0=np.percentile(100 * bs[:, 1] / bs[:, 2], 2.5), hi0=np.percentile(100 * bs[:, 1] / bs[:, 2], 97.5)))
    P[['source', 'group', 'y_true', 'VIS_Cubist', 'VIS_LRM', 'y_mir', 'MIR_Cubist']].to_parquet(output / 'paired_confirmation_soils.parquet', index=False)
    return pd.DataFrame(rows)


def main(pilot_results, matched_predictions, paired_predictions, mir_phase, output):
    output.mkdir(parents=True, exist_ok=True)
    V = visnir_rows(pilot_results)
    M = mir_rows(pilot_results, mir_phase)
    H = holdout_rows(pilot_results); H.to_parquet(output / 'holdout_coherence.parquet', index=False)
    curves_c, hsum = [], []
    rank0 = lambda s: s.groupby(V.fold).rank(pct=True)
    qV = ((rank0(V.snrC) + rank0(V.snrN)) / 2).to_numpy()
    eV = (V.y - (V.cub + V.cnn) / 2).to_numpy(); dV = ((V.dC_k20 + V.dN_k20) / 2).to_numpy()
    gV = eV ** 2 - (eV - dV) ** 2
    grid = np.linspace(0, 1, 101)
    for lab, g, q, ee in [('VisNIR spatial folds', gV, qV, eV ** 2)] + \
            [(f'{m} {s} holdout', d.gain.to_numpy(), d.coherence.to_numpy(), d.ee.to_numpy()) for (m, s), d in H.groupby(['modality', 'scenario'])]:
        c = concentration(g, q, grid)
        curves_c += [dict(series=lab, fraction=x, cum_gain_share=v) for x, v in zip(grid, c)]
        qi = np.minimum((q * 5).astype(int), 4)
        hsum.append(dict(series=lab, n=len(g), total_gain_pct=100 * g.sum() / ee.sum(), top30_share_pct=100 * c[30],
                         top10_share_pct=100 * c[10],
                         **{f'q{k + 1}_gain_pct': 100 * g[qi == k].sum() / ee[qi == k].sum() for k in range(5)}))
    pd.DataFrame(curves_c).to_csv(output / 'coherence_concentration_curves.csv', index=False)
    pd.DataFrame(hsum).to_csv(output / 'coherence_holdout_summary.csv', index=False)
    # Two coherence-based alternatives (VisNIR): positive-part James-Stein shrinkage of each branch
    # correction (parameter-free) and a cross-fitted scale that varies linearly with the coherence rank.
    bV = ((V.cub + V.cnn) / 2).to_numpy(); yV = V.y.to_numpy()
    shrink = lambda d, s: d * np.clip(1 - 1 / np.maximum(s, 1e-12), 0, 1)
    zJS = bV + (shrink(V.dC_k20.to_numpy(), V.snrC.to_numpy()) + shrink(V.dN_k20.to_numpy(), V.snrN.to_numpy())) / 2
    X = np.c_[dV, qV * dV]; zq = np.zeros(len(V))
    for f in range(5):
        tr = V.fold.values != f; zq[~tr] = X[~tr] @ np.linalg.lstsq(X[tr], eV[tr], rcond=None)[0]
    alt = []
    for name, z in [('Ordinary average', bV), ('LRM', bV + dV), ('James-Stein shrinkage', zJS), ('Coherence-dependent cross-fitted scale', bV + zq)]:
        alt.append(dict(rule=name, rmse_log1p=float(np.sqrt(np.mean((z - yV) ** 2))),
                        rmse_wt=float(np.sqrt(np.mean((np.expm1(z) - np.expm1(yV)) ** 2))),
                        improved_pct=100 * float(((yV - bV) ** 2 - (yV - z) ** 2 > 0).mean())))
    pd.DataFrame(alt).to_csv(output / 'coherence_rule_alternatives.csv', index=False)
    print(pd.DataFrame(alt).round(4).to_string())
    fold_source_ci(matched_predictions).to_csv(output / 'fold_source_gain_ci.csv', index=False)
    paired_soils(paired_predictions, output).to_csv(output / 'paired_soil_wins.csv', index=False)
    print(pd.DataFrame(hsum).round(2).to_string())
    V.to_parquet(output / 'visnir_branch_corrections.parquet', index=False)
    M.to_parquet(output / 'mir_branch_corrections.parquet', index=False)
    scale, curves, xf, lams, harms, agr, ktab, bins, planes = [], [], [], [], {}, [], [], [], []
    for df, mod in [(V, 'VisNIR'), (M, 'MIR')]:
        r, c = scale_table(df, mod); scale += r; curves += c
        r, l, h = crossfit(df, mod); xf += r; lams += l; harms[mod] = h
        agr += agreement(df, mod); ktab += k_table(df, mod)
        planes.append(plane(df, mod, 0.6 if mod == 'VisNIR' else 0.4))
    rank = lambda s: s.groupby(V.fold).rank(pct=True)
    dist = (rank(V.distC) + rank(V.distN)) / 2
    coh = (rank(V.snrC) + rank(V.snrN)) / 2
    bins += binned(V, dist.to_numpy(), 10, 'VisNIR', 'neighbour distance')
    bins += binned(V, coh.to_numpy(), 10, 'VisNIR', 'neighbour coherence')
    for name, obj in [('scale_readouts', scale), ('scale_curves', curves), ('crossfit_scale', xf), ('crossfit_lambdas', lams),
                      ('branch_agreement', agr), ('k_sensitivity', ktab), ('binned_transfer', bins)]:
        pd.DataFrame(obj).to_csv(output / f'{name}.csv', index=False)
    pd.concat(planes).to_csv(output / 'correction_plane.csv', index=False)
    json.dump(dict(models_fitted=0, new_spectral_predictions=False, bootstrap_repeats=B, bootstrap_seed=SEED,
                   lrm_reproduced_from_side_arrays=True, mir_confirmation_folds=[0, 2, 3, 4], visnir_folds=list(range(5)),
                   harm_fraction=harms,
                   note='lam* and binned/agreement readouts use query outcomes and are retrospective. '
                        'Cross-fitted scale uses other folds outer-test predictions only; these folds were evaluated before.'),
              open(output / 'transferable_error_scope.json', 'w'), indent=2)
    print(pd.DataFrame(scale).query("~component.str.contains('fold')").round(3).to_string())
    print(pd.DataFrame(xf).round(4).to_string()); print(pd.DataFrame(lams).round(3).to_string()); print(harms)
    print(pd.DataFrame(agr).round(3).to_string()); print(pd.DataFrame(ktab).round(3).to_string())
    print(pd.DataFrame(bins).round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-results', type=Path, default=Path('pilot/results'))
    parser.add_argument('--matched-predictions', required=True, type=Path)
    parser.add_argument('--paired-predictions', required=True, type=Path)
    parser.add_argument('--mir-phase', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    main(**vars(arguments))
