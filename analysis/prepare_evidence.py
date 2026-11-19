"""Recover row identities and summarize existing predictions; never fit a model.

The MIR identity table is recovered using the original complete-spectrum/SOC
filters. Every row, source, label and spatial group must match frozen predictions.
"""
import argparse
from pathlib import Path
import csv
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd

SEED, B = 20261012, 2000
KEY = ['dataset.code_ascii_txt', 'id.layer_uuid_txt']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(8 << 20), b''):
            h.update(c)
    return h.hexdigest()


def restore_mir(raw, evidence):
    output = evidence / 'mir_soc_meta_recovered.parquet'
    if output.exists():
        return pd.read_parquet(output)
    p = raw / 'ossl_mir_L0_v1.2.csv.gz'
    assert sha(p) == '8bfed513e9b565664c67b308fa253d3c401d977bef5b75529dc85ffbef2a4118'
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        header = next(csv.reader(f))
    wave = [c for c in header if c.startswith('scan_mir.')]
    assert len(wave) == 1701
    instrument = 'scan.mir.model.name_utf8_txt'
    parts = []
    scanned = 0
    for d in pd.read_csv(p, usecols=KEY + [instrument] + wave,
                         dtype={c:'float32' for c in wave}, chunksize=1000,
                         na_values=['','NA'], low_memory=False):
        scanned += len(d)
        parts.append(d.loc[d[wave].notna().all(axis=1), KEY + [instrument]])
        if scanned % 20000 == 0:
            print('MIR spectra checked:', scanned, flush=True)
    keys = pd.concat(parts, ignore_index=True)
    lab = pd.read_csv(raw/'ossl_soillab_L1_v1.2.csv.gz', usecols=KEY+['oc_usda.c729_w.pct'], low_memory=False)
    sitecols = ['longitude.point_wgs84_dd','latitude.point_wgs84_dd','id.dataset.site_ascii_txt',
                'layer.upper.depth_usda_cm','layer.lower.depth_usda_cm','location.country_iso.3166_txt']
    site = pd.read_csv(raw/'ossl_soilsite_L0_v1.2.csv.gz',usecols=KEY+sitecols, low_memory=False)
    meta = keys.merge(lab,on=KEY,how='left',validate='one_to_one').merge(site,on=KEY,how='left',validate='one_to_one')
    meta = meta.loc[meta['oc_usda.c729_w.pct'].notna()].reset_index(drop=True)
    meta = meta.rename(columns={KEY[0]:'source','oc_usda.c729_w.pct':'soc',
                               'longitude.point_wgs84_dd':'lon','latitude.point_wgs84_dd':'lat',
                               'id.dataset.site_ascii_txt':'site_id'})
    cell = 'cell_' + np.floor(meta.lon.fillna(0)).astype(int).astype(str) + '_' + np.floor(meta.lat.fillna(0)).astype(int).astype(str)
    sitekey = 'site_' + meta.source + '_' + meta.site_id.astype(str)
    meta['group_block'] = np.where(meta.lon.notna() & meta.lat.notna(),cell,
                                  np.where(meta.site_id.notna(),sitekey,'src_'+meta.source))
    meta['data_row'] = np.arange(len(meta))
    meta.to_parquet(output,index=False)
    print('Recovered MIR metadata:', len(meta), flush=True)
    return meta


def pair_data(vis, mir, vm, mm, evidence):
    for name,d,m in [('VisNIR',vis,vm),('MIR',mir,mm)]:
        assert d.row.is_unique and m.data_row.is_unique
        assert len(d) == len(m)
        joined = d.merge(m[['data_row','source','soc','group_block']],left_on='row',right_on='data_row',validate='one_to_one',suffixes=('','_meta'))
        assert (joined.source == joined.source_meta).all(), name
        assert (joined.group == joined.group_block).all(), name
        assert np.allclose(np.expm1(joined.y_true),joined.soc,atol=1e-9,rtol=1e-8), name
    vv = vis.merge(vm[['data_row','id.layer_uuid_txt']],left_on='row',right_on='data_row',validate='one_to_one')
    vv = vv[['source','id.layer_uuid_txt','row','y_true','group','OSSL_Cubist','SoilICL_lite']].rename(columns={'row':'row_visnir','OSSL_Cubist':'VIS_Cubist','SoilICL_lite':'VIS_LRM'})
    m = mir.merge(mm[['data_row','id.layer_uuid_txt']],left_on='row',right_on='data_row',validate='one_to_one')
    m = m[['source','id.layer_uuid_txt','row','y_true','fold','group','M0_OSSL_Cubist','S2_primary_SoilICL_lite']].rename(columns={'row':'row_mir','y_true':'y_mir','fold':'fold_mir','group':'group_mir','M0_OSSL_Cubist':'MIR_Cubist','S2_primary_SoilICL_lite':'MIR_LRM'})
    j = vv.merge(m,on=['source','id.layer_uuid_txt'],validate='one_to_one')
    assert np.allclose(j.y_true,j.y_mir,atol=1e-9)
    assert (j.group == j.group_mir).all()
    assert len(j) == 14277 and len(j[j.fold_mir != 1]) == 11414
    j.to_parquet(evidence/'paired_predictions_all.parquet',index=False)
    return j


def summarize_pair(j, pilot_results, evidence):
    models = ['VIS_Cubist','VIS_LRM','MIR_Cubist','MIR_LRM']
    rows, contrast = [], []
    # Primary paired panels exclude MIR's development fold for every source.
    confirmation = j[j.fold_mir != 1]
    subsets = [('confirmation','All paired',confirmation)]
    subsets += [('confirmation',s,q) for s,q in confirmation.groupby('source')]
    subsets += [('all_including_development','All paired',j)]
    subsets += [('all_including_development',s,q) for s,q in j.groupby('source')]
    for scope,source,d in subsets:
        groups, inv = np.unique(d.group, return_inverse=True)
        cnt = np.bincount(inv)
        for scale in ['log1p','wt_pct']:
            y = d.y_true.to_numpy()
            pp = d[models].to_numpy()
            if scale == 'wt_pct':
                y,pp = np.expm1(y),np.expm1(pp)
            sq = (pp-y[:,None])**2
            gse = np.array([np.bincount(inv,weights=sq[:,i],minlength=len(groups)) for i in range(4)]).T
            point = np.sqrt(sq.mean(axis=0))
            rng = np.random.default_rng(SEED)
            draws = []
            for _ in range(B):
                ix = rng.integers(0,len(groups),len(groups))
                draws.append(np.sqrt(gse[ix].sum(axis=0)/cnt[ix].sum()))
            draws = np.array(draws)
            for i,m in enumerate(models):
                lo,hi = np.quantile(draws[:,i],[.025,.975])
                rows.append(dict(scope=scope,source=source,scale=scale,model=m,n=len(d),groups=len(groups),RMSE=point[i],ci_low=lo,ci_high=hi))
            den = point[0]-point[2]
            den_draw = draws[:,0]-draws[:,2]
            num = point[0]-point[1]
            num_draw = draws[:,0]-draws[:,1]
            den_lo,den_hi = np.quantile(den_draw,[.025,.975])
            # A positive point denominator alone is not sufficient for a stable ratio.
            usable = den > 0 and den_lo > 0
            ratio = 100*num/den if usable else np.nan
            ratio_draw = 100*num_draw/den_draw if usable else np.array([np.nan])
            clo,chi = np.quantile(ratio_draw,[.025,.975])
            rec = dict(scope=scope,source=source,scale=scale,n=len(d),groups=len(groups),
                       baseline_gap=den,gap_ci_low=den_lo,gap_ci_high=den_hi,
                       visnir_memory_gain=num,ratio_usable=bool(usable),gap_closure_pct=ratio,
                       closure_ci_low=clo,closure_ci_high=chi,
                       visnir_lrm_over_mir_cubist=point[1]/point[2])
            for key,values in [('memory_gain',num_draw),('remaining_gap',draws[:,1]-draws[:,2])]:
                rec[key+'_ci_low'],rec[key+'_ci_high'] = np.quantile(values,[.025,.975])
            rec['remaining_gap'] = point[1]-point[2]
            contrast.append(rec)
    metrics = pd.DataFrame(rows)
    contrasts = pd.DataFrame(contrast)
    metrics.to_csv(evidence/'paired_rmse_groupboot.csv',index=False,encoding='utf-8-sig')
    contrasts.to_csv(evidence/'paired_gap_groupboot.csv',index=False,encoding='utf-8-sig')
    old = pd.read_csv(pilot_results/'V_visnir/summary/visnir_mir_gap_closure.csv')
    check = []
    for _,r in old.iterrows():
        scope = 'confirmation' if r['subset'].startswith('paired,') else 'all_including_development'
        source = r['subset'].removeprefix('source ') if r['subset'].startswith('source ') else 'All paired'
        q = metrics[(metrics.scope==scope)&(metrics.source==source)&(metrics.scale=='log1p')].set_index('model')
        for m,oldm in zip(models,['VIS_Cubist','VIS_SoilICL','MIR_Cubist','MIR_SoilICL']):
            diff = abs(q.loc[m,'RMSE']-r['RMSE_log1p_'+oldm])
            assert diff < 1e-10
            check.append(dict(subset=r['subset'],model=m,absolute_difference=diff))
    (evidence/'paired_historical_reproduction.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    return metrics,contrasts


def provenance(mm, pilot_results, evidence):
    p = pd.read_csv(pilot_results/'P2/provenance_enrichment.csv')
    counts = []
    for scenario,d in p.groupby('scenario',sort=False):
        c = np.load(pilot_results/'P2'/scenario/'cubist_side_slim.npz')
        tr,tg = c['tr'],c['tg']
        assert len(tg) == int(d.n_query.iloc[0])
        assert np.allclose(c['ytg'],np.log1p(mm.soc.to_numpy()[tg]),atol=1e-8)
        lib = mm.iloc[tr].source.value_counts()
        if scenario in ['SCHIEDUNG','SERBIA']:
            qgroups = mm.iloc[tg].source + '_' + mm.iloc[tg].site_id.astype(str)
        else:
            qgroups = mm.iloc[tg].group_block
        for _,r in d.iterrows():
            nlib = int(lib[r.neighbour_source])
            assert np.isclose(nlib/len(tr),r.library_share)
            nretrieval = int(round(r.neighbour_share*20*len(tg)))
            assert np.isclose(nretrieval/(20*len(tg)),r.neighbour_share,atol=1e-12)
            counts.append(dict(**r.to_dict(),n_reference=nlib,n_retrieval_events=nretrieval,
                               query_groups=qgroups.nunique(),
                               display=(nretrieval==0 or nretrieval>=10) and qgroups.nunique()>=5))
    out = pd.DataFrame(counts)
    out.to_csv(evidence/'provenance_supported.csv',index=False,encoding='utf-8-sig')
    return out


def adaptation(pilot_results, evidence, adaptation_check=None):
    rows, allruns = [], []
    scenarios = [('MIR',s) for s in ['AfSIS','ICRAF','LUCAS_Woodwell','SCHIEDUNG','KSSL']]+[('VisNIR','ICRAF')]
    old = pd.read_csv(adaptation_check) if adaptation_check is not None else None
    for modality,scenario in scenarios:
        raw = pd.read_csv(pilot_results/('P2' if modality=='MIR' else 'V_visnir')/scenario/'adaptation_runs.csv')
        d = raw[raw.method=='SoilICL_lite_insertion'].copy()
        zero = d[d.k==0]
        assert len(zero)==1 and d.n_local_test.nunique()==1
        b = float(zero.RMSE_log1p.iloc[0])
        d['gain_pct'] = 100*(1-d.RMSE_log1p/b)
        d['modality'],d['scenario'] = modality,scenario
        allruns.append(d)
        for k,q in d.groupby('k'):
            mean = q.gain_pct.mean()
            sd = q.gain_pct.std(ddof=1) if len(q)>1 else 0.
            if old is not None:
                ref = old[(old.modality==modality)&(old.scenario==scenario)&(old.k==k)&(old.method=='SoilICL_lite_insertion')].iloc[0]
                assert np.isclose(mean,ref.gain_mean) and np.isclose(sd,ref.gain_sd)
            rows.append(dict(modality=modality,scenario=scenario,k=k,n_query=int(q.n_local_test.iloc[0]),
                             repeats=len(q),gain_mean=mean,gain_sd=sd,
                             negative_draws=int((q.gain_pct<0).sum()),RMSE_k0=b,RMSE_k=float(q.RMSE_log1p.mean())))
    pd.DataFrame(rows).to_csv(evidence/'lrm_budget_heatmap.csv',index=False,encoding='utf-8-sig')
    pd.concat(allruns,ignore_index=True).to_csv(evidence/'lrm_adaptation_runs.csv',index=False,encoding='utf-8-sig')


def phase(vis, mir, evidence, decomposition_check=None):
    rows = []
    for modality,d,base,lrm in [('VisNIR',vis,'OSSL_Cubist','SoilICL_lite'),('MIR',mir[mir.fold!=1],'M0_OSSL_Cubist','S2_primary_SoilICL_lite')]:
        e = d.y_true.to_numpy()-(d[base]+d.CNN).to_numpy()/2
        delta = (d[lrm]-(d[base]+d.CNN)/2).to_numpy()
        gain = 2*e*delta-delta**2
        assert np.allclose(gain,e**2-(e-delta)**2,rtol=1e-8,atol=1e-10)
        pts = pd.DataFrame(dict(row=d.row,group=d.group,e=e,delta=delta,mse_gain=gain))
        pts.to_parquet(evidence/f'correction_phase_{modality}.parquet',index=False)
        benefit = 100*(gain>0).mean()
        if decomposition_check is not None:
            old = pd.read_csv(decomposition_check)
            ref = old[(old.modality==modality)&(old.kind=='all')&(old.scale=='log1p')].iloc[0]
            assert np.isclose(benefit,ref.beneficial_pct)
        groups,inv = np.unique(d.group,return_inverse=True)
        cnt = np.bincount(inv)
        beneficial = np.bincount(inv,weights=gain>0)
        rng = np.random.default_rng(SEED)
        draws=[]
        for _ in range(B):
            ix=rng.integers(0,len(groups),len(groups))
            draws.append(100*beneficial[ix].sum()/cnt[ix].sum())
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(modality=modality,n=len(d),groups=len(groups),beneficial_pct=benefit,
                         harmful_pct=100*(gain<0).mean(),unchanged_pct=100*(gain==0).mean(),
                         beneficial_ci_low=lo,beneficial_ci_high=hi,
                         residual_min=e.min(),residual_max=e.max(),correction_min=delta.min(),correction_max=delta.max()))
    pd.DataFrame(rows).to_csv(evidence/'correction_phase_summary.csv',index=False,encoding='utf-8-sig')


def main(raw, visnir_meta, pilot_results, evidence, adaptation_check=None, decomposition_check=None):
    start = time.perf_counter()
    evidence.mkdir(parents=True, exist_ok=True)
    inputs=[pilot_results/'summary_stage2/all_predictions_log1p.parquet',
            pilot_results/'V_visnir/summary/P1_all_predictions_log1p.parquet',
            visnir_meta,pilot_results/'P2/provenance_enrichment.csv']
    before = {p:sha(p) for p in inputs}
    mm = restore_mir(raw, evidence)
    mir,vis = pd.read_parquet(inputs[0]),pd.read_parquet(inputs[1])
    vm = pd.read_parquet(inputs[2])
    j = pair_data(vis, mir, vm, mm, evidence)
    metrics,contrasts = summarize_pair(j, pilot_results, evidence)
    p = provenance(mm, pilot_results, evidence)
    adaptation(pilot_results, evidence, adaptation_check)
    phase(vis, mir, evidence, decomposition_check)
    assert all(sha(k)==v for k,v in before.items())
    report = dict(status='passed',models_fitted=0,new_predictions=False,
                  bootstrap_repeats=B,bootstrap_seed=SEED,
                  primary_pair='MIR fold 1 excluded consistently, including all source-specific panels',
                  n_mir_recovered=len(mm),n_paired_all=len(j),n_paired_confirmation=int((j.fold_mir!=1).sum()),
                  pair_group_labels_identical=True,original_metrics_reproduced=True,
                  provenance='Existing MIR holdout retrieval shares; counts and library shares checked against recovered identities and saved indices. Not causal contribution.',
                  support_mask='Positive cells with fewer than 10 neighbour events or query rows with fewer than 5 groups are masked. Observed zero remains 0; absent library sources are NA.',
                  phases='Full frozen VisNIR and MIR confirmation; all signs retained; no gate optimized.',
                  raw_mir_sha256='8bfed513e9b565664c67b308fa253d3c401d977bef5b75529dc85ffbef2a4118',
                  source_hashes={p.name: digest for p, digest in before.items()},seconds=time.perf_counter()-start)
    (evidence/'analysis_scope.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(contrasts[(contrasts.scope=='confirmation')&(contrasts.scale=='log1p')].round(5).to_string(index=False))
    print(p.groupby('scenario').query_groups.first().to_string())
    print('Saved verified evidence',evidence,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('pilot/data'))
    parser.add_argument('--visnir-meta', type=Path, default=Path('pilot/data/visnir_soc_meta.parquet'))
    parser.add_argument('--pilot-results', type=Path, default=Path('pilot/results'))
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--adaptation-check', type=Path)
    parser.add_argument('--decomposition-check', type=Path)
    main(**vars(parser.parse_args()))
