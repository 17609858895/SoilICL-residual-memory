"""E1a and simple E1c controls. Fixed saved outer splits; no TabPFN invocation.

Default full grid: PLS {10,20,40} x committees {1,5,10,15,20} x nn {0,5,9},
three training-only group folds. --smoke uses a new inner-training holdout,
2000 fitting rows and <=400 query rows, with a reduced grid, never outer-query scores.
All output directories are new/isolated; completed runs are not overwritten.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CUBIST = ROOT / "_deps/cubist_1_2_2"
if LOCAL_CUBIST.exists():
    sys.path.insert(0,str(LOCAL_CUBIST))
LOCAL_ARROW = ROOT / "_deps/pyarrow_16_1_0"
if LOCAL_ARROW.exists():
    sys.path.insert(0,str(LOCAL_ARROW))

import joblib
import numpy as np
import pandas as pd
from cubist import Cubist
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from threadpoolctl import threadpool_limits,threadpool_info
from e1_controls_core import snv_copy,fit_pls1,scores,standardize_scores,memory_correction,metrics,paired_gain_ci

warnings.filterwarnings("ignore",message="X does not have valid feature names")


def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def json_save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def load_inputs(data,data_dir):
    if data_dir is None:
        data_dir=ROOT.parent/"data_v3" if data=="visnir" else ROOT/"data"
    xfile=data_dir/f"{data}_soc_X.npy"
    mfile=data_dir/f"{data}_soc_meta.parquet"
    if not xfile.exists() or not mfile.exists():
        raise FileNotFoundError(f"Missing restored input: {xfile} / {mfile}")
    x=np.load(xfile,mmap_mode="r"); meta=pd.read_parquet(mfile)
    if len(x)!=len(meta) or not {"soc","source","group_block","id.layer_uuid_txt"}.issubset(meta):
        raise ValueError("Input schema/count mismatch")
    if meta.duplicated(["source","id.layer_uuid_txt"]).any():
        raise ValueError("source+layer identities not unique")
    reference=ROOT/"results/V_visnir/summary/P1_all_predictions_log1p.parquet" if data=="visnir" else ROOT/"results/summary_stage2/all_predictions_log1p.parquet"
    return x,meta,xfile,mfile,reference


def saved_splits(meta,reference):
    table=pd.read_parquet(reference,columns=["row","fold","group","source","y_true"])
    if table.row.duplicated().any():
        raise ValueError("Saved split table repeats rows; inspect stage/columns rather than regenerate")
    if not np.array_equal(np.sort(table.row.to_numpy()),np.arange(len(meta))):
        raise ValueError("Saved fold rows do not cover current input exactly")
    table=table.sort_values("row")
    if not np.array_equal(meta.group_block.to_numpy(),table.group.to_numpy()):
        raise ValueError("Saved groups and restored metadata disagree")
    if not np.array_equal(meta.source.to_numpy(),table.source.to_numpy()):
        raise ValueError("Saved sources and restored metadata disagree")
    if not np.allclose(np.log1p(meta.soc.to_numpy(float)),table.y_true.to_numpy(),atol=1e-6):
        raise ValueError("Saved SOC labels disagree")
    if table.groupby("group").fold.nunique().max()!=1:
        raise ValueError("Saved outer split crosses independent groups")
    labels=table.fold.to_numpy()
    if np.issubdtype(labels.dtype,np.number):
        if not np.equal(labels,labels.astype(int)).all():
            raise ValueError("Saved fold labels are not integers")
        labels=np.array([f"fold{int(v)}" for v in labels])
    else:
        labels=labels.astype(str)
    if set(labels)!=set(f"fold{i}" for i in range(5)):
        raise ValueError(f"Unexpected saved five-fold labels: {np.unique(labels)}")
    return labels


def define_rows(args,meta,reference):
    if args.scenario.startswith("fold"):
        assignments=saved_splits(meta,reference)
        if args.scenario not in assignments:
            raise ValueError(f"Unknown saved scenario: {args.scenario}")
        train=np.where(assignments!=args.scenario)[0]; query=np.where(assignments==args.scenario)[0]
    else:
        sources={"visnir":{"Europe":["LUCAS.SSL","LUCAS.WOODWELL.SSL"],"ICRAF":["ICRAF.ISRIC"],"KSSL":["KSSL.SSL"]},
                 "mir":{"AfSIS":["AFSIS1.SSL","AFSIS2.SSL"],"ICRAF":["ICRAF.ISRIC"],"KSSL":["KSSL.SSL"],"LUCAS_Woodwell":["LUCAS.WOODWELL.SSL"],"SCHIEDUNG":["SCHIEDUNG.SSL"],"SERBIA":["SERBIA.SSL"]}}
        target=sources[args.data][args.scenario]; mask=meta.source.isin(target).to_numpy()
        train=np.where(~mask)[0]; query=np.where(mask)[0]
        # A source holdout can share spatial cells across sources; group cross-source overlap is recorded.
    if args.smoke:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=1042).split(train,groups=meta.group_block.to_numpy()[train]))
        rng=np.random.default_rng(1042)
        fit_pool,query_pool=train[a],train[b]
        train=np.sort(rng.choice(fit_pool,size=min(2000,len(fit_pool)),replace=False))
        query=np.sort(rng.choice(query_pool,size=min(400,len(query_pool)),replace=False))
    if len(train)<50 or len(query)<1:
        raise ValueError("Insufficient training/query support")
    if np.intersect1d(train,query).size:
        raise ValueError("Train/query identity overlap")
    overlap=set(meta.group_block.to_numpy()[train])&set(meta.group_block.to_numpy()[query])
    if (args.smoke or args.scenario.startswith("fold")) and overlap:
        raise ValueError("Independent groups cross train/query roles")
    return train,query,len(overlap)


def cubist_fit(t,y,committees,nn):
    return Cubist(n_rules=100,n_committees=int(committees),neighbors=None if nn==0 else int(nn),auto=False,random_state=42).fit(t,y)


def run(args):
    out=args.out_dir or ROOT/"results"/("E1_controls_smoke" if args.smoke else "E1_controls_v41")/args.data/args.scenario
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {out}")
    out.mkdir(parents=True)
    start=time.perf_counter()
    try:
        x,meta,xfile,mfile,reference=load_inputs(args.data,args.data_dir)
        train,query,overlap=define_rows(args,meta,reference)
        config_list=list(itertools.product([10,20] if args.smoke else [10,20,40],[1,5] if args.smoke else [1,5,10,15,20],[0,9] if args.smoke else [0,5,9]))
        info={"status":"prepared","start_utc":datetime.now(timezone.utc).isoformat(),"client_timezone":"Pacific/Auckland",
              "scope":"technical training-side smoke, no original outer-query results" if args.smoke else "follow-up on already analysed saved outer split",
              "data":args.data,"scenario":args.scenario,"n_train":len(train),"n_query":len(query),"train_groups":int(meta.group_block.iloc[train].nunique()),"query_groups":int(meta.group_block.iloc[query].nunique()),
              "cross_source_spatial_group_overlap":overlap,"smoke":args.smoke,"grid":config_list,"inner_group_folds":3,"selection":"pooled inner-validation log1p RMSE; ties: fewer components, committees, nn",
              "input_sha256":{"X":sha_file(xfile),"meta":sha_file(mfile),"reference":sha_file(reference) if reference.exists() else None},
              "code_sha256":{"runner":sha_file(Path(__file__)),"core":sha_file(Path(__file__).with_name("e1_controls_core.py"))},
              "python":sys.version,"platform":platform.platform(),"packages":{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","cubist","joblib","pyarrow"]},
              "threads":args.threads,"threadpools":threadpool_info(),"TabPFN_run":False,"pending_controls":["fairly tuned MBL","strict nested CNN-OOF stacking","OOF residual memory"]}
        np.savez_compressed(out/"split_manifest.npz",train=train,query=query)
        json_save(out/"run_info.json",info)
        if args.dry_run:
            info["status"]="dry_run_no_fits";json_save(out/"run_info.json",info);print(json.dumps({"status":info["status"],"n_train":len(train),"n_query":len(query)},ensure_ascii=False));return
        Xtr=snv_copy(x[train]); Xq=snv_copy(x[query]); del x
        ytr=np.log1p(meta.soc.to_numpy(float)[train]); yq=np.log1p(meta.soc.to_numpy(float)[query])
        groups=meta.group_block.to_numpy()[train]
        totals={c:{"sse":0.,"n":0,"failed":False} for c in config_list}
        attempts=[]; inner_manifest={}
        for inner,(a,b) in enumerate(GroupKFold(n_splits=3).split(Xtr,groups=groups)):
            if set(groups[a])&set(groups[b]):
                raise ValueError("Inner group overlap")
            inner_manifest[f"train{inner}"]=train[a];inner_manifest[f"validation{inner}"]=train[b]
            t=time.perf_counter();basis=fit_pls1(Xtr[a],ytr[a],max(20,max(c[0] for c in config_list)))
            print(f"{args.scenario} inner{inner}: PLS rank={basis['actual_components']} seconds={time.perf_counter()-t:.1f}",flush=True)
            cached={}
            for components in sorted({c[0] for c in config_list}):
                if components<=basis["actual_components"]:
                    ta,tb,_=standardize_scores(scores(Xtr[a],basis,components),scores(Xtr[b],basis,components))
                    cached[components]=(ta,tb)
            for components,committees,nn in config_list:
                key=(components,committees,nn); tick=time.perf_counter()
                record={"inner_fold":inner,"components":components,"committees":committees,"neighbors":nn,"n_validation":len(b)}
                try:
                    if components not in cached:
                        raise ValueError("Requested component count exceeds training rank")
                    ta,tb=cached[components]; model=cubist_fit(ta,ytr[a],committees,nn);p=model.predict(tb)
                    if not np.isfinite(p).all():
                        raise ValueError("Non-finite Cubist prediction")
                    sse=float(np.sum((p-ytr[b])**2));totals[key]["sse"]+=sse;totals[key]["n"]+=len(b)
                    record.update(status="ok",sse=sse,RMSE_log1p=float(np.sqrt(sse/len(b))))
                except Exception as error:
                    totals[key]["failed"]=True;record.update(status="failed",error=repr(error))
                record["seconds"]=time.perf_counter()-tick;attempts.append(record)
                pd.DataFrame(attempts).to_csv(out/"inner_attempts.csv",index=False)
            print(f"{args.scenario} inner{inner}: {len(config_list)} configurations completed",flush=True)
        np.savez_compressed(out/"inner_group_manifest.npz",**inner_manifest)
        choices=[{"components":c[0],"committees":c[1],"neighbors":c[2],"RMSE_inner_log1p":float(np.sqrt(v['sse']/v['n'])) if v['n'] and not v['failed'] else None,"all_inner_folds_ok":not v['failed'] and v['n']==len(train)} for c,v in totals.items()]
        pd.DataFrame(choices).to_csv(out/"inner_selection.csv",index=False)
        valid=[c for c in choices if c['all_inner_folds_ok']]
        if not valid:
            raise RuntimeError("All prescribed configurations failed; no replacement grid is selected")
        selected=min(valid,key=lambda c:(c['RMSE_inner_log1p'],c['components'],c['committees'],c['neighbors']))
        print(f"{args.scenario} selected {selected}",flush=True)
        t=time.perf_counter(); basis=fit_pls1(Xtr,ytr,max(20,selected['components']))
        ts,tq,sd_selected=standardize_scores(scores(Xtr,basis,selected['components']),scores(Xq,basis,selected['components']))
        selected_model=cubist_fit(ts,ytr,selected['committees'],selected['neighbors'])
        selected_q=selected_model.predict(tq);selected_tr=selected_model.predict(ts)
        uncorrected_model=selected_model if selected['neighbors']==0 else cubist_fit(ts,ytr,selected['committees'],0)
        uncorrected_q=uncorrected_model.predict(tq);uncorrected_tr=uncorrected_model.predict(ts)
        ztr,zq,sd=standardize_scores(scores(Xtr,basis,20),scores(Xq,basis,20))
        correction_selected=memory_correction(ztr,ytr-selected_tr,zq)
        correction_plain=memory_correction(ztr,ytr-uncorrected_tr,zq)
        predictions=pd.DataFrame({"row":query,"y_true":yq,"group":meta.group_block.to_numpy()[query],"source":meta.source.to_numpy()[query],
                                  "PLS_Cubist_selected":selected_q,"PLS_Cubist_selected+memory":selected_q+correction_selected,
                                  "PLS_Cubist_selected+bias":selected_q+(ytr-selected_tr).mean(),
                                  "PLS_Cubist_no_internal":uncorrected_q,"PLS_Cubist_no_internal+memory":uncorrected_q+correction_plain,
                                  "PLS_Cubist_generic_kNN_residual":uncorrected_q+correction_plain})
        if args.scenario.startswith('fold'):
            predictions['fold']=int(args.scenario[4:])
        if not args.smoke and args.scenario.startswith('fold') and reference.exists():
            old=pd.read_parquet(reference).set_index('row').loc[query]
            for name in ['OSSL_Cubist','CNN','CNN+memory','SoilICL_lite']:
                if name in old:
                    predictions['Frozen_'+name]=old[name].to_numpy()
            if 'CNN' in old and 'OSSL_Cubist' in old:
                predictions['Frozen_plain_average']=(old['CNN'].to_numpy()+old['OSSL_Cubist'].to_numpy())/2
                predictions['Selected_Cubist_CNN_average']=(selected_q+old['CNN'].to_numpy())/2
                predictions['Selected_Cubist_memory_CNN_memory_average']=(selected_q+correction_selected+old['CNN+memory'].to_numpy())/2
        model_columns=[c for c in predictions if c not in ['row','fold','group','source','y_true']]
        metric_rows=[]
        for name in model_columns:
            measured=metrics(yq,predictions[name].to_numpy())
            for scale,m in measured.items():
                metric_rows.append({'model':name,'scale':scale,'n':len(query),'groups':int(predictions.group.nunique()),**m})
        predictions.to_parquet(out/'predictions_log1p.parquet',index=False)
        pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False)
        contrasts=[('PLS_Cubist_selected+memory','PLS_Cubist_selected'),('PLS_Cubist_no_internal+memory','PLS_Cubist_no_internal')]
        if 'Frozen_SoilICL_lite' in predictions:
            contrasts += [('Frozen_SoilICL_lite','PLS_Cubist_selected'),('Frozen_SoilICL_lite','Frozen_plain_average')]
        cis=[]
        for method,ref in contrasts:
            for scale in ['log1p','native_wt_pct']:
                yy=yq if scale=='log1p' else np.expm1(yq)
                pm=predictions[method].to_numpy();pr=predictions[ref].to_numpy()
                if scale!='log1p':pm,pr=np.expm1(pm),np.expm1(pr)
                cis.append({'method':method,'reference':ref,'scale':scale,**paired_gain_ci(yy,pm,pr,predictions.group.to_numpy(),reps=2000)})
        pd.DataFrame(cis).to_csv(out/'paired_contrasts.csv',index=False)
        joblib.dump({'selected_model':selected_model,'no_internal_model':uncorrected_model,'basis':basis,'score_sd_selected':sd_selected,'score_sd_memory20':sd,'selected_config':selected,'training_residual_selected':ytr-selected_tr,'training_residual_no_internal':ytr-uncorrected_tr,'memory_z':ztr,'train_rows':train},out/'fitted_controls.joblib',compress=3)
        info.update(status='completed',selected=selected,n_inner_attempts=len(attempts),n_models_fit=sum(a['status']=='ok' for a in attempts)+(1 if selected['neighbors']==0 else 2),fitted_controls_seconds=time.perf_counter()-t,total_seconds=time.perf_counter()-start,
                    generic_residual_same_rule=True,model_columns=model_columns,original_model_changed=False,end_utc=datetime.now(timezone.utc).isoformat())
        json_save(out/'run_info.json',info)
        print(json.dumps({'status':info['status'],'scope':info['scope'],'n_train':len(train),'n_query':len(query),'selected':selected,'seconds':info['total_seconds']},ensure_ascii=False),flush=True)
    except Exception:
        (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',choices=['visnir','mir'],default='visnir')
    ap.add_argument('--scenario',required=True)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__=='__main__':
    main()
