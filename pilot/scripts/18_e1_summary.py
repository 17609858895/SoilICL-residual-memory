"""Aggregate completed E1 spatial folds, checking identities against frozen results.

Default requires all five VisNIR folds. --allow-partial explicitly reports a
partial comparison, never a substitute for the full primary endpoint. No fitting,
model selection, clipping or replacement of original prediction files occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'_deps/pyarrow_16_1_0'))
import numpy as np
import pandas as pd
from e1_controls_core import metrics,paired_gain_ci


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs-dir',type=Path,default=ROOT/'results/E1_controls_v41/visnir')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Preserve earlier summaries; use a new output directory.')
    oldpath=ROOT/'results/V_visnir/summary/P1_all_predictions_log1p.parquet'
    old=pd.read_parquet(oldpath).set_index('row')
    tables=[];configs=[];input_hashes={};states={}
    for fold in range(5):
        folder=args.runs_dir/f'fold{fold}'
        path=folder/'run_info.json'
        if not path.exists():
            states[f'fold{fold}']='not_started';continue
        info=json.loads(path.read_text(encoding='utf-8'))
        states[f'fold{fold}']=info['status']
        if info['status']!='completed':continue
        if info['smoke'] or info['data']!='visnir' or info['scenario']!=f'fold{fold}':
            raise ValueError('Incompatible run scope')
        prediction_path=folder/'predictions_log1p.parquet'
        p=pd.read_parquet(prediction_path)
        expected=old.index[old.fold==f'fold{fold}'].to_numpy()
        if p.row.duplicated().any() or not np.array_equal(np.sort(p.row.to_numpy()),np.sort(expected)):
            raise ValueError(f'Incomplete/changed original query identity in fold{fold}')
        reference=old.loc[p.row.to_numpy()]
        if not np.allclose(p.y_true,reference.y_true,atol=1e-6):raise ValueError('Labels disagree')
        if not np.array_equal(p.group,reference.group) or not np.array_equal(p.source,reference.source):raise ValueError('Metadata disagree')
        manifest=np.load(folder/'split_manifest.npz')
        if not np.array_equal(manifest['query'],p.row.to_numpy()):raise ValueError('Manifest mismatch')
        for model in ['OSSL_Cubist','MBL','Cubist+memory','CNN','CNN+memory','SoilICL_lite']:
            target='Frozen_'+model
            if target in p and not np.array_equal(p[target].to_numpy(),reference[model].to_numpy()):
                raise ValueError(f'Frozen model changed: {model}')
            p[target]=reference[model].to_numpy()
        if not np.array_equal(p['PLS_Cubist_no_internal+memory'],p['PLS_Cubist_generic_kNN_residual']):
            raise ValueError('Expected identical generic residual correction is not identical')
        tables.append(p)
        configs.append({'fold':f'fold{fold}','n_query':len(p),'n_train':info['n_train'],
                        'total_seconds':info['total_seconds'],**info['selected']})
        input_hashes[f'fold{fold}']={'predictions':sha(prediction_path),'run_info':sha(path),
                                  'runner_code':info['code_sha256'],'inputs':info['input_sha256']}
    if not tables:raise RuntimeError('No completed formal folds; smoke is excluded.')
    if len(tables)<5 and not args.allow_partial:raise RuntimeError(f'Full endpoint not ready: {states}')
    p=pd.concat(tables,ignore_index=True).sort_values('row').reset_index(drop=True)
    if p.row.duplicated().any() or p.groupby('group').fold.nunique().max()!=1:raise ValueError('Pooled identity/group mismatch')
    model_columns=[c for c in p if c not in ['row','fold','y_true','group','source']]
    contrasts=[('Frozen_SoilICL_lite','PLS_Cubist_selected','primary_strong_Cubist'),
               ('PLS_Cubist_selected+memory','PLS_Cubist_selected','memory_after_internal_correction'),
               ('PLS_Cubist_no_internal+memory','PLS_Cubist_no_internal','memory_rule_only'),
               ('Frozen_SoilICL_lite','Frozen_plain_average','memory_beyond_original_average'),
               ('PLS_Cubist_selected','Frozen_OSSL_Cubist','tuned_backbone_vs_historical'),
               ('Selected_Cubist_memory_CNN_memory_average','Selected_Cubist_CNN_average','memory_beyond_strong_average'),
               ('Selected_Cubist_memory_CNN_memory_average','Frozen_SoilICL_lite','prespecified_candidate_not_promoted')]
    metric_rows=[];ci_rows=[]
    for source,part in [('pooled',p)]+list(p.groupby('source',sort=True)):
        y=part.y_true.to_numpy()
        for model in model_columns:
            for scale,m in metrics(y,part[model].to_numpy()).items():
                metric_rows.append({'source':source,'model':model,'scale':scale,'n':len(part),'groups':part.group.nunique(),**m})
        for method,reference,purpose in contrasts:
            for scale in ['log1p','native_wt_pct']:
                yy=y;pm=part[method].to_numpy();pr=part[reference].to_numpy()
                if scale=='native_wt_pct':yy,pm,pr=np.expm1(yy),np.expm1(pm),np.expm1(pr)
                ci_rows.append({'source':source,'method':method,'reference':reference,'purpose':purpose,'scale':scale,
                                **paired_gain_ci(yy,pm,pr,part.group.to_numpy(),reps=2000)})
    args.out_dir.mkdir(parents=True)
    p.to_parquet(args.out_dir/'matched_predictions_log1p.parquet',index=False)
    measured=pd.DataFrame(metric_rows);paired=pd.DataFrame(ci_rows)
    measured.to_csv(args.out_dir/'metrics_by_source.csv',index=False)
    paired.to_csv(args.out_dir/'paired_gain_ci.csv',index=False)
    pd.DataFrame(configs).to_csv(args.out_dir/'selected_training_configs.csv',index=False)
    status={'created_utc':datetime.now(timezone.utc).isoformat(),'client_timezone':'Pacific/Auckland','completed_folds':len(tables),'fold_states':states,
            'endpoint':'complete five-fold follow-up' if len(tables)==5 else 'partial; full five-fold endpoint pending',
            'n_query':len(p),'groups':int(p.group.nunique()),'original_prediction_sha256':sha(oldpath),'run_sha256':input_hashes,
            'generic_kNN_rule_exactly_equal':True,'TabPFN_run':False,'main_model_promoted':False,
            'ci_scope':'paired independent-group resampling of fixed predictions; marginal 95% CI, no multiplicity-adjusted significance or model-refitting uncertainty',
            'pending':['tuned MBL','strict nested CNN-OOF stacking','cross-fitted residual sensitivity','MIR/source E1 runs','E2 missing source/adaptation','E3 coverage interventions']}
    (args.out_dir/'summary_info.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    selected_names=['Frozen_OSSL_Cubist','Frozen_MBL','Frozen_CNN','Frozen_plain_average','Frozen_SoilICL_lite','PLS_Cubist_selected',
                    'PLS_Cubist_selected+memory','Selected_Cubist_CNN_average','Selected_Cubist_memory_CNN_memory_average']
    report=['# E1 强 Cubist 对照：实际完成结果','',f"完成 {len(tables)}/5 折，{len(p):,} 个原测试土样，{p.group.nunique():,} 个独立空间组。",
            '完整端点已完成。' if len(tables)==5 else '**这是部分折结果，不能替代全五折主结论。**','',
            '新 Cubist 在每折原训练部分，以三折分组内层验证选择 PLS 组分/committees/实例校正；45组参数没有按测试成绩删改。原 CNN、原 LRM 和历史 MBL 使用保存的同折预测。TabPFN 未运行。',
            '','| 模型 | RMSE log1p | RMSE wt% | R² 原尺度 |','|---|---:|---:|---:|']
    for model in selected_names:
        q=measured[(measured.source=='pooled')&(measured.model==model)].set_index('scale')
        report.append(f"| {model} | {q.loc['log1p','RMSE']:.4f} | {q.loc['native_wt_pct','RMSE']:.4f} | {q.loc['native_wt_pct','R2']:.4f} |")
    report+=['','正值表示前一模型的 RMSE 更低。以下为条件于固定预测的边际95%组bootstrap CI。','',
             '| 比较 | log1p 改善%（95% CI） | 原尺度改善%（95% CI） |','|---|---:|---:|']
    for method,reference,purpose in contrasts:
        q=paired[(paired.source=='pooled')&(paired.method==method)&(paired.reference==reference)].set_index('scale')
        cells=[]
        for scale in ['log1p','native_wt_pct']:
            r=q.loc[scale];cells.append(f"{r.gain_pct:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
        report.append(f"| {method} vs {reference} | {' | '.join(cells)} |")
    report+=['','普通kNN残差对照与同空间、同k、同权重的外置残差记忆逐样本完全相等。这是规则身份核验，不构成另一个创新模型。',
             '','预定的新 Cubist/CNN 组合是候选版本；当前主模型仍为原 SoilICL-lite。看过本轮结果后改变主模型，需要新增未消费评价证据。',
             '','仍需完成公平调优MBL、严格嵌套CNN-OOF stacking和交叉拟合残差敏感性，才能完成全部E1c；MIR、来源迁移和覆盖实验不能用本次空间五折替代。',
             '','已知原结果的本地后续评价，不称全新盲验证；Windows环境与原云环境差异保留在各折manifest。所有原文件保持。']
    (args.out_dir/'实际结果说明.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:status[k] for k in ['completed_folds','endpoint','n_query','groups','TabPFN_run']},ensure_ascii=False))


if __name__=='__main__':main()
