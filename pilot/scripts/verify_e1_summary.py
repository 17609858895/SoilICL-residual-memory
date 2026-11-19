"""End-to-end identity/metric checks using synthetic temporary fixtures only."""
from contextlib import redirect_stdout
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('e1_summary',Path(__file__).with_name('18_e1_summary.py'))
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
import numpy as np
import pandas as pd


class SummaryChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'pilot';self.runs=self.root/'results/E1_controls_v41/visnir'
        self.output=self.root/'synthetic_summary'
        row=np.arange(200);truth=np.log1p(.2+row%13)
        old=pd.DataFrame({'row':row,'y_true':truth,'group':[f'g{i}' for i in row],
                          'source':[f'Source{i%4}' for i in row],'fold':[f'fold{i%5}' for i in row]})
        for name,shift in [('OSSL_Cubist',.2),('MBL',.18),('Cubist+memory',.15),('CNN',.1),('CNN+memory',.05),('SoilICL_lite',.05)]:
            old[name]=truth+shift
        oldpath=self.root/'results/V_visnir/summary/P1_all_predictions_log1p.parquet'
        oldpath.parent.mkdir(parents=True);old.to_parquet(oldpath,index=False)
        for fold in range(5):
            folder=self.runs/f'fold{fold}';folder.mkdir(parents=True)
            q=old[old.fold==f'fold{fold}'].copy();q['fold']=fold
            t=q.y_true.to_numpy()
            q=q[['row','y_true','group','source','fold']]
            for name,shift in [('PLS_Cubist_selected',.2),('PLS_Cubist_selected+memory',.15),('PLS_Cubist_no_internal',.2),
                               ('PLS_Cubist_no_internal+memory',.15),('PLS_Cubist_generic_kNN_residual',.15),
                               ('Frozen_plain_average',.15),('Selected_Cubist_CNN_average',.15),('Selected_Cubist_memory_CNN_memory_average',.1)]:
                q[name]=t+shift
            q.to_parquet(folder/'predictions_log1p.parquet',index=False)
            np.savez_compressed(folder/'split_manifest.npz',query=q.row.to_numpy(),train=old.loc[old.fold!=f'fold{fold}','row'].to_numpy())
            info={'status':'completed','smoke':False,'data':'visnir','scenario':f'fold{fold}',
                  'n_train':160,'total_seconds':0,'selected':{'components':20,'committees':5,'neighbors':9},
                  'code_sha256':{'runner':'synthetic-fixture-only'},'input_sha256':{'fixture':True}}
            (folder/'run_info.json').write_text(json.dumps(info),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def run_summary(self):
        argv=['18_e1_summary.py','--runs-dir',str(self.runs),'--out-dir',str(self.output)]
        with patch.object(summary,'ROOT',self.root),patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
            summary.main()

    def test_complete_aggregation_known_effect(self):
        self.run_summary()
        info=json.loads((self.output/'summary_info.json').read_text(encoding='utf-8'))
        self.assertEqual(info['completed_folds'],5);self.assertEqual(info['n_query'],200);self.assertEqual(info['groups'],200)
        ci=pd.read_csv(self.output/'paired_gain_ci.csv')
        row=ci[(ci.source=='pooled')&(ci.purpose=='primary_strong_Cubist')&(ci.scale=='log1p')].iloc[0]
        self.assertAlmostEqual(row.gain_pct,75.,places=10);self.assertAlmostEqual(row.ci_low,75.,places=10)
        self.assertFalse(info['main_model_promoted']);self.assertFalse(info['TabPFN_run'])

    def test_missing_fold_cannot_be_full_endpoint(self):
        (self.runs/'fold4/run_info.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'Full endpoint not ready'):self.run_summary()
        self.assertFalse(self.output.exists())

    def test_changed_query_row_rejected(self):
        path=self.runs/'fold0/predictions_log1p.parquet';p=pd.read_parquet(path);p.loc[0,'row']=1;p.to_parquet(path,index=False)
        with self.assertRaisesRegex(ValueError,'query identity'):self.run_summary()

    def test_generic_residual_identity_rejected_if_changed(self):
        path=self.runs/'fold0/predictions_log1p.parquet';p=pd.read_parquet(path);p.loc[0,'PLS_Cubist_generic_kNN_residual']+=.01;p.to_parquet(path,index=False)
        with self.assertRaisesRegex(ValueError,'not identical'):self.run_summary()


if __name__=='__main__':unittest.main(verbosity=2)
