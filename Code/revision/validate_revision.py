"""Check corpus identities, saved metrics and package parity without retraining."""
import hashlib,json,sys
import numpy as np
import pandas as pd
from revision_paths import ROOT,PACKAGE,RESULTS
from analyse_revision import frame,metrics

def main():
    raw,pairs,d,X=frame()
    q=pd.read_csv(RESULTS/'oof_predictions.csv')
    columns=['dataset','clf','sampler','seed','pair_id','label','F1']
    pd.testing.assert_frame_equal(q[columns],d[columns].reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    assert (q.groupby('pair_id').size()==2).all()
    assert (q.groupby('pair_id').dataset.nunique()==1).all()
    assert (q.groupby('pair_id').label.sum()==1).all()
    expected=pd.read_csv(RESULTS/'baselines.csv').set_index('model')
    for name in ['Full','F1 only','Descriptors','F1 and descriptors','Logistic','Raw F1 rule']:
        got=metrics(q.label.to_numpy(),q.F1.to_numpy() if name=='Raw F1 rule' else q[name].to_numpy())
        for key,value in got.items():assert np.isclose(value,expected.loc[name,key],atol=1e-12)
    summary=json.loads((RESULTS/'summary.json').read_text())
    for relative,checksum in summary['hashes'].items():
        f=ROOT/relative
        assert f.is_file(),f
        assert hashlib.sha256(f.read_bytes()).hexdigest()==checksum,f
    sys.path.insert(0,str(PACKAGE/'src'))
    import leakcheck
    assert leakcheck.__version__=='1.0.1'
    assert leakcheck.selftest()
    assert leakcheck.check(.92,5.2,237,44,'RF').score==leakcheck.check(.92,5.2,237,44,'RF').probability
    for name in ['training_flow.pdf','validation.pdf','shap_dependence.pdf']:
        assert (RESULTS/name).read_bytes().startswith(b'%PDF')
    print(json.dumps({'root':str(ROOT),'package':str(PACKAGE),'records':len(q),'pairs':len(pairs),'datasets':q.dataset.nunique(),'auc':expected.loc['Full','auc'],'checks':'passed'},indent=2))

if __name__=='__main__':main()
