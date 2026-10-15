"""Reanalyse archived results; never regenerate or relabel source experiments.
Run with Python and requirements_analysis.txt. Outputs are local revision assets.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json, time, sys, platform, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision_paths import CODE as HERE, ROOT, RESULTS as OUT, PACKAGE
CORPUS = ROOT
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

def frame():
    raw = pd.read_csv(CORPUS/'results/raw_results.csv')
    valid = raw[raw.error.isna()] if 'error' in raw else raw
    keys = ['dataset','clf','sampler','seed']
    p = valid[valid.condition.isin(['leaked','correct'])].pivot_table(index=keys, columns='condition', values='f1').dropna().reset_index()
    p['pair_id'] = np.arange(len(p))
    man = pd.read_csv(CORPUS/'data/processed/manifest.csv')
    p = p.merge(man, left_on='dataset', right_on='name', validate='many_to_one')
    d = pd.concat([p.assign(F1=p.leaked, label=1), p.assign(F1=p.correct, label=0)], ignore_index=True)
    X = pd.DataFrame({'F1':d.F1,'log_ir':np.log10(d.imbalance_ratio),'log_nmin':np.log10(d.n_minority),'log_p':np.log10(d.n_features)})
    X = pd.concat([X,pd.get_dummies(d.clf,prefix='clf').astype(float)],axis=1)
    assert len(p)==11200 and len(d)==22400 and d.dataset.nunique()==41
    assert (d.groupby('pair_id').label.sum()==1).all()
    return raw,p,d,X

def fold(arg):
    ds,d,X = arg
    y=d.label.to_numpy(); te=np.flatnonzero(d.dataset==ds); tr=np.flatnonzero(d.dataset!=ds)
    assert set(d.iloc[tr].pair_id).isdisjoint(d.iloc[te].pair_id)
    specs={'Full':list(X.columns),'F1 only':['F1'],'Descriptors':[c for c in X if c!='F1'],'F1 and descriptors':list(X.columns[:4]),'Logistic':list(X.columns)}
    pred={}; pert={}; elapsed=0
    for name,cols in specs.items():
        m=LogisticRegression(max_iter=2000) if name=='Logistic' else GradientBoostingClassifier(random_state=0)
        t=time.perf_counter(); m.fit(X.iloc[tr][cols],y[tr]); dt=time.perf_counter()-t
        pred[name]=m.predict_proba(X.iloc[te][cols])[:,1]
        if name=='Full':
            elapsed=dt
            for delta in [-.05,-.01,.01,.05]:
                z=X.iloc[te].copy(); z['F1']=np.clip(z.F1+delta,0,1)
                pert[f'F1 {delta:+.2f}']=m.predict_proba(z)[:,1]
            for fac in [.5,2]:
                z=X.iloc[te].copy(); z['log_p']+=np.log10(fac)
                pert[f'Features x{fac:g}']=m.predict_proba(z)[:,1]
            for fam in [c for c in X if c.startswith('clf_')]:
                z=X.iloc[te].copy(); z.loc[:,[c for c in X if c.startswith('clf_')]]=0; z[fam]=1
                pert[f'Family {fam[4:]}']=m.predict_proba(z)[:,1]
    return ds,te,pred,pert,elapsed

def metrics(y,s,t=.5,w=None):
    w=np.ones(len(y)) if w is None else w
    pos=y==1; neg=~pos; flag=s>=t
    return {'auc':float(roc_auc_score(y,s,sample_weight=w)), 'ap':float(average_precision_score(y,s,sample_weight=w)), 'brier':float(np.average((s-y)**2,weights=w)), 'sensitivity':float(w[pos&flag].sum()/w[pos].sum()),'specificity':float(w[neg&~flag].sum()/w[neg].sum())}

def main():
    raw,p,d,X=frame(); y=d.label.to_numpy(); groups=d.dataset.to_numpy(); datasets=np.unique(groups)
    cache=OUT/'oof_predictions.csv'
    times=[]
    if (OUT/'summary.json').exists():
        times=json.loads((OUT/'summary.json').read_text()).get('fold_fit_seconds',[])
    if cache.exists():
        q=pd.read_csv(cache)
        expected=d[['dataset','clf','sampler','seed','pair_id','label','F1']].reset_index(drop=True)
        pd.testing.assert_frame_equal(q[expected.columns],expected,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        pred={c:q[c].to_numpy() for c in ['Full','F1 only','Descriptors','F1 and descriptors','Logistic']}
        pert={c[5:]:q[c].to_numpy() for c in q if c.startswith('pert:')}
    else:
        pred={k:np.zeros(len(y)) for k in ['Full','F1 only','Descriptors','F1 and descriptors','Logistic']}; pert={}
        times=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for ds,te,pr,pe,dt in pool.map(fold,[(ds,d,X) for ds in datasets]):
                print('Completed',ds,flush=True); times.append(dt)
                for k,v in pr.items(): pred[k][te]=v
                for k,v in pe.items(): pert.setdefault(k,np.zeros(len(y)))[te]=v
        q=d[['dataset','clf','sampler','seed','pair_id','label','F1']].copy()
        for k,v in pred.items():q[k]=v
        for k,v in pert.items():q['pert:'+k]=v
        q.to_csv(cache,index=False)
    pred['Raw F1 rule']=d.F1.to_numpy()
    base=pd.DataFrame([dict(model=k,**metrics(y,v)) for k,v in pred.items()]); base.to_csv(OUT/'baselines.csv',index=False)
    s=pred['Full']
    pd.DataFrame([dict(dataset=ds,n=int((groups==ds).sum()),**metrics(y[groups==ds],s[groups==ds])) for ds in datasets]).to_csv(OUT/'per_dataset.csv',index=False)
    pd.DataFrame([dict(sampler=k,**metrics(y[d.sampler==k],s[d.sampler==k])) for k in sorted(d.sampler.unique())]).to_csv(OUT/'per_sampler.csv',index=False)
    thresholds=pd.DataFrame([dict(threshold=t,flagged=float((s>=t).mean()),**metrics(y,s,t)) for t in [.2,.3,.35,.4,.5,.6,.65,.7,.8]])
    thresholds.to_csv(OUT/'thresholds.csv',index=False)
    pp=[]
    for t in [.35,.5,.65]:
        m=metrics(y,s,t); se=m['sensitivity']; sp=m['specificity']
        for pi in [.01,.05,.1,.2,.5]:
            pp.append(dict(threshold=t,prevalence=pi,ppv=pi*se/(pi*se+(1-pi)*(1-sp)),npv=(1-pi)*sp/((1-pi)*sp+pi*(1-se))))
    pd.DataFrame(pp).to_csv(OUT/'predictive_value_scenarios.csv',index=False)
    bands=lambda a:np.digitize(a,[.35,.65])
    sens=[]
    for k,v in pert.items():
        sens.append(dict(perturbation=k,mean_absolute_change=float(np.abs(v-s).mean()),band_change=float((bands(v)!=bands(s)).mean()),**metrics(y,v)))
    pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False)
    error=[]
    for field in ['dataset','clf','sampler']:
        for value in sorted(d[field].unique()):
            z=d[field]==value; yy=y[z]; ss=s[z]
            error.append(dict(group=field,value=value,n=int(z.sum()),false_positive_rate=float((ss[yy==0]>=.65).mean()),miss_rate=float((ss[yy==1]<.65).mean())))
    pd.DataFrame(error).to_csv(OUT/'errors.csv',index=False)
    # Cluster percentile bootstrap of FIXED out-of-fold predictions. Repeats keep all paired records.
    rng=np.random.default_rng(20261015); gi=np.searchsorted(datasets,groups); samples=[]
    for b in range(1000):
        counts=np.bincount(rng.integers(0,41,41),minlength=41); w=counts[gi]
        m=metrics(y,s,w=w)
        m['auc_difference_logistic']=m['auc']-roc_auc_score(y,pred['Logistic'],sample_weight=w)
        samples.append(m)
    boot=pd.DataFrame(samples); boot.to_csv(OUT/'cluster_bootstrap.csv',index=False)
    ci={c:[float(v) for v in np.quantile(boot[c],[.025,.975])] for c in boot}
    final=GradientBoostingClassifier(random_state=0); start=time.perf_counter(); final.fit(X,y); fit_time=time.perf_counter()-start
    sys.path.insert(0,str(PACKAGE/'src')); import leakcheck
    cols=leakcheck.MODEL.columns
    numpy_scores=leakcheck.MODEL.predict_proba(X[cols].to_numpy())
    parity=float(np.max(np.abs(numpy_scores-final.predict_proba(X)[:,1])))
    batch=X[cols].to_numpy(); bench=[]
    for _ in range(5):
        start=time.perf_counter(); leakcheck.MODEL.predict_proba(batch); bench.append(time.perf_counter()-start)
    import psutil
    info={'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'ram_gib':psutil.virtual_memory().total/2**30,'full_fit_seconds':fit_time,'fold_fit_seconds':times,'numpy_batch_median_seconds':float(np.median(bench)),'batch_records':len(X),'input_matrix_bytes':int(batch.nbytes),'model_bytes':(PACKAGE/'src/leakcheck/model.json').stat().st_size,'parity_max_error':parity,'ci':ci,'attempted_rows':len(raw),'failed_rows':int(raw.error.notna().sum()),'pairs':len(p),'group_macro_auc':float(pd.read_csv(OUT/'per_dataset.csv').auc.mean()),'hashes':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in [CORPUS/'results/raw_results.csv',CORPUS/'data/processed/manifest.csv',PACKAGE/'src/leakcheck/model.json']}}
    (OUT/'summary.json').write_text(json.dumps(info,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for name in ['Full','F1 only','F1 and descriptors','Logistic']:
        a,b,_=roc_curve(y,pred[name]);axes[0].plot(a,b,label=name)
        a,b,_=precision_recall_curve(y,pred[name]);axes[1].plot(b,a,label=name)
    axes[0].plot([0,1],[0,1],'k:',lw=.8);axes[0].set(xlabel='False positive rate',ylabel='True positive rate')
    axes[1].set(xlabel='Recall',ylabel='Precision');axes[1].legend(fontsize=7)
    bins=pd.qcut(pd.Series(s),10,duplicates='drop'); rel=pd.DataFrame({'score':s,'label':y,'bin':bins}).groupby('bin',observed=True).agg(mean_score=('score','mean'),fraction_positive=('label','mean'),n=('label','size')).reset_index(drop=True)
    rel.to_csv(OUT/'reliability.csv',index=False)
    axes[2].plot(rel.mean_score,rel.fraction_positive,'o-');axes[2].plot([0,1],[0,1],'k:');axes[2].set(xlabel='Mean screening score',ylabel='Controlled positive fraction')
    fig.tight_layout();fig.savefig(OUT/'validation.pdf');plt.close(fig)
    import shap
    idx=np.random.default_rng(0).choice(len(X),min(2000,len(X)),replace=False); sample=X.iloc[idx]
    sv=shap.TreeExplainer(final)(sample)
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,col in zip(axes,['F1','log_ir','log_p']):
        j=X.columns.get_loc(col); sc=ax.scatter(sample[col],sv.values[:,j],c=sample.F1,s=4,cmap='viridis',alpha=.5);ax.axhline(0,color='gray',lw=.6);ax.set(xlabel=col,ylabel='SHAP contribution to log odds')
    fig.tight_layout();fig.savefig(OUT/'shap_dependence.pdf');plt.close(fig)
    print(base.to_string(index=False));print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
