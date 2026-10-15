"""Create interface illustrations from the shipped JSON; never fit a model."""
import hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from revision_paths import ROOT,PACKAGE,RESULTS,MANUSCRIPT,SUBMISSION

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    model=PACKAGE/'src/leakcheck/model.json'
    before=hashlib.sha256(model.read_bytes()).hexdigest()
    sys.path.insert(0,str(PACKAGE/'src'))
    import leakcheck
    fig,ax=plt.subplots(figsize=(8.4,2.7));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc='#f5f7f9',ec='#456078'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    box(.15,.35,3.1,2.3,'Reported minority-class F1\nImbalance ratio\nMinority count\nEncoded predictor count\nSupported model family')
    box(3.9,.8,2.05,1.4,'leakcheck package\nJSON model\nNumPy inference')
    box(6.6,.45,3.1,2.1,'Screening score\nLow / intermediate / high band\nMethods-check recommendation\nInput and extrapolation notes')
    for x1,x2 in [(3.35,3.8),(6.05,6.5)]:ax.add_patch(FancyArrowPatch((x1,1.5),(x2,1.5),arrowstyle='-|>',mutation_scale=14,color='#456078'))
    fig.tight_layout();fig.savefig(RESULTS/'user_workflow.pdf',bbox_inches='tight');plt.close(fig)
    rows=[]
    for ir in [2,10,50]:
        for f1 in [.45,.70,.92]:
            r=leakcheck.check(f1=f1,imbalance_ratio=ir,n_minority=200,n_features=30,clf='RandomForest')
            rows.append({'imbalance_ratio':ir,'f1':f1,'n_minority':200,'n_features':30,'clf':'RandomForest','score':r.score,'band':r.verdict})
    pd.DataFrame(rows).to_csv(RESULTS/'interface_grid.csv',index=False)
    # Reuse the original density illustration without altering its underlying data.
    original=MANUSCRIPT/'figures/image2.png'
    density=RESULTS/'f1_density.png'
    if original.exists():shutil.copy2(original,density)
    elif not density.exists():raise FileNotFoundError('The original F1 density image must accompany this script.')
    if MANUSCRIPT.exists():
        dst=SUBMISSION/'analysis';dst.mkdir(parents=True,exist_ok=True)
        for name in ['user_workflow.pdf','f1_density.png']:shutil.copy2(RESULTS/name,dst/name)
    assert hashlib.sha256(model.read_bytes()).hexdigest()==before
    (RESULTS/'interface_assets_provenance.json').write_text(json.dumps({'json_sha256':before,'model_refitted':False,'density_sha256':hashlib.sha256(density.read_bytes()).hexdigest(),'grid_source':'shipped JSON via leakcheck.check','grid_fixed_inputs':{'n_minority':200,'n_features':30,'clf':'RandomForest'}},indent=2))
    print('Interface workflow and 9 JSON-only grid scores generated; model unchanged.')

if __name__=='__main__':main()
