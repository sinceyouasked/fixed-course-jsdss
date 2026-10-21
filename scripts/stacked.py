import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
D='data/'; d=pd.read_csv(D+'hole_year_panel_1983_2026.csv'); d['y']=d.year.astype(str); d['h']=d.hole.astype(str)
ev=[]
for hh,g in d.sort_values('year').groupby('hole'):
    g=g.reset_index(drop=True)
    for i in range(1,len(g)):
        if g.dy[i]-g.dy[i-1]>=15: ev.append((hh,int(g.year[i]),int(g.dy[i]-g.dy[i-1])))
rows=[]
for i,(hh,yr,yd) in enumerate(ev):
    win=d[(d.year>=yr-6)&(d.year<=yr+6)].copy()
    chg=win.groupby('hole').apply(lambda g: (g[['dy','green','trees_added','trees_removed','bunker_fairway']].nunique()>1).any())
    ctrl=[c for c in chg.index if (not chg[c]) and c!=hh]
    sub=win[win.hole.isin(ctrl+[hh])].copy(); sub['stk']=i; sub['post']=(sub.year>=yr).astype(float)
    sub['treat']=((sub.hole==hh)).astype(float)*sub.post*yd/10; sub['treatb']=(sub.hole==hh).astype(float)*sub.post
    sub['st']=sub.year-yr; rows.append(sub)
S=pd.concat(rows); S['sh']=S.stk.astype(str)+'_'+S.h; S['sy']=S.stk.astype(str)+'_'+S.y
rng=np.random.default_rng(7); holes=np.arange(1,19)
def est(df,f,term): return smf.ols(f,data=df).fit().params[term]
for lab,f,term in [('per 10 yds','rel ~ treat + C(sh) + C(sy)','treat'),('binary','rel ~ treatb + C(sh) + C(sy)','treatb'),('per 10 yds + stack-hole trends','rel ~ treat + C(sh) + C(sy) + C(sh):st','treat')]:
    b=est(S,f,term); bs=[]
    for r in range(400):
        pick=rng.choice(holes,18,replace=True); df=pd.concat([S[S.hole==h].assign(h=f'{h}_{k}',sh=lambda x,k=k: x.sh+f'_{k}') for k,h in enumerate(pick)])
        try: bs.append(est(df,f,term))
        except Exception: pass
    bs=np.array(bs); print(f'{lab}: {b:+.4f}; hole-cluster bootstrap SE {bs.std():.4f}; 95% CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] ({len(bs)} draws)',flush=True)
