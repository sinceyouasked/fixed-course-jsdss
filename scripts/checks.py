import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
D='data/'; rng=np.random.default_rng(5)
f36=pd.read_csv(D+'masters_fsa_first36_1934_2026.csv'); w=pd.read_csv(D+'fig_aug_weather.csv'); s=f36.merge(w[['year','tempP','windP','gustP','rhP','precP']],on='year',how='left')
m=s[s.year>=1983]
print('corr', {c:round(m.fsa.corr(m[c]),2) for c in ['tempP','windP','gustP','rhP','precP']})
print('mean 1983-2026', round(m.fsa.mean(),2), 'SD', round(m.fsa.std(),2), 'max |yoy|', round(m.fsa.diff().abs().max(),2))
r=sm.OLS(m.fsa,sm.add_constant(m[['tempP','windP','rhP']])).fit(cov_type='HAC',cov_kwds={'maxlags':2}); print('humidity model wind',round(r.params.windP,3),'p',round(r.pvalues.windP,2),'R2',round(r.rsquared,2))
c=m[m.year!=2020]; r=sm.OLS(c.fsa,sm.add_constant(c[['tempP','windP']])).fit(); print('drop2020',r.params.round(3).to_dict(),round(r.rsquared,2))
s48=s[s.year>=1948]; r36=sm.OLS(s48.first36,sm.add_constant(s48[['tempP','windP']])).fit(); res=pd.Series(r36.resid.values,index=s48.year.values)+s48.first36.mean()
for a,b in [(1948,1956),(1957,1980),(1981,2001),(2002,2018),(2019,2026)]: print('adj era',a,b,round(res.loc[a:b].mean(),2))
print('pre-1983 wind', round(s48[s48.year<1983].windP.mean()-s48[s48.year>=1983].windP.mean(),2), 'temp 2019+',round(m[m.year>=2019].tempP.mean(),1),'before',round(m[m.year<2019].tempP.mean(),1))
# 2019 dated contrast: sieve-bootstrap p
mA=sm.OLS(m.fsa,sm.add_constant(m[['tempP','windP']])).fit(); e=pd.Series(mA.resid.values,index=m.year.values); x=(e.index>=2019).astype(int)
b0=sm.OLS(e.values,sm.add_constant(x)).fit().params[1]; y=e.values-e.values.mean(); rho=np.sum(y[1:]*y[:-1])/np.sum(y[:-1]**2); inn=y[1:]-rho*y[:-1]; inn-=inn.mean(); n=len(y); bs=[]
for _ in range(5000):
    ys=np.zeros(n); ys[0]=y[rng.integers(n)]; eb=rng.choice(inn,n)
    for t in range(1,n): ys[t]=rho*ys[t-1]+eb[t]
    bs.append(sm.OLS(ys,sm.add_constant(x)).fit().params[1])
print('2019 contrast',round(b0,2),'sieve p',np.mean(np.abs(bs)>=abs(b0)),'iid SE',round(sm.OLS(e.values,sm.add_constant(x)).fit().bse[1],2))
# power with autocorrelation
sd=np.sqrt(mA.mse_resid); n1,n2=19,25; infl=np.sqrt((1+rho)/(1-rho)); print('MDE 80%', round(2.8*sd*np.sqrt(1/n1+1/n2)*infl,2), 'rho',round(rho,2),'resid sd',round(sd,2))
# panel: untouched holes around 2002 and 2006
d=pd.read_csv(D+'hole_year_panel_1983_2026.csv')
def untouched(y0,y1,pre0,pre1):
    ch=d[(d.year>=pre0)&(d.year<=y1)].groupby('hole').apply(lambda g:(g[['dy','green','trees_added','trees_removed','bunker_fairway']].nunique()>1).any())
    un=[h for h in ch.index if not ch[h]]; tch=[h for h in ch.index if ch[h]]
    pre=d[(d.year>=pre0)&(d.year<=pre1)]; post=d[(d.year>=y0)&(d.year<=y1)]
    du=post[post.hole.isin(un)].groupby('hole').rel.mean().sum()-pre[pre.hole.isin(un)].groupby('hole').rel.mean().sum()
    dt=post[post.hole.isin(tch)].groupby('hole').rel.mean().sum()-pre[pre.hole.isin(tch)].groupby('hole').rel.mean().sum()
    print(f'{y0}-{y1} vs {pre0}-{pre1}: untouched holes {un} sum change {du:+.2f}; changed holes sum change {dt:+.2f}; total {du+dt:+.2f}')
untouched(2002,2005,1998,2001); untouched(2006,2009,2002,2005); untouched(2002,2009,1995,2001)
e1=d[(d.year>=2010)&(d.year<=2018)].groupby('hole').rel.mean(); e2=d[d.year>=2019].groupby('hole').rel.mean(); dif=e2-e1; print('holes harder 2019-26 vs 2010-18:', list(dif[dif>0].index), dif[dif>0].round(2).to_dict())
b1=d[(d.year>=2010)&(d.year<=2018)]; b2=d[d.year>=2019]; print('bogey/rd', round((b1.bogey/b1.played).sum()/9*1,2), round((b2.bogey/b2.played).sum()/8,2), 'birdie/rd', round((b1.birdie/b1.played).sum()/9,2), round((b2.birdie/b2.played).sum()/8,2))
print('green alterations', d.groupby('hole').green.max().sum(), 'holes', (d.groupby('hole').green.max()>0).sum())
# stacked design: analytic cluster SE and wild-cluster p
ev=[]
for hh,g in d.sort_values('year').groupby('hole'):
    g=g.reset_index(drop=True)
    for i in range(1,len(g)):
        if g.dy[i]-g.dy[i-1]>=15: ev.append((hh,int(g.year[i]),int(g.dy[i]-g.dy[i-1])))
rows=[]
for i,(hh,yr,yd) in enumerate(ev):
    win=d[(d.year>=yr-6)&(d.year<=yr+6)].copy()
    chg=win.groupby('hole').apply(lambda g:(g[['dy','green','trees_added','trees_removed','bunker_fairway']].nunique()>1).any())
    ctrl=[c for c in chg.index if (not chg[c]) and c!=hh]; sub=win[win.hole.isin(ctrl+[hh])].copy(); sub['stk']=i
    sub['treat']=(sub.hole==hh).astype(float)*(sub.year>=yr)*yd/10; sub['st']=sub.year-yr; rows.append(sub)
S=pd.concat(rows); S['sh']=S.stk.astype(str)+'_'+S.hole.astype(str); S['sy']=S.stk.astype(str)+'_'+S.year.astype(str)
def wild(formula,data,term,B=999):
    mm=smf.ols(formula,data=data).fit(); X=np.asarray(mm.model.exog); j=mm.model.exog_names.index(term); yv=np.asarray(mm.model.endog)
    Xr=np.delete(X,j,axis=1); br=np.linalg.lstsq(Xr,yv,rcond=None)[0]; ur=yv-Xr@br; fr=Xr@br
    cl=data.hole.values; cls=np.unique(cl); masks=[cl==c for c in cls]; XtXi=np.linalg.pinv(X.T@X); G=len(cls); n,k=X.shape
    def ct(y):
        b=XtXi@(X.T@y); e=y-X@b; V=np.zeros((k,k))
        for mk in masks: sc=X[mk].T@e[mk]; V+=np.outer(sc,sc)
        V=XtXi@V@XtXi*G/(G-1); return b[j]/np.sqrt(V[j,j]), b[j], np.sqrt(V[j,j])
    t0,b,se=ct(yv); ts=[]
    for _ in range(B):
        wv=rng.choice([-1.,1.],G); ys=fr+ur*np.array([wv[np.searchsorted(cls,c)] for c in cl]); ts.append(ct(ys)[0])
    return b,se,np.mean(np.abs(ts)>=abs(t0))
for lab,f in [('stacked per 10','rel ~ treat + C(sh) + C(sy)'),('stacked + stack-hole trends','rel ~ treat + C(sh) + C(sy) + C(sh):st')]:
    b,se,p=wild(f,S,'treat'); print(lab, round(b,4),'cluster SE (no df corr)',round(se,4),'wild p',round(p,3))
