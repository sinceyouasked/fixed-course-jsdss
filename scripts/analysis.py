"""JQAS manuscript: re-inference at the correct level.
Series, weather models, breaks (sup-F with AR(1) sieve null), era contrasts (HAC), amplitude,
hole-year panel (TWFE, hole trends, event study, stacked, wild-cluster, randomisation inference, Holm),
Pebble station contrast with n stated.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
import json, sys
D='data/'
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); out.append(s); open('scripts/analysis_results.txt','a').write(s+'\n')
rng=np.random.default_rng(20261020)
open('scripts/analysis_results.txt','w').close()
import warnings; warnings.filterwarnings('ignore')

# ---------------- series
f36=pd.read_csv(D+'masters_fsa_first36_1934_2026.csv')           # year,fsa,first36
w=pd.read_csv(D+'fig_aug_weather.csv')  # corrected playing-hours windows, 1948-2026 (the deposit's weather_augusta_hourly_windows.csv is a stale build)
s=f36.merge(w[['year','tempP','windP','gustP','rhP','precP']],on='year',how='left')
s['era']=pd.cut(s.year,[1933,1942,1956,1980,2001,2018,2026],labels=['1934-42','1946-56','1957-80','1981-2001','2002-18','2019-26'])
P('=== 1. Ninety Masters: era means (headline and first-36) ===')
g=s.groupby('era',observed=True).agg(n=('fsa','size'),fsa=('fsa','mean'),f36=('first36','mean'),sd=('fsa','std'),sd36=('first36','std'))
P(g.round(2).to_string())
P(f'all: n={len(s)} headline {s.fsa.mean():.2f} (SD {s.fsa.std():.2f}); first36 {s.first36.mean():.2f} (SD {s.first36.std():.2f})')

def hac_ols(y,X,maxlags=2):
    X=sm.add_constant(X); m=sm.OLS(y,X,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':maxlags}); return m

# ---------------- weather models
P('\n=== 2. Weather models (playing-hours temperature and wind; Newey-West HAC) ===')
m83=s[s.year>=1983]
for lab,sub,ycol in [('1983-2026 headline',m83,'fsa'),('1948-2026 headline',s[s.year>=1948],'fsa'),('1948-2026 first36',s[s.year>=1948],'first36'),('1983-2026 first36',m83,'first36')]:
    for L in [0,2,4]:
        m=hac_ols(sub[ycol],sub[['tempP','windP']],maxlags=L)
        if L==2: keep=m
        P(f'{lab} lags={L}: const {m.params.const:+.2f}; temp {m.params.tempP:+.4f} (SE {m.bse.tempP:.4f}, p {m.pvalues.tempP:.3f}); wind {m.params.windP:+.4f} (SE {m.bse.windP:.4f}, p {m.pvalues.windP:.3f}); R2 {m.rsquared:.2f}; resid SD {np.sqrt(m.mse_resid):.2f}; n {int(m.nobs)}')
    r=keep.resid; P(f'   residual lag-1 autocorrelation {pd.Series(r.values).autocorr():.2f}; DW {sm.stats.durbin_watson(r):.2f}')
# era-stability of weather coefficients (interaction test), 1948-2026 headline
sub=s[s.year>=1948].copy(); sub['post']=(sub.year>=1983).astype(int)
m=smf.ols('fsa ~ tempP*post + windP*post',data=sub).fit(cov_type='HAC',cov_kwds={'maxlags':2})
P(f'era interaction 1948-82 vs 1983-2026: temp x post {m.params["tempP:post"]:+.4f} (p {m.pvalues["tempP:post"]:.2f}); wind x post {m.params["windP:post"]:+.4f} (p {m.pvalues["windP:post"]:.2f})')
# weather-adjusted residuals
mA=hac_ols(m83.fsa,m83[['tempP','windP']]); res83=pd.Series(mA.resid.values,index=m83.year.values)
s48=s[s.year>=1948]; mB=hac_ols(s48.first36,s48[['tempP','windP']]); res48=pd.Series(mB.resid.values,index=s48.year.values)
mC=hac_ols(s48.fsa,s48[['tempP','windP']]); res48h=pd.Series(mC.resid.values,index=s48.year.values)
# bootstrap CI for temp effect of 10F colder (block bootstrap, year blocks of 4)
def block_boot(df,ycol,B=5000,blk=4):
    n=len(df); est=[]
    for b in range(B):
        idx=[];
        while len(idx)<n:
            st=rng.integers(0,n-blk+1); idx+=list(range(st,st+blk))
        idx=idx[:n]; bb=df.iloc[idx]; m=sm.OLS(bb[ycol],sm.add_constant(bb[['tempP','windP']])).fit(); est.append(m.params.values)
    return np.array(est)
bb=block_boot(m83,'fsa'); P(f'block-bootstrap (4-yr blocks) 95% CI: temp per F [{np.percentile(bb[:,1],2.5):+.4f},{np.percentile(bb[:,1],97.5):+.4f}]; 10F colder [{-10*np.percentile(bb[:,1],97.5):+.2f},{-10*np.percentile(bb[:,1],2.5):+.2f}]; wind per mph [{np.percentile(bb[:,2],2.5):+.4f},{np.percentile(bb[:,2],97.5):+.4f}]')

# ---------------- era contrasts at the year level (HAC), on weather-adjusted residuals
P('\n=== 3. Era contrasts with the year as the unit (HAC SE, lags 2) ===')
def era_contrast(res,cut,label):
    y=res.values; x=(res.index>=cut).astype(int)
    m=sm.OLS(y,sm.add_constant(x)).fit(cov_type='HAC',cov_kwds={'maxlags':2})
    m0=sm.OLS(y,sm.add_constant(x)).fit()
    P(f'{label}: shift at {cut} = {m.params[1]:+.2f} (HAC SE {m.bse[1]:.2f}, p {m.pvalues[1]:.3f}; iid SE {m0.bse[1]:.2f}); n before {int((x==0).sum())}, after {int((x==1).sum())}')
    return m.params[1],m.bse[1],m.pvalues[1]
era_contrast(res83,2019,'1983-2026 headline, weather-adjusted')
era_contrast(res83[res83.index!=2020],2019,'  same, dropping Nov 2020')
era_contrast(res48,1957,'1948-2026 first36, weather-adjusted')
era_contrast(res48,1981,'  first36 at 1981')
era_contrast(res48[(res48.index>=1981)],2002,'  first36 2002 vs 1981-2001')
era_contrast(res83,2002,'  headline 2002 vs 1983-2001')
era_contrast(res83[(res83.index<2019)],2006,'  headline 2006 vs (1983-2005 vs 2006-18)')

# ---------------- sup-F for a break of unknown date, AR(1) sieve bootstrap null
P('\n=== 4. Break of unknown date: sup-F (15% trimming) with AR(1)-sieve bootstrap null, 5,000 draws ===')
def supF(res,trim=0.15):
    y=res.values; n=len(y); lo=int(np.floor(n*trim)); hi=n-lo; Fs=[]
    rss0=np.sum((y-y.mean())**2)
    for k in range(lo,hi):
        a=y[:k]; b=y[k:]; rss1=np.sum((a-a.mean())**2)+np.sum((b-b.mean())**2); Fs.append((rss0-rss1)/(rss1/(n-2)))
    k=int(np.argmax(Fs)); return max(Fs),res.index[lo+k],np.array(Fs)
def supF_sieve(res,B=2000,trim=0.15):
    F0,yr,_=supF(res,trim); y=res.values-res.values.mean(); n=len(y)
    rho=np.sum(y[1:]*y[:-1])/np.sum(y[:-1]**2); e=y[1:]-rho*y[:-1]; e=e-e.mean()
    Fb=[]
    for b in range(B):
        ys=np.zeros(n); ys[0]=y[rng.integers(n)]
        eb=rng.choice(e,size=n,replace=True)
        for t in range(1,n): ys[t]=rho*ys[t-1]+eb[t]
        Fb.append(supF(pd.Series(ys,index=res.index),trim)[0])
    Fb=np.array(Fb); return F0,yr,np.mean(Fb>=F0),rho,np.percentile(Fb,95)
for lab,r in [('1983-2026 headline adj.',res83),('  dropping 2020',res83[res83.index!=2020]),('1948-2026 first36 adj.',res48),('1948-2026 headline adj.',res48h),('1957-2026 first36 adj. (post-cut)',res48[res48.index>=1957]),('1983-2026 first36 adj.',res48[res48.index>=1983])]:
    F0,yr,p,rho,c95=supF_sieve(r); P(f'{lab}: sup-F {F0:.1f} at {yr}; AR(1) rho {rho:.2f}; sieve p {p:.3f}; sieve 5% critical {c95:.1f}; n {len(r)}')
# Bai-Perron style second break on first36 1948-2026: split at first break and search each side
F0,yr,_=supF(res48); P(f'first36 1948-2026: after a break at {yr}, sub-sample sup-F: ')
for lab,r in [('before',res48[res48.index<yr]),('after',res48[res48.index>=yr])]:
    if len(r)>=20:
        F1,y1,p1,rho1,_=supF_sieve(r,B=3000); P(f'   {lab} {r.index.min()}-{r.index.max()}: sup-F {F1:.1f} at {y1}, sieve p {p1:.3f}')

# ---------------- amplitude
P('\n=== 5. Amplitude ===')
h=s.set_index('year').fsa
for lab,ser in [('Augusta 1983-2026',h[h.index>=1983]),('Augusta 1948-2026',h[h.index>=1948]),('Augusta 1957-2026',h[h.index>=1957])]:
    d1=ser.diff().abs().dropna(); P(f'{lab}: SD {ser.std():.2f}; mean |yoy| {d1.mean():.2f}; median |yoy| {d1.median():.2f}; share >=1 {np.mean(d1>=1):.2f} ({int((d1>=1).sum())}/{len(d1)})')
r=res83; P(f'weather-adjusted 1983-2026: SD {r.std():.2f}; mean |yoy| {r.diff().abs().mean():.2f}')
# rolling 15-yr windows
ser=h[h.index>=1948]; sds=[ser.iloc[i:i+15].std() for i in range(len(ser)-14)]; mads=[ser.iloc[i:i+15].diff().abs().mean() for i in range(len(ser)-14)]
P(f'15-yr windows since 1948 ({len(sds)}): SD range {min(sds):.2f}-{max(sds):.2f}; mean|yoy| range {min(mads):.2f}-{max(mads):.2f}')
# benchmark (Data Golf visit averages, from the working paper table)
bench=[('Royal St George\'s','Open',[3.33,0.45]),('Pebble Beach','U.S. Open',[3.98,1.22]),('Shinnecock Hills','U.S. Open',[4.39,2.68]),('St Andrews','Open',[0.88,-0.56,-0.64]),('Royal Liverpool','Open',[0.31,1.43]),('Royal Portrush','Open',[1.17,0.08]),('Quail Hollow','PGA',[2.09,1.46]),('Oak Hill','PGA',[1.88,2.43]),('Royal Birkdale','Open',[1.28,0.74]),('Oakmont','U.S. Open',[3.33,3.82]),('Royal Troon','Open',[2.15,2.60]),('Whistling Straits','PGA',[0.76,0.36]),('Pinehurst No. 2','U.S. Open',[3.09,2.78]),('Valhalla','PGA',[-0.41,-0.72]),('Kiawah Island','PGA',[1.95,2.01])]
gaps=[max(v)-min(v) if len(v)==2 else max(abs(v[i]-v[j]) for i in range(len(v)) for j in range(i+1,len(v))) for _,_,v in bench]
gaps=np.array(gaps); P(f'benchmark 15 venues: median gap {np.median(gaps):.2f}; mean {gaps.mean():.2f}; >=1: {int((gaps>=1).sum())}/15')
# Augusta's change at gaps equal to benchmark gaps (mean 8.3 yrs): use gaps list from paper
bgaps=[10,9,8,5,9,6,8,10,9,9,8,5,10,10,9]
aug=h[h.index>=1983]; vals=[]
for gph in bgaps:
    dd=[abs(aug[y+gph]-aug[y]) for y in aug.index if (y+gph) in aug.index]; vals.append(np.mean(dd))
P(f'Augusta mean |change| at the benchmark gaps (mean gap {np.mean(bgaps):.1f}): {np.mean(vals):.2f}; stationary-series expectation 2*SD/sqrt(pi) = {2*aug.std()/np.sqrt(np.pi):.2f}')
# Mann-Whitney on |yoy| vs benchmark gaps
u=stats.mannwhitneyu(aug.diff().abs().dropna(),gaps); P(f'Mann-Whitney Augusta |yoy| vs benchmark gaps: p {u.pvalue:.2f}')

# ---------------- hole-year panel
P('\n=== 6. Hole-year panel, 42 Masters, 756 hole-years ===')
d=pd.read_csv(D+'hole_year_panel_1983_2026.csv'); d['y']=d.year.astype(str); d['h']=d.hole.astype(str); d['dy10']=d.dy/10
d['t']=d.year-1983; d['bog']=d.bogey/d.played; d['bird']=d.birdie/d.played
# hole-specific trends
for hh in range(1,19): d[f'tr{hh}']=(d.hole==hh)*d.t
trend_terms=' + '.join(f'tr{hh}' for hh in range(1,18))  # hole 18 trend omitted (collinear with year FE)
classes=['dy10','green','trees_added','trees_removed','bunker_fairway']
base='rel ~ dy10 + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)'
def wild(formula,data,term,B=2000,seed=1):
    m=smf.ols(formula,data=data).fit(); bhat=m.params[term]
    X=np.asarray(m.model.exog); names=m.model.exog_names; j=names.index(term); Xr=np.delete(X,j,axis=1)
    yv=np.asarray(m.model.endog); br=np.linalg.lstsq(Xr,yv,rcond=None)[0]; ur=yv-Xr@br; fit_r=Xr@br
    cl=data.hole.values; cls=np.unique(cl); r=np.random.default_rng(seed)
    XtXi=np.linalg.inv(X.T@X); G=len(cls); n,k=X.shape; masks=[cl==c for c in cls]
    def ct(y):
        b=XtXi@(X.T@y); e=y-X@b; V=np.zeros((k,k))
        for mk in masks: sc=X[mk].T@e[mk]; V+=np.outer(sc,sc)
        V=XtXi@V@XtXi*G/(G-1)*(n-1)/(n-k); return b[j]/np.sqrt(V[j,j])
    t0=ct(yv); ts=np.empty(B)
    for b in range(B):
        wv=r.choice([-1.,1.],size=G); ystar=fit_r+ur*np.concatenate([np.full(mk.sum(),wv[i]) for i,mk in enumerate(masks)]) if False else fit_r+ur*np.array([wv[np.searchsorted(cls,c)] for c in cl]); ts[b]=ct(ystar)
    p=np.mean(np.abs(ts)>=abs(t0)); se=m.get_robustcov_results(cov_type='cluster',groups=data.hole).bse[j]; return bhat,se,p
P('-- 6a. Two-way FE, all classes (SE clustered by hole; wild-cluster bootstrap p, Rademacher, 5,000 draws)')
resA={}
for c in classes:
    b,se,p=wild(base,d,c); resA[c]=(b,se,p); P(f'   {c}: {b:+.4f} (SE {se:.4f}, wild p {p:.3f})')
P('-- 6b. Adding hole-specific linear trends')
baseT=base.replace('C(h)','C(h) + '+trend_terms)
resB={}
for c in classes:
    b,se,p=wild(baseT,d,c); resB[c]=(b,se,p); P(f'   {c}: {b:+.4f} (SE {se:.4f}, wild p {p:.3f})')
# Holm within the five-class family
def holm(ps):
    names=list(ps); pv=np.array([ps[n] for n in names]); o=np.argsort(pv); adj=np.empty(len(pv)); m=len(pv); run=0
    for i,idx in enumerate(o): run=max(run,(m-i)*pv[idx]); adj[idx]=min(1,run)
    return dict(zip(names,adj))
P('   Holm-adjusted wild p (five classes), no trends: '+', '.join(f'{k} {v:.3f}' for k,v in holm({c:resA[c][2] for c in classes}).items()))
P('   Holm-adjusted wild p, with trends:            '+', '.join(f'{k} {v:.3f}' for k,v in holm({c:resB[c][2] for c in classes}).items()))
P('-- 6c. Length only, and dose/par heterogeneity (with hole trends)')
b,se,p=wild('rel ~ dy10 + C(h) + C(y) + '+trend_terms,d,'dy10'); P(f'   length only: {b:+.4f} (SE {se:.4f}, wild p {p:.3f})')
d['dy10sq']=d.dy10**2; d['par3']=(d.par==3).astype(int); d['par5']=(d.par==5).astype(int)
b,se,p=wild(baseT.replace('dy10 ','dy10 + dy10sq '),d,'dy10sq'); P(f'   quadratic term: {b:+.4f} (SE {se:.4f}, wild p {p:.3f})')
m=smf.ols(baseT.replace('dy10 ','dy10 + dy10:par3 + dy10:par5 '),data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
b5,se5,p5=wild(baseT.replace('dy10 ','dy10 + dy10:par3 + dy10:par5 '),d,'dy10:par5')
P(f'   par 4 {m.params.dy10:+.4f}; par 3 {m.params.dy10+m.params["dy10:par3"]:+.4f} (int. p {m.pvalues["dy10:par3"]:.2f}); par 5 {m.params.dy10+m.params["dy10:par5"]:+.4f} (int. SE {se5:.4f}, wild p {p5:.3f})')
P('-- 6d. Bogey and birdie rates per 10 yards (with trends)')
for yv in ['bog','bird']:
    b,se,p=wild(baseT.replace('rel ~',f'{yv} ~'),d,'dy10'); P(f'   {yv}: {100*b:+.2f} pct points per 10 yds (SE {100*se:.2f}, wild p {p:.3f})')
    b,se,p=wild(baseT.replace('rel ~',f'{yv} ~'),d,'green'); P(f'   {yv} green: {100*b:+.2f} pct points (SE {100*se:.2f}, wild p {p:.3f})')
P('-- 6e. Randomisation inference: permute the hole-change histories across holes (keeping timing), 2,000 draws')
def coef(formula,data,term): return smf.ols(formula,data=data).fit().params[term]
b0=coef(baseT,d,'dy10'); wide={c:d.pivot(index='year',columns='hole',values=c) for c in classes}; perm=[]
for b in range(1000):
    pm=rng.permutation(np.arange(1,19)); dd=d.copy()
    for c in classes:
        W=wide[c].copy(); W.columns=pm; dd[c]=[W.loc[yv,hv] for yv,hv in zip(dd.year,dd.hole)]
    perm.append(coef(baseT,dd,'dy10'))
perm=np.array(perm); P(f'   lengthening: observed {b0:+.4f}; randomisation p {np.mean(np.abs(perm)>=abs(b0)):.3f} (null SD {perm.std():.4f})')
b0g=coef(baseT,d,'green'); permg=[]
for b in range(1000):
    pm=rng.permutation(np.arange(1,19)); dd=d.copy()
    for c in classes:
        W=wide[c].copy(); W.columns=pm; dd[c]=[W.loc[yv,hv] for yv,hv in zip(dd.year,dd.hole)]
    permg.append(coef(baseT,dd,'green'))
permg=np.array(permg); P(f'   green: observed {b0g:+.4f}; randomisation p {np.mean(np.abs(permg)>=abs(b0g)):.3f}')

P('-- 6f. Event study around lengthenings of 15+ yards (hole & year FE; with and without hole trends)')
ev=[]
for hh,g in d.sort_values('year').groupby('hole'):
    g=g.reset_index(drop=True)
    for i in range(1,len(g)):
        if g.dy[i]-g.dy[i-1]>=15: ev.append((hh,int(g.year[i]),int(g.dy[i]-g.dy[i-1])))
P(f'   events: {len(ev)}; mean yards {np.mean([e[2] for e in ev]):.1f}')
def event_study(formula_extra=''):
    dd=d.copy(); K=range(-6,7)
    for k in K:
        if k==-1: continue
        nm=f'k{k}'.replace('-','m'); dd[nm]=0.0
        for hh,yr,yd in ev:
            mk=(dd.hole==hh)&(dd.year==yr+k)
            if k==6: mk=(dd.hole==hh)&(dd.year>=yr+6)   # bin end
            if k==-6: mk=(dd.hole==hh)&(dd.year<=yr-6)
            dd.loc[mk,nm]=dd.loc[mk,nm]+yd/10.0
    terms=[f'k{k}'.replace('-','m') for k in K if k!=-1]
    f='rel ~ '+' + '.join(terms)+' + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)'+formula_extra
    m=smf.ols(f,data=dd).fit(cov_type='cluster',cov_kwds={'groups':dd.hole})
    return {k:(m.params[f'k{k}'.replace('-','m')],m.bse[f'k{k}'.replace('-','m')]) for k in K if k!=-1}
for lab,extra in [('no trends',''),('hole trends',' + '+trend_terms)]:
    es=event_study(extra); P(f'   {lab} (per 10 yds, k: coef (SE)): '+'; '.join(f'{k}:{v[0]:+.3f}({v[1]:.3f})' for k,v in es.items()))
    pre=np.mean([es[k][0] for k in range(-5,-1)]); post03=np.mean([es[k][0] for k in range(0,3)]); post3=np.mean([es[k][0] for k in range(3,7)])
    P(f'      mean pre (k=-5..-2) {pre:+.3f}; k=0..2 {post03:+.3f}; k=3..6+ {post3:+.3f}')
P('-- 6g. Stacked clean-control design (±6-yr windows; controls with no change in window), with window FE and window-hole trends')
rows=[]
for i,(hh,yr,yd) in enumerate(ev):
    win=d[(d.year>=yr-6)&(d.year<=yr+6)].copy()
    chg=win.groupby('hole').apply(lambda g: (g[['dy','green','trees_added','trees_removed','bunker_fairway']].nunique()>1).any())
    ctrl=[c for c in chg.index if (not chg[c]) and c!=hh]
    sub=win[win.hole.isin(ctrl+[hh])].copy(); sub['stk']=i; sub['treat']=((sub.hole==hh)&(sub.year>=yr)).astype(float)*yd/10
    sub['treatb']=((sub.hole==hh)&(sub.year>=yr)).astype(float); rows.append(sub)
S=pd.concat(rows); S['sh']=S.stk.astype(str)+'_'+S.h; S['sy']=S.stk.astype(str)+'_'+S.y; S['st']=S.year-S.groupby('stk').year.transform('min')
P(f'   stacks {S.stk.nunique()}, mean clean controls per event {S.groupby("stk").hole.nunique().mean()-1:.1f}')
for lab,f in [('per 10 yds','rel ~ treat + C(sh) + C(sy)'),('binary','rel ~ treatb + C(sh) + C(sy)'),('per 10 yds + stack-hole trends','rel ~ treat + C(sh) + C(sy) + C(sh):st')]:
    term='treatb' if 'treatb' in f else 'treat'
    m=smf.ols(f,data=S).fit(cov_type='cluster',cov_kwds={'groups':S.hole}); P(f'   {lab}: {m.params[term]:+.4f} (SE clustered by hole {m.bse[term]:.4f}, p {m.pvalues[term]:.3f})')
# pre-trend matched: drop controls whose pre-window slope differs from treated's by > 1 SD of slopes
P('-- 6h. Counterfactual course without lengthening (year totals minus relative effect of cumulative yards)')
tot=d.groupby('year').apply(lambda g: g.rel.sum()).rename('actual')
cum=d.groupby('year').dy.sum()/10
for lab,rate in [('pooled TWFE',resA['dy10'][0]),('TWFE with trends',resB['dy10'][0]),('event-study k>=3 (trends)',np.mean([event_study(' + '+trend_terms)[k][0] for k in range(3,7)]))]:
    cf=tot-rate*cum
    P(f'   {lab} rate {rate:+.4f}: 2002-18 actual {tot.loc[2002:2018].mean():+.2f} -> counterfactual {cf.loc[2002:2018].mean():+.2f}; 2019-26 {tot.loc[2019:2026].mean():+.2f} -> {cf.loc[2019:2026].mean():+.2f}')
P(f'   relative effect summed over 660 yards: pooled {resA["dy10"][0]*66:+.2f}; trends {resB["dy10"][0]*66:+.2f}')
P('-- 6i. Where the recent easing sits: 2019-26 vs 2010-18 by hole class (descriptive)')
e1=d[(d.year>=2010)&(d.year<=2018)].groupby('hole').rel.mean(); e2=d[d.year>=2019].groupby('hole').rel.mean(); dif=(e2-e1)
par=d.groupby('hole').par.first()
P(f'   total {dif.sum():+.2f}; front {dif.loc[1:9].sum():+.2f}; back {dif.loc[10:18].sum():+.2f}; par3 {dif[par==3].sum():+.2f}; par4 {dif[par==4].sum():+.2f}; par5 {dif[par==5].sum():+.2f}')
# year-level test of that difference with HAC on the course total
r2=res83; era_contrast(r2[r2.index>=2010],2019,'   2019-26 vs 2010-18, weather-adjusted total')

# ---------------- Pebble station contrast
P('\n=== 7. Pebble Beach: station exposure (n stated) ===')
pw=pd.read_csv(D+'pws_shoreline_playing_hours.csv'); pa=pd.read_csv(D+'weather_pebble_hourly_windows.csv')
reg=pw[(pw.year<=2023)&pw.gustAvg.notna()]
r,p=stats.pearsonr(reg.gustAvg,reg.fsa); P(f'shoreline gust vs FSA, regular era: n {len(reg)}, r {r:.2f}, p {p:.3f}')
m=sm.OLS(reg.fsa,sm.add_constant(reg.gustAvg)).fit(cov_type='HC1'); P(f'   slope {m.params.gustAvg:+.3f} per mph (HC1 SE {m.bse.gustAvg:.3f}, p {m.pvalues.gustAvg:.3f}); R2 {m.rsquared:.2f}')
ra,pa_=stats.pearsonr(reg.airport_wind,reg.fsa); P(f'airport wind vs FSA, same years: n {len(reg)}, r {ra:.2f}, p {pa_:.3f}')
regA=pa[pa.era=='regular']; r1,p1=stats.pearsonr(regA.windP,regA.fsa); r2_,p2=stats.pearsonr(regA.tempP,regA.fsa)
P(f'airport 2004-2023 (n {len(regA)}): wind r {r1:.2f} (p {p1:.2f}); temp r {r2_:.2f} (p {p2:.2f})')
# Steiger test for dependent correlations (same sample)
rr,_=stats.pearsonr(reg.gustAvg,reg.airport_wind); n=len(reg)
z1=np.arctanh(r); z2=np.arctanh(ra); det=1-r**2-ra**2-rr**2+2*r*ra*rr; rbar=(r+ra)/2
t=(r-ra)*np.sqrt((n-1)*(1+rr)/(2*((n-1)/(n-3))*det+rbar**2*(1-rr)**3)); P(f'   shoreline vs airport correlation difference (Williams t, n={n}): t {t:.2f}, p {2*(1-stats.t.cdf(abs(t),n-3)):.2f}; r(shoreline,airport) {rr:.2f}')
P(f'Pebble regular era amplitude 2004-2023: SD {regA.fsa.std():.2f}; mean |yoy| {regA.fsa.diff().abs().mean():.2f}')
P('DONE')
