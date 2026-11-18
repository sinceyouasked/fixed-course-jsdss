import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
t=open('aging_raw.txt').read().split('#BIO pid,born\n')
M=pd.DataFrame([l.split(',') for l in t[0].split('\n')[1:] if l],columns=['year','pid','am','n','sumrel']).astype({'year':int,'am':int,'n':int,'sumrel':float})
B=pd.DataFrame([l.split(',') for l in t[1].split('\n') if l],columns=['pid','born']); B['born']=pd.to_datetime(B.born,errors='coerce'); B=B.dropna()
R=pd.read_csv('../riv/riviera_laopen_rounds_1926_2026.csv'); R=R[(R.venue=='Riviera')&(R.year>=1973)].copy(); R['pid']=R.pid.astype(str).str.zfill(5)
Rg=R.groupby(['year','pid','am']).rel.agg(['size','sum']).reset_index().rename(columns={'size':'n','sum':'sumrel'})
out=[]
def P(*a): s=' '.join(str(x) for x in a); print(s); out.append(s)
def prep(df,name,event_month):
    d=df.merge(B,on='pid',how='left'); d['age']=d.year+event_month/12-(d.born.dt.year+d.born.dt.dayofyear/365.25); d['mean']=d.sumrel/d.n
    cov=d.born.notna().mean(); d=d[d.born.notna()&(d.n>=1)]
    nobs=d.groupby('pid').year.nunique(); d=d[d.pid.map(nobs)>=3]
    P(f'{name}: birth-date coverage {cov:.1%} of player-tournaments; sample with 3+ appearances: {d.pid.nunique()} players, {len(d)} player-tournaments, {int(d.n.sum())} rounds; age range {d.age.min():.0f}-{d.age.max():.0f}')
    return d
A=prep(M,'Augusta 1983-2026',4/12*0+4); Rv=prep(Rg,'Riviera 1973-2026',2)
# aging with an age spline (knots) + player FE + year FE, weighted by rounds. APC: with FE and year dummies, the linear part of age is unidentified. Resolve by
# (1) assuming no linear aging across the sample as the paper did (not identified); (2) BRL: identify from the curvature + an external anchor (peak age) -> we use
# the standard solution: age enters as deviations from a reference age window via a flexible nonlinear function with the linear component pinned by the
# cross-sectional age profile of ENTRANTS (who are selected), i.e. not identified. Instead we report: (a) within-player quadratic (no linear term) as before;
# (b) a two-way FE with age measured relative to each player's own career peak window estimated from their own trajectory is circular;
# (c) the cleanest identified object: year effects net of player FE, for players observed in a 'prime' window (ages 27-38) only, where published aging curves
#     (Berry et al. 1999; Data Golf) are flat to within ~0.1 strokes/round. Within this window the linear age term is ~0 by assumption from the literature, so the
#     year effects are identified without estimating an aging slope.
def yeareff(d,label,lo,hi):
    e=d[(d.age>=lo)&(d.age<=hi)].copy(); e['y']=e.year.astype(str)
    m=smf.wls('mean ~ C(y) + C(pid)',data=e,weights=e.n).fit()
    ye=pd.Series({int(k[7:11]):v for k,v in m.params.items() if k.startswith('C(y)[T.')}); ye.loc[e.year.min()]=0; ye=ye.sort_index()
    return ye,e
for d,label,eras in [(A,'Augusta',[(1983,1997),(1998,2008),(2009,2018),(2019,2026)]),(Rv,'Riviera',[(1973,1987),(1988,1997),(1998,2008),(2009,2018),(2019,2023),(2024,2026)])]:
    for lo,hi in [(27,38),(25,40)]:
        ye,e=yeareff(d,label,lo,hi)
        raw=d.groupby('year').apply(lambda g: np.average(g['mean'],weights=g.n))
        t=(ye.index-ye.index.min())/10; sl=np.polyfit(t,ye.values,1)[0]; slr=np.polyfit((raw.index-raw.index.min())/10,raw.values,1)[0]
        P(f'{label}, prime-age window {lo}-{hi} ({e.pid.nunique()} players, {len(e)} player-tournaments): year effects net of player FE: slope {sl:+.2f}/decade (raw field series slope {slr:+.2f}/decade); era means: '+', '.join(f'{a}-{b} {ye.loc[a:b].mean():+.2f} (raw {raw.loc[a:b].mean():+.2f})' for a,b in eras))
        ye.to_csv(f'year_effects_primeage_{label}_{lo}_{hi}.csv',header=['year_effect'])
# cross-check of the flat-prime assumption inside our data: within-player age profile with year FE, quadratic in age only (curvature), peak implied
for d,label in [(A,'Augusta'),(Rv,'Riviera')]:
    e=d.copy(); e['y']=e.year.astype(str); e['age2']=(e.age-32)**2
    m=smf.wls('mean ~ C(y) + C(pid) + age2',data=e,weights=e.n).fit()
    P(f'{label}: curvature of the within-player age profile (year and player FE): {m.params.age2:+.4f} strokes per (year from 32)^2 -> ages 27 and 38 differ from 32 by {m.params.age2*25:+.2f} and {m.params.age2*36:+.2f} on curvature alone; the linear term is unidentified (APC)')
# Berry-Reese-Larkey anchored version: impose the published Data Golf / BRL-type aging profile: peak at ~31, decline 0.03/yr to 35, 0.08/yr after 35, improvement 0.1/yr before 25 (approx). Report sensitivity to the anchor.
def anchored(d,label,profile):
    e=d.copy(); e['y']=e.year.astype(str); e['agecurve']=profile(e.age); e['adj']=e['mean']-e.agecurve
    m=smf.wls('adj ~ C(y) + C(pid)',data=e,weights=e.n).fit()
    ye=pd.Series({int(k[7:11]):v for k,v in m.params.items() if k.startswith('C(y)[T.')}); ye.loc[e.year.min()]=0; ye=ye.sort_index(); return ye
def prof_brl(age):  # stylised Berry-Reese-Larkey: improve to ~30, flat 30-34, decline after
    a=np.asarray(age,float); return np.where(a<30,-0.08*(a-30), np.where(a<=34,0.0, 0.05*(a-34)))
def prof_dg(age):   # stylised modern (Data Golf-type) curve: peak ~31, slow decline after 35
    a=np.asarray(age,float); return np.where(a<31,-0.06*(a-31), np.where(a<=35,0.01*(a-31), 0.04+0.07*(a-35)))
for d,label,eras in [(A,'Augusta',[(1983,1997),(2009,2018),(2019,2026)]),(Rv,'Riviera',[(1973,1987),(1988,1997),(2009,2018),(2019,2023)])]:
    for nm_,pf in [('BRL-type',prof_brl),('modern-type',prof_dg)]:
        ye=anchored(d,label,pf); t=(ye.index-ye.index.min())/10; sl=np.polyfit(t,ye.values,1)[0]
        P(f'{label}, anchored aging ({nm_}): year effects slope {sl:+.2f}/decade; '+', '.join(f'{a}-{b} {ye.loc[a:b].mean():+.2f}' for a,b in eras))
open('aging_results.txt','w').write('\n'.join(out))
