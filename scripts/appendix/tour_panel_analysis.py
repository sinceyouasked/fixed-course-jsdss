import pandas as pd, numpy as np, statsmodels.formula.api as smf
d=pd.read_csv('tour_panel.csv'); d=d[d.rounds>=100].copy()
d=d[~d.suf.isin([535,536,544,545,547,548,549])]  # duplicate ids from the 2020-21 wraparound season
d=d[~((d.suf==7)&(d.year.isin([1983,1998,2025])))]  # L.A. Open away from Riviera
# event = suffix (tournament id); treat as course proxy; flag limited field = no cut and players<=80
d['limited']=((d.cut==0)&(d.players<=80)).astype(int)
d['nocut']=(d.cut==0).astype(int)
out=[]
def P(*a): s=' '.join(str(x) for x in a); print(s); out.append(s)
P('=== Tour-wide limited-field DiD, 2019-2026 (event FE + year FE; SE clustered by event) ===')
m=d[(d.year>=2019)&(d.year<=2026)].copy()
# exclude majors/WGC/playoffs that are limited by design throughout? Keep but they have limited=1 always -> absorbed by event FE; the identification comes from switchers.
sw=m.groupby('suf').limited.agg(['min','max']); switchers=sw[(sw['min']==0)&(sw['max']==1)].index
P(f'events in window: {m.suf.nunique()}; event-years {len(m)}; switchers (full->limited within window): {len(switchers)}: '+', '.join(sorted(set(m[m.suf.isin(switchers)].name))[:20]))
m['y']=m.year.astype(str); m['e']=m.suf.astype(str)
for f,lab in [('mean ~ limited + C(e) + C(y)','all events'),]:
    r=smf.ols(f,data=m).fit(cov_type='cluster',cov_kwds={'groups':m.suf})
    P(f'{lab}: limited-field effect {r.params.limited:+.3f} (SE {r.bse.limited:.3f}, p {r.pvalues.limited:.3f}); n={len(m)}')
# restrict to switchers + never-limited full-field events as controls
ctrl=sw[(sw['max']==0)].index; ms=m[m.suf.isin(list(switchers)+list(ctrl))]
r=smf.ols('mean ~ limited + C(e) + C(y)',data=ms).fit(cov_type='cluster',cov_kwds={'groups':ms.suf})
P(f'switchers + never-limited controls: {r.params.limited:+.3f} (SE {r.bse.limited:.3f}, p {r.pvalues.limited:.3f}); n={len(ms)}, events {ms.suf.nunique()}')
# event study: relative year to first limited year among switchers
first=m[m.limited==1].groupby('suf').year.min(); ms2=ms.copy(); ms2['k']=ms2.year-ms2.suf.map(first); ms2['k']=ms2.k.fillna(-99)
for k in [-3,-2,0,1,2]:
    ms2[f'k{k}'.replace('-','m')]=(ms2.k==k).astype(int)
r=smf.ols('mean ~ km3 + km2 + k0 + k1 + k2 + C(e) + C(y)',data=ms2).fit(cov_type='cluster',cov_kwds={'groups':ms2.suf})
P('event study (ref k=-1): '+', '.join(f'k={k}:{r.params[n]:+.2f}(SE {r.bse[n]:.2f})' for k,n in [(-3,'km3'),(-2,'km2'),(0,'k0'),(1,'k1'),(2,'k2')]))
# per-switcher raw change
P('per-event: pre (2019-23 full) mean -> limited mean')
for s in switchers:
    g=m[m.suf==s]; pre=g[g.limited==0]['mean'].mean(); post=g[g.limited==1]['mean'].mean(); P(f'  {g.name.iloc[-1][:34]:34s} {pre:+.2f} -> {post:+.2f} ({post-pre:+.2f}); limited years {sorted(g[g.limited==1].year.tolist())}')
# first-36 instrument (cut-free) version
r=smf.ols('mean12 ~ limited + C(e) + C(y)',data=ms).fit(cov_type='cluster',cov_kwds={'groups':ms.suf})
P(f'on first-36 (cut-free) field average: {r.params.limited:+.3f} (SE {r.bse.limited:.3f}, p {r.pvalues.limited:.3f})')

P('\n=== Synthetic control: Riviera (suffix 7) step at 1987/88 ===')
W=pd.read_csv('../riv/riviera_series_weather_1973_2026.csv',index_col=0)
riv_adj=(W.fsa_all-0.149*(W.windP-W.windP.mean())+0.559*(W.precP-W.precP.mean()))
# donor pool: events with complete series 1975-1997 (raw, unadjusted), excluding Riviera, majors handled by FE anyway
pre=(1975,1987); post=(1988,1997)
pv=d.pivot_table(index='year',columns='suf',values='mean')
yrs=list(range(pre[0],post[1]+1)); pv=pv.loc[yrs]
donors=[c for c in pv.columns if c!=7 and pv[c].notna().all()]
pv=pv.copy(); pv[7]=pv[7].interpolate()  # 1983 (Rancho Park) interpolated for the fit
P(f'donor events with complete 1975-1997 series: {len(donors)}')
Y=pv[7].values; X=pv[donors].values
pre_idx=[yrs.index(y) for y in range(pre[0],pre[1]+1)]; post_idx=[yrs.index(y) for y in range(post[0],post[1]+1)]
from scipy.optimize import minimize
def fit(Y,X,pre_idx):
    k=X.shape[1]; w0=np.ones(k)/k
    obj=lambda w: np.sum((Y[pre_idx]-X[pre_idx]@w)**2)
    cons=({'type':'eq','fun':lambda w: np.sum(w)-1},); bnds=[(0,1)]*k
    r=minimize(obj,w0,bounds=bnds,constraints=cons,method='SLSQP',options={'maxiter':500}); return r.x
w=fit(Y,X,pre_idx); synth=X@w
gap=Y-synth; pre_rmse=np.sqrt(np.mean(gap[pre_idx]**2)); post_gap=gap[post_idx].mean()
P(f'Riviera: pre-period RMSPE {pre_rmse:.2f}; mean post-1988 gap (actual − synthetic) {post_gap:+.2f}; actual change pre->post {Y[post_idx].mean()-Y[pre_idx].mean():+.2f}; synthetic change {synth[post_idx].mean()-synth[pre_idx].mean():+.2f}')
top=sorted(zip(w,donors),reverse=True)[:6]; names={s:d[d.suf==s].name.iloc[-1] for s in donors}
P('  weights: '+', '.join(f'{names[s][:28]} {wt:.2f}' for wt,s in top if wt>0.01))
# placebo in space: every donor as treated
ratios=[]
for j,c in enumerate(donors):
    Yc=pv[c].values; Xc=pv[[x for x in donors if x!=c]+[7]].values; wc=fit(Yc,Xc,pre_idx); g=Yc-Xc@wc
    ratios.append((np.abs(g[post_idx].mean())/np.sqrt(np.mean(g[pre_idx]**2)), g[post_idx].mean(), c))
riv_ratio=abs(post_gap)/pre_rmse; rank=1+sum(r[0]>riv_ratio for r in ratios)
P(f'  placebo (in space): Riviera post/pre ratio {riv_ratio:.2f}, rank {rank} of {len(donors)+1} -> permutation p = {rank/(len(donors)+1):.3f}; donors with post gap <= Riviera\'s: {sum(r[1]<=post_gap for r in ratios)} of {len(donors)}')
# tour-wide mean change for reference
tour=pv[donors].mean(axis=1); P(f'  simple donor-average change 1975-87 -> 1988-97: {tour.loc[post[0]:post[1]].mean()-tour.loc[pre[0]:pre[1]].mean():+.2f} (Riviera {Y[post_idx].mean()-Y[pre_idx].mean():+.2f})')
# Augusta vs Riviera divergence 1983-2023 with synthetic control on a common donor pool
P('\n=== Synthetic control: venue drift 1983-2023, Augusta (suffix 14) and Riviera (7) against the tour ===')
yrs=list(range(1983,2024)); pv2=d.pivot_table(index='year',columns='suf',values='mean').loc[yrs]
for tgt,lab in [(14,'Augusta'),(7,'Riviera')]:
    donors2=[c for c in pv2.columns if c not in (7,14) and pv2[c].notna().sum()>=38]
    pvf=pv2[donors2].interpolate(limit_direction='both'); Y=pv2[tgt].interpolate(limit_direction='both').values; X=pvf.values
    pre_idx=[yrs.index(y) for y in range(1983,1998)]; post_idx=[yrs.index(y) for y in range(2009,2024)]
    w=fit(Y,X,pre_idx); synth=X@w; g=Y-synth
    P(f'{lab}: fitted on 1983-97 (RMSPE {np.sqrt(np.mean(g[pre_idx]**2)):.2f}); gap 2009-23 actual−synthetic {g[post_idx].mean():+.2f}; actual drift {Y[post_idx].mean()-Y[pre_idx].mean():+.2f}, synthetic drift {synth[post_idx].mean()-synth[pre_idx].mean():+.2f}; donors {len(donors2)}')
open('tour_panel_results.txt','w').write('\n'.join(out))
