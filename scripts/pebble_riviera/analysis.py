import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
rounds=pd.read_csv('riviera_laopen_rounds_1926_2026.csv')
S=pd.read_csv('riviera_series_weather_1973_2026.csv',index_col=0)
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
# ---------- championships
ch={}; cur=None
for line in open('riviera_championships_raw.txt'):
    line=line.strip()
    if line.startswith('#'): cur=line[1:]; ch[cur]=[]; continue
    ch[cur].append(line.split(','))
P('=== Championship weeks at Riviera (par 71) ===')
champ={}
for k,v in ch.items():
    if k=='USWO2026':
        df=pd.DataFrame(v,columns=['id','name','am','pos','rnd','score']);
    else:
        df=pd.DataFrame(v,columns=['pid','am','pos','rnd','score','rel'])
    df['score']=df.score.astype(int); df['rnd']=df.rnd.astype(int); df['rel']=df.score-71
    df=df[(df.score>=55)&(df.score<=100)]
    r12=df[df.rnd<=2].rel.mean(); rall=df.rel.mean(); r4=df[df.rnd==4].rel.mean()
    champ[k]=dict(all=rall,r12=r12,r4=r4,n=df.groupby('name' if 'name' in df else 'pid').ngroups,rounds=len(df))
    P(f'{k}: players {champ[k]["n"]} rounds {len(df)} FSA all {rall:+.2f} first36 {r12:+.2f} final {r4:+.2f}')
# tour baseline same era
riv=S.copy()
for k,yrs in [('USO1948',(1949,1953)),('PGA1983',(1980,1986)),('PGA1995',(1992,1998)),('USSO1998',(1995,2001)),('USWO2026',(2019,2023))]:
    base=pd.read_csv('riviera_fsa_series_1926_2026.csv',index_col=0)
    b=base[(base.venue=='Riviera')&(base.index>=yrs[0])&(base.index<=yrs[1])]
    P(f'  {k} vs L.A. Open {yrs[0]}-{yrs[1]} (n={len(b)}): tour mean {b.fsa_all.mean():+.2f} -> championship-week effect {champ[k]["all"]-b.fsa_all.mean():+.2f} (first36 basis {champ[k]["r12"]-b.fsa_r1r2.mean():+.2f})')
# matched players: PGA 1983 vs LA Open 1983/84 same pids; PGA 1995 vs LA Open 1995/96
for k,yrs in [('PGA1983',(1982,1984)),('PGA1995',(1994,1996))]:
    c=pd.DataFrame(ch[k],columns=['pid','am','pos','rnd','score','rel']); c['score']=c.score.astype(int); c['rel']=c.score-71
    c['pid']=c.pid.astype(str).str.zfill(5); cm=c.groupby('pid').rel.mean()
    la=rounds[(rounds.year.isin(yrs))&(rounds.venue=='Riviera')].copy(); la['pid']=la.pid.astype(str).str.zfill(5); lm=la.groupby('pid').rel.mean()
    j=cm.to_frame('champ').join(lm.to_frame('tour'),how='inner')
    d=j.champ-j.tour
    P(f'  {k} matched players (n={len(j)}): championship − tour, same players = {d.mean():+.2f} (SE {d.std()/np.sqrt(len(d)):.2f})')
# ---------- limited field 2024-26: same players vs their own 2019-23
P('\n=== Limited field (2024, 2026: 70-72 players, no cut) ===')
rv=rounds[rounds.venue=='Riviera']
lim=rv[rv.year.isin([2024,2026])]; pre=rv[rv.year.between(2019,2023)]
pl=lim.groupby('pid').rel.mean(); pp=pre.groupby('pid').rel.mean()
j=pl.to_frame('lim').join(pp.to_frame('pre'),how='inner')
P(f'raw: 2019-23 all players {pre.rel.mean():+.2f}; 2024/26 all players {lim.rel.mean():+.2f}; gap {lim.rel.mean()-pre.rel.mean():+.2f}')
P(f'same players (n={len(j)}): their 2019-23 mean {j.pre.mean():+.2f}, their 2024/26 mean {j.lim.mean():+.2f} -> within-player change {j.lim.mean()-j.pre.mean():+.2f} (SE {(j.lim-j.pre).std()/np.sqrt(len(j)):.2f})')
P(f'composition (who plays): pre-period mean of eventual limited-field players {j.pre.mean():+.2f} vs all pre-period players {pre.rel.mean():+.2f} -> {j.pre.mean()-pre.rel.mean():+.2f}')
# player FE regression 2015-2026 with year effects
d=rv[rv.year>=2015].copy(); d['y']=d.year.astype(str)
m=smf.ols('rel ~ C(y) + C(pid)',data=d).fit()
ye={2015:0};
for y in range(2016,2027):
    if f'C(y)[T.{y}]' in m.params: ye[y]=m.params[f'C(y)[T.{y}]']
P('player-FE year effects (rel to 2015): '+', '.join(f'{y}:{v:+.2f}' for y,v in ye.items()))
# ---------- trend forms: linear vs step at 1987 vs both, on raw with wind/precip
S2=S.copy(); S2['t']=(S2.index-1973)/10; S2['step88']=(S2.index>=1988).astype(int); S2['limited']=(S2.index>=2024).astype(int)
for f in ['fsa_all ~ windP + precP + t + limited','fsa_all ~ windP + precP + step88 + limited','fsa_all ~ windP + precP + step88 + t + limited']:
    mm=smf.ols(f,data=S2.dropna(subset=['windP','precP'])).fit(cov_type='HAC',cov_kwds={'maxlags':2})
    P(f'{f}: R2 {mm.rsquared:.3f} AIC {mm.aic:.1f} | '+' '.join(f'{k}={v:+.3f}(p{mm.pvalues[k]:.3f})' for k,v in mm.params.items() if k!='Intercept'))
# ---------- amplitude vs other venues
P('\n=== Amplitude ===')
full=S[(S.index>=1973)]
P(f'Riviera 1973-2026 (n={len(full)}): mean {full.fsa_all.mean():+.2f} SD {full.fsa_all.std():.2f} mean|Δ| {full.fsa_all.diff().abs().mean():.2f}; detrended SD {np.std(sm.OLS(full.fsa_all.values,sm.add_constant(full.index.values.astype(float))).fit().resid,ddof=2):.2f}')
ex=full[full.index<=2023]
P(f'Riviera 1973-2023 full fields (n={len(ex)}): mean {ex.fsa_all.mean():+.2f} SD {ex.fsa_all.std():.2f} mean|Δ| {ex.fsa_all.diff().abs().mean():.2f}; weather-adj resid SD {ex.resid.std():.2f}')
P('Augusta 1983-2026: mean +1.34 SD 0.84 (paper); Pebble AT&T 2004-23: mean +0.02 SD 0.79 (paper)')
# round-level SD
P(f'round-level SD of scores: Riviera {rv[rv.year>=1983].rel.std():.2f}')
open('analysis_results.txt','w').write('\n'.join(out))
