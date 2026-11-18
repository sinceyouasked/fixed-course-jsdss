import pandas as pd, numpy as np, json, re
import statsmodels.api as sm
# ---- parse raw
blocks={}; cur=None; windows={}; daily=None
for line in open('riviera_weather_raw.txt'):
    line=line.rstrip('\n')
    if line.startswith('#'):
        cur=line[1:]; blocks[cur]=[]; continue
    if cur=='WINDOWS':
        k,a,b=line.split('|'); windows[k]=(a,b); continue
    if cur=='DAILY':
        daily=json.loads(line); continue
    blocks[cur].append(line)
rows=[]
for k,ls in blocks.items():
    if k in ('WINDOWS','DAILY'): continue
    st,yr=k.split('_')
    for l in ls:
        f=l.split('|')
        if len(f)<8: continue
        d,typ,t,w,g,p,rh,wx=f[:8]
        def num(x):
            x=x.replace('s','').replace('*','').strip()
            try: return float(x)
            except: return np.nan
        rows.append((st,yr,d,typ,num(t),num(w),num(g),num(p) if p not in ('T','') else (0.005 if p=='T' else np.nan),num(rh),wx))
obs=pd.DataFrame(rows,columns=['st','key','dt','typ','temp','wind','gust','prcp','rh','wx'])
obs['dt']=pd.to_datetime(obs.dt); obs['hour']=obs.dt.dt.hour; obs['date']=obs.dt.dt.date
obs.to_csv('riviera_lcd_obs_playinghours.csv',index=False)
# playing hours 08:00-17:59
ph=obs[(obs.hour>=8)&(obs.hour<=17)]
hourly=ph[ph.typ.isin(['FM-15','SAO'])]
agg=hourly.groupby(['st','key']).agg(tempP=('temp','mean'),windP=('wind','mean'),rhP=('rh','mean'),nobs=('temp','size'),days=('date','nunique')).reset_index()
gust=ph.groupby(['st','key']).gust.max().rename('gustP').reset_index()
# precip: sum of hourly FM-15 precip during playing hours; rain hours = hours with any RA in wx or prcp>0
pr=hourly.groupby(['st','key']).agg(precP=('prcp','sum')).reset_index()
rh_=ph.assign(rain=((ph.prcp>0)|ph.wx.str.contains('RA|DZ',na=False))).groupby(['st','key']).rain.sum().rename('rainHrs').reset_index()
W=agg.merge(gust,on=['st','key']).merge(pr,on=['st','key']).merge(rh_,on=['st','key'])
W['year']=W.key.str.replace('PGA','').astype(int); W['event']=np.where(W.key.str.startswith('PGA'),'PGA','LAOpen')
W=W.round(2)
# add 2026 from daily summaries (SMO daily: TMAX/TMIN mean, AWND)
d26=pd.DataFrame(daily['smo2026']);
t26=((d26.TMAX.astype(float)+d26.TMIN.astype(float))/2).mean(); w26=d26.AWND.astype(float).mean(); p26=d26.PRCP.astype(float).sum()
W=pd.concat([W,pd.DataFrame([{'st':'SMOdaily','key':'2026','tempP':round(t26,2),'windP':round(w26,2),'rhP':np.nan,'nobs':4,'days':4,'gustP':d26.WSF2.astype(float).max(),'precP':p26,'rainHrs':np.nan,'year':2026,'event':'LAOpen'}])],ignore_index=True)
W.to_csv('riviera_weather_windows.csv',index=False)
# ---- join to scoring
S=pd.read_csv('riviera_fsa_series_1926_2026.csv',index_col=0)
lax=W[(W.st=='LAX')&(W.event=='LAOpen')].set_index('year')
smo=W[(W.st=='SMO')&(W.event=='LAOpen')].set_index('year')
riv=S[(S.venue=='Riviera')&(S.index>=1973)].copy()
riv=riv.join(lax[['tempP','windP','gustP','precP','rainHrs']].add_prefix('lax_')).join(smo[['tempP','windP','gustP','precP','rainHrs']].add_prefix('smo_'))
# 2026: use SMO daily as LAX proxy (flag)
riv.loc[2026,['lax_tempP','lax_windP','lax_precP']]=[t26,w26,p26]
riv['tempP']=riv.lax_tempP; riv['windP']=riv.lax_windP; riv['precP']=riv.lax_precP
print(riv[['fsa_all','tempP','windP','gustP' if 'gustP' in riv else 'lax_gustP','precP','smo_tempP','smo_windP']].round(2).to_string())
# cross-station agreement 2006-2025
both=riv.dropna(subset=['smo_tempP'])
print('\nLAX vs SMO 2006-25: temp corr',round(both.lax_tempP.corr(both.smo_tempP),3),'mean diff(SMO-LAX)',round((both.smo_tempP-both.lax_tempP).mean(),2),
      '| wind corr',round(both.lax_windP.corr(both.smo_windP),3),'mean diff',round((both.smo_windP-both.lax_windP).mean(),2))
# ---- weather models
def fit(df,cols,label):
    d=df.dropna(subset=cols+['fsa_all'])
    X=sm.add_constant(d[cols]); m=sm.OLS(d.fsa_all,X).fit(cov_type='HAC',cov_kwds={'maxlags':2})
    print(f'\n{label}: n={len(d)} R2={m.rsquared:.3f}')
    for c in cols: print(f'  {c:8s} {m.params[c]:+.4f} (SE {m.bse[c]:.4f}, p {m.pvalues[c]:.3f})')
    return m,d
m1,d1=fit(riv,['tempP','windP'],'LAX temp+wind, 1973-2026')
m2,d2=fit(riv,['tempP','windP','precP'],'LAX temp+wind+precip')
m3,d3=fit(riv.assign(limited=(riv.index>=2024).astype(int)),['tempP','windP','limited'],'LAX temp+wind + limited-field dummy')
m4,d4=fit(riv.assign(trend=(riv.index-1973)/10.0,limited=(riv.index>=2024).astype(int)),['tempP','windP','limited','trend'],'…+ linear trend per decade')
m5,d5=fit(both.assign(smo_tempP=both.smo_tempP,smo_windP=both.smo_windP),['smo_tempP','smo_windP'],'SMO temp+wind 2006-25')
m6,d6=fit(both,['lax_tempP','lax_windP'],'LAX temp+wind 2006-25 (same years)')
# residual series (weather-adjusted) from m1
riv['resid']=m1.resid.reindex(riv.index)
riv['adj']=riv.resid+riv.fsa_all.mean()
riv.round(3).to_csv('riviera_series_weather_1973_2026.csv')
# era means adjusted
for lo,hi in [(1973,1982),(1984,1997),(1999,2009),(2010,2018),(2019,2023),(2024,2026)]:
    e=riv.loc[lo:hi]; print(f'{lo}-{hi} raw {e.fsa_all.mean():+.2f} adj {e.adj.mean():+.2f} temp {e.tempP.mean():.1f} wind {e.windP.mean():.1f}')
# Chow at 2024 (limited field) and sup-F over 1990-2015 on residuals
r=riv.dropna(subset=['resid'])
from numpy.linalg import lstsq
def supF(y):
    n=len(y); best=(0,None)
    for k in range(8,n-8):
        a,b=y[:k],y[k:]; rss1=((a-a.mean())**2).sum()+((b-b.mean())**2).sum(); rss0=((y-y.mean())**2).sum()
        F=(rss0-rss1)/(rss1/(n-2));
        if F>best[0]: best=(F,k)
    return best
F,k=supF(r.resid.values); print(f'\nsup-F on weather residuals: F={F:.2f} at {r.index[k]}')
# permutation p
rng=np.random.default_rng(1); cnt=0; N=2000
for _ in range(N):
    Fp,_=supF(rng.permutation(r.resid.values)); cnt+=Fp>=F
print('permutation p',cnt/N)
# PGA weeks weather
print('\nPGA weeks:'); print(W[W.event=='PGA'][['key','tempP','windP','gustP','precP']].to_string(index=False))
