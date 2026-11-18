import pandas as pd, numpy as np
rows=[]; year=None; par=None
for line in open('riviera_compact.txt'):
    line=line.rstrip('\n')
    if line.startswith('Y'):
        y,p=line[1:].split(); year=int(y); par=int(p); continue
    pid,am,pos,sc=line.split(',',3)
    scores=sc.split(' ')
    for i,s in enumerate(scores):
        if s=='' : continue
        rows.append((year,par,pid,int(am),pos,i+1,int(s)))
df=pd.DataFrame(rows,columns=['year','par','pid','am','pos','rnd','score'])
df=df[(df.year!=1943)]  # not held; API echoed 2026
venue={}
for y in range(1926,2027): venue[y]='other'
for y in [1929,1930,1941]+list(range(1945,1954))+list(range(1973,1983))+list(range(1984,1998))+list(range(1999,2025))+[2026]: venue[y]='Riviera'
for y in list(range(1956,1968))+list(range(1969,1973)): venue[y]='Rancho Park'
venue[1983]='Rancho Park'
df['venue']=df.year.map(venue)
df['rel']=df.score-df.par
# sanity: drop impossible scores
df=df[(df.score>=55)&(df.score<=100)]
df.to_csv('riviera_laopen_rounds_1926_2026.csv',index=False)
g=df.groupby('year')
out=pd.DataFrame({'venue':g.venue.first(),'par':g.par.first(),'players':g.pid.nunique(),'rounds':g.size(),
  'fsa_all':g.rel.mean().round(3),
  'fsa_r1r2':df[df.rnd<=2].groupby('year').rel.mean().round(3),
  'fsa_final':df[df.rnd==df.groupby('year').rnd.transform('max')].groupby('year').rel.mean().round(3),
  'n_r4':df[df.rnd==4].groupby('year').size(),
  'amateurs':g.apply(lambda x: x[x.am==1].pid.nunique()),
  'sd_round':g.rel.std().round(2)})
out['finalists']=df[df.rnd==4].groupby('year').pid.nunique()
out.to_csv('riviera_fsa_series_1926_2026.csv')
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
print(out.to_string())
riv=out[out.venue=='Riviera']
print('\nRiviera editions:',len(riv))
for lo,hi in [(1929,1953),(1973,1982),(1984,1997),(1999,2009),(2010,2018),(2019,2026)]:
    s=riv.loc[lo:hi]
    print(f'{lo}-{hi}: n={len(s)} all={s.fsa_all.mean():+.2f} r1r2={s.fsa_r1r2.mean():+.2f} final={s.fsa_final.mean():+.2f} sd={s.fsa_all.std():.2f}')
full=riv.loc[1973:]
print('\n1973-2026 Riviera: mean',round(full.fsa_all.mean(),2),'sd',round(full.fsa_all.std(),2),'n',len(full))
print('|year-to-year diff| mean',round(full.fsa_all.diff().abs().mean(),2))
import statsmodels.api as sm
X=sm.add_constant(full.index.values.astype(float)); m=sm.OLS(full.fsa_all.values,X).fit()
print('linear trend per decade',round(m.params[1]*10,3),'p',round(m.pvalues[1],3))
rp=out[out.venue=='Rancho Park']
print('\nRancho Park 1956-72,83: mean',round(rp.fsa_all.mean(),2),'sd',round(rp.fsa_all.std(),2),'n',len(rp))
