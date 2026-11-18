import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
d=pd.read_csv('../paper/v11_hole_year_panel_1983_2026.csv'); d['y']=d.year.astype(str); d['h']=d.hole.astype(str)
d['dy10']=d.dy/10; d['dy10sq']=d.dy10**2; d['par3']=(d.par==3).astype(int); d['par5']=(d.par==5).astype(int)
base=d[d.year==1983].set_index('hole').ydg_off; d['len83']=d.hole.map(base); d['lenc']=(d.len83-d.len83.mean())/100
out=[]
def P(*a): s=' '.join(str(x) for x in a); print(s); out.append(s)
def wild(formula,data,term,B=2000,seed=1):
    # wild cluster bootstrap (Rademacher) p-value for H0: beta_term=0, restricted null, clusters=hole
    m=smf.ols(formula,data=data).fit(); bhat=m.params[term]
    X=m.model.exog; names=m.model.exog_names; j=names.index(term); Xr=np.delete(X,j,axis=1)
    yv=m.model.endog; br=np.linalg.lstsq(Xr,yv,rcond=None)[0]; ur=yv-Xr@br; fit_r=Xr@br
    cl=data.hole.values; cls=np.unique(cl); rng=np.random.default_rng(seed)
    def cluster_t(y):
        b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; XtXi=np.linalg.inv(X.T@X); V=np.zeros((X.shape[1],)*2)
        for c in cls: Xc=X[cl==c]; ec=e[cl==c]; s=Xc.T@ec; V+=np.outer(s,s)
        G=len(cls); n,k=X.shape; V=XtXi@V@XtXi*G/(G-1)*(n-1)/(n-k); return b[j]/np.sqrt(V[j,j])
    t0=cluster_t(yv); ts=[]
    for b in range(B):
        w=rng.choice([-1,1],size=len(cls)); wmap=dict(zip(cls,w)); ystar=fit_r+ur*np.array([wmap[c] for c in cl]); ts.append(cluster_t(ystar))
    p=np.mean(np.abs(ts)>=abs(t0)); return bhat,m.get_robustcov_results(cov_type='cluster',groups=data.hole).bse[j],p
P('=== A2.1 Heterogeneity of the lengthening effect (hole & year FE; SE clustered by hole; wild-cluster bootstrap p, 2,000 draws) ===')
base_f='rel ~ dy10 + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)'
b,se,p=wild(base_f,d,'dy10'); P(f'pooled: per 10 yds {b:+.4f} (SE {se:.4f}, wild p {p:.3f})')
b,se,p=wild(base_f.replace('dy10 ','dy10 + dy10sq '),d,'dy10sq'); m=smf.ols(base_f.replace('dy10 ','dy10 + dy10sq '),data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
P(f'quadratic: dy10 {m.params.dy10:+.4f} (SE {m.bse.dy10:.4f}), dy10^2 {m.params.dy10sq:+.4f} (SE {m.bse.dy10sq:.4f}, wild p {p:.3f}); implied effect of first 20 yds {2*m.params.dy10+4*m.params.dy10sq:+.3f}, of yards 40-60 {(6*m.params.dy10+36*m.params.dy10sq)-(4*m.params.dy10+16*m.params.dy10sq):+.3f}')
f=base_f.replace('dy10 ','dy10 + dy10:par3 + dy10:par5 '); m=smf.ols(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
P(f'by par: par 4 {m.params.dy10:+.4f} (SE {m.bse.dy10:.4f}); par 3 {m.params.dy10+m.params["dy10:par3"]:+.4f} (interaction SE {m.bse["dy10:par3"]:.4f}, p {m.pvalues["dy10:par3"]:.2f}); par 5 {m.params.dy10+m.params["dy10:par5"]:+.4f} (interaction SE {m.bse["dy10:par5"]:.4f}, p {m.pvalues["dy10:par5"]:.2f})')
f=base_f.replace('dy10 ','dy10 + dy10:lenc '); m=smf.ols(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole}); b,se,p=wild(f,d,'dy10:lenc')
P(f'by 1983 length: per 10 yds at mean-length hole {m.params.dy10:+.4f}; change per +100 yds of baseline length {m.params["dy10:lenc"]:+.4f} (SE {se:.4f}, wild p {p:.3f})')
# relative-yards: 10 yards as share of hole length
d['dpct']=d.dy/d.len83*100
f='rel ~ dpct + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)'; b,se,p=wild(f,d,'dpct'); P(f'per 1% of 1983 length added: {b:+.4f} (SE {se:.4f}, wild p {p:.3f}); 10 yds on a 400-yd hole = 2.5% -> {2.5*b:+.3f}')
P('\n=== A2.2 Heterogeneous-timing estimators ===')
# dCDH-style DID_M on switchers: for each t, holes with dy change at t (switchers) vs holes with no change t-1->t (stayers)
rows=[]
for t in sorted(d.year.unique())[1:]:
    cur=d[d.year==t].set_index('hole'); prev=d[d.year==t-1].set_index('hole') if (t-1) in d.year.values else None
    if prev is None: continue
    j=cur[['rel','dy','green','trees_added','trees_removed','bunker_fairway']].join(prev[['rel','dy','green','trees_added','trees_removed','bunker_fairway']],lsuffix='',rsuffix='_p')
    j['ddy']=j.dy-j.dy_p; j['drel']=j.rel-j.rel_p
    other=((j.green!=j.green_p)|(j.trees_added!=j.trees_added_p)|(j.trees_removed!=j.trees_removed_p)|(j.bunker_fairway!=j.bunker_fairway_p))
    sw=j[(j.ddy>0)&~other]; st=j[(j.ddy==0)&~other]
    if len(sw)==0 or len(st)==0: continue
    for h,r in sw.iterrows(): rows.append(dict(year=t,hole=h,ddy=r.ddy,did=r.drel-st.drel.mean(),n_st=len(st)))
R=pd.DataFrame(rows); R['per10']=R.did/(R.ddy/10)
w=R.ddy; est=np.sum(R.did)/np.sum(R.ddy)*10
# cluster bootstrap by hole
rng=np.random.default_rng(2); hs=R.hole.unique(); bs=[]
for b in range(4000):
    pick=rng.choice(hs,size=len(hs),replace=True); S=pd.concat([R[R.hole==h] for h in pick]); bs.append(np.sum(S.did)/np.sum(S.ddy)*10)
P(f'de Chaisemartin–D\'Haultfoeuille DID_M (switchers vs stayers, t-1 to t, yards-weighted): {est:+.4f} per 10 yds, hole-cluster bootstrap 95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]; {len(R)} switching hole-years, mean stayers {R.n_st.mean():.1f}')
P(f'  unweighted mean of per-10 effects {R.per10.mean():+.4f}; 15+ yd switches only: {np.sum(R[R.ddy>=15].did)/np.sum(R[R.ddy>=15].ddy)*10:+.4f} (n={int((R.ddy>=15).sum())})')
# Sun–Abraham interaction-weighted estimator on first lengthening >=15 yds (absorbing), controls = not-yet-treated and never-treated
first={}
for h,g in d.sort_values('year').groupby('hole'):
    g=g.reset_index(drop=True); ev=g[(g.dy.diff()>=15)]
    first[h]=int(ev.year.iloc[0]) if len(ev) else np.inf
d['E']=d.hole.map(first); d['k']=d.year-d.E
never=[h for h,e in first.items() if e==np.inf]; P(f'first lengthening >=15 yds: cohorts {sorted(set(int(e) for e in first.values() if e!=np.inf))}; never-treated holes: {never}')
# IW: regress rel on cohort x relative-time dummies (k in -5..5, excluding -1), hole FE, year FE; sample excludes k outside window for treated; controls = never + not-yet-treated (k<-5 counts as not yet)
dd=d.copy(); dd['kk']=dd.k.clip(-6,6); dd.loc[dd.E==np.inf,'kk']=-1
dd=dd[(dd.kk>=-5)&(dd.kk<=5)|(dd.E==np.inf)]
dd['kk']=dd.kk.astype(int); dd['coh']=dd.E.replace(np.inf,0).astype(int).astype(str)
terms=[]; 
for c in sorted(dd.coh.unique()):
    if c=='0': continue
    for k in range(-5,6):
        if k==-1: continue
        nm=f'c{c}_k{k}'.replace('-','m'); dd[nm]=((dd.coh==c)&(dd.kk==k)).astype(int)
        if dd[nm].sum()>0: terms.append(nm)
m=smf.ols('rel ~ '+' + '.join(terms)+' + C(h) + C(y)',data=dd).fit(cov_type='cluster',cov_kwds={'groups':dd.hole})
# aggregate by k with cohort shares
agg={}
for k in range(-5,6):
    if k==-1: continue
    ts=[t for t in terms if t.endswith(f'_k{k}'.replace('-','m'))]; 
    if not ts: continue
    wts=np.array([dd[t].sum() for t in ts],float); wts/=wts.sum(); agg[k]=float(np.dot(wts,[m.params[t] for t in ts]))
post=np.mean([agg[k] for k in range(0,6) if k in agg]); pre=np.mean([agg[k] for k in range(-5,-1) if k in agg])
P('Sun–Abraham interaction-weighted event study (first >=15-yd lengthening; hole & year FE; controls never/not-yet treated): '+', '.join(f'k={k}:{v:+.3f}' for k,v in agg.items()))
P(f'  mean post (k=0..5) {post:+.3f}, mean pre (k=-5..-2) {pre:+.3f}; mean first-lengthening size among treated {d[d.k==0].groupby("hole").dy.first().mean():.0f} yds (event-study, Section 4: {0.0}) ')
P('\n=== A2.3 Distributional signature: bogey and birdie rates by hole-year (hole FE) ===')
d['bog']=d.bogey/d.played; d['bir']=d.birdie/d.played; d['era19']=(d.year>=2019).astype(int); d['era10']=((d.year>=2010)&(d.year<=2018)).astype(int)
d['long']=(d.len83>=d.len83.median()).astype(int)
for dep in ['bog','bir','rel']:
    m=smf.ols(f'{dep} ~ era19 + era19:long + era10 + era10:long + dy10 + green + trees_added + trees_removed + bunker_fairway + C(h)',data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
    P(f'{dep}: 2019– vs 1983–2009, short holes {m.params.era19:+.4f} (SE {m.bse.era19:.4f}); long-hole difference {m.params["era19:long"]:+.4f} (SE {m.bse["era19:long"]:.4f}, p {m.pvalues["era19:long"]:.2f}); 2010–18 short {m.params.era10:+.4f}, long diff {m.params["era10:long"]:+.4f}')
# birdie/bogey by lengthening
for dep in ['bog','bir']:
    m=smf.ols(f'{dep} ~ dy10 + dy10:long + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)',data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
    P(f'{dep} per 10 yds added: short holes {m.params.dy10:+.4f} (SE {m.bse.dy10:.4f}); long-hole difference {m.params["dy10:long"]:+.4f} (p {m.pvalues["dy10:long"]:.2f})')
open('panel_ext_results.txt','w').write('\n'.join(out))
