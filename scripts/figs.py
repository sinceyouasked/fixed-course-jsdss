import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, statsmodels.api as sm, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
D='data/'; O='figures/'
import os; os.makedirs(O,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,'savefig.dpi':300})
K='#222'; G='#888'; B='#1f4e79'; R='#b03a2e'
def save(fig,name):
    fig.savefig(O+name+'.tif',bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(O+name+'.png',bbox_inches='tight'); plt.close(fig)

f36=pd.read_csv(D+'masters_fsa_first36_1934_2026.csv'); ser=pd.read_csv(D+'fig_aug_series.csv'); w=pd.read_csv(D+'fig_aug_weather.csv')
# ---- Figure 1: ninety Masters
fig,(a1,a2)=plt.subplots(2,1,figsize=(6.5,4.2),sharex=True,gridspec_kw={'height_ratios':[3,1],'hspace':0.08})
a1.plot(f36.year,f36.fsa,color=K,lw=1.1,label='Field scoring average − par (headline)')
a1.plot(f36.year,f36.first36,color=B,lw=0.9,ls='--',label='Cut-free first-36 series')
for y0,y1,m in [(1934,1942,2.98),(1946,1956,3.38),(1957,1980,1.98),(1981,2001,1.49),(2002,2018,1.49),(2019,2026,0.81)]:
    a1.hlines(m,y0-0.4,y1+0.4,color=R,lw=1.6)
a1.axhline(0,color=G,lw=0.5); a1.set_ylabel('Strokes over par'); a1.legend(loc='upper right',frameon=False)
a1.text(1934,5.0,'A',fontweight='bold');
yd=ser.dropna(subset=['yds']); a2.step(yd.year,yd.yds,where='post',color=K,lw=1); a2.set_ylabel('Yards'); a2.set_xlabel('Masters'); a2.text(1934,7550,'B',fontweight='bold'); a2.set_ylim(6600,7700)
a2.set_xlim(1932,2028); save(fig,'fig1')

# ---- Figure 2: weather
m=w[w.year>=1983]; X=sm.add_constant(m[['tempP','windP']]); reg=sm.OLS(m.fsa,X).fit()
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.5,2.6),gridspec_kw={'width_ratios':[1,1.6],'wspace':0.3})
a1.scatter(m.tempP,m.fsa,s=12,color=K); xs=np.linspace(m.tempP.min(),m.tempP.max(),10)
a1.plot(xs,reg.params.const+reg.params.tempP*xs+reg.params.windP*m.windP.mean(),color=R,lw=1)
for _,r in m[m.year.isin([2007,2016,2019,2020])].iterrows(): a1.annotate(str(int(r.year)),(r.tempP,r.fsa),xytext=(3,2),textcoords='offset points',fontsize=6,color=G)
a1.set_xlabel('Playing-hours temperature (°F)'); a1.set_ylabel('Strokes over par'); a1.text(0.88,0.95,'A',transform=a1.transAxes,fontweight='bold')
wl=w[w.year>=1948]; Xl=sm.add_constant(wl[['tempP','windP']]); rl=sm.OLS(wl.fsa,Xl).fit()
a2.plot(wl.year,wl.fsa,color=G,lw=0.8,label='Raw'); a2.plot(wl.year,rl.resid+wl.fsa.mean(),color=B,lw=1,label='Weather-adjusted (re-centred)')
a2.axhline(wl.fsa.mean(),color=G,lw=0.4,ls=':'); a2.set_xlabel('Masters'); a2.legend(frameon=False,loc='upper right'); a2.text(0.02,0.95,'B',transform=a2.transAxes,fontweight='bold')
save(fig,'fig2')

# ---- Figure 3: amplitude
bench=[("Royal St George's",2.88),('Pebble Beach',2.76),('Shinnecock Hills',1.71),('St Andrews',1.52),('Royal Liverpool',1.12),('Royal Portrush',1.09),('Quail Hollow',0.63),('Oak Hill',0.55),('Royal Birkdale',0.54),('Oakmont',0.49),('Royal Troon',0.45),('Whistling Straits',0.40),('Pinehurst No. 2',0.31),('Valhalla',0.31),('Kiawah Island',0.06)]
aug=f36[f36.year>=1983].set_index('year').fsa; yoy=aug.diff().abs().dropna()
fig,ax=plt.subplots(figsize=(6.5,2.4))
rng=np.random.default_rng(1)
ax.scatter(yoy.values,rng.uniform(0.85,1.15,len(yoy)),s=10,color=K,alpha=0.7,label='Augusta, consecutive Masters 1983–2026 (43 pairs)')
ax.scatter([g for _,g in bench],rng.uniform(1.85,2.15,len(bench)),s=14,color=R,marker='D',label='Rotating major venues, between visits 2010–2026 (15)')
ax.vlines(np.median(yoy),0.7,1.3,color=K,lw=1); ax.vlines(np.median([g for _,g in bench]),1.7,2.3,color=R,lw=1)
ax.set_yticks([1,2]); ax.set_yticklabels(['Fixed venue,\nyear to year','Rotating venues,\nvisit to visit']); ax.set_xlabel('Absolute change in field scoring average (strokes)'); ax.set_ylim(0.4,2.6)
ax.legend(frameon=False,loc='upper right'); save(fig,'fig3')

# ---- panel
d=pd.read_csv(D+'hole_year_panel_1983_2026.csv'); d['y']=d.year.astype(str); d['h']=d.hole.astype(str); d['dy10']=d.dy/10; d['t']=d.year-1983
for hh in range(1,19): d[f'tr{hh}']=(d.hole==hh)*d.t
tt=' + '.join(f'tr{hh}' for hh in range(1,18))
classes=['dy10','green','trees_added','trees_removed','bunker_fairway']; labels=['Lengthening\n(per 10 yards)','Green rebuilt or\nrecontoured','Trees added','Trees removed','Bunker or\nfairway work']
base='rel ~ dy10 + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y)'
mA=smf.ols(base,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole}); mB=smf.ols(base+' + '+tt,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.hole})
fig,ax=plt.subplots(figsize=(6.5,2.4)); x=np.arange(5)
ax.errorbar(x-0.12,[mA.params[c] for c in classes],yerr=[1.96*mA.bse[c] for c in classes],fmt='o',color=K,ms=4,capsize=2,label='Hole and year fixed effects')
ax.errorbar(x+0.12,[mB.params[c] for c in classes],yerr=[1.96*mB.bse[c] for c in classes],fmt='s',color=B,ms=4,capsize=2,label='+ hole-specific linear trends')
ax.axhline(0,color=G,lw=0.5); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Strokes on the hole, relative\nto the other seventeen'); ax.legend(frameon=False,loc='upper right'); save(fig,'fig4')

# ---- event study
ev=[]
for hh,g in d.sort_values('year').groupby('hole'):
    g=g.reset_index(drop=True)
    for i in range(1,len(g)):
        if g.dy[i]-g.dy[i-1]>=15: ev.append((hh,int(g.year[i]),int(g.dy[i]-g.dy[i-1])))
dd=d.copy(); K_=list(range(-6,7))
for k in K_:
    if k==-1: continue
    nm=f'k{k}'.replace('-','m'); dd[nm]=0.0
    for hh,yr,yd_ in ev:
        mk=(dd.hole==hh)&(dd.year==yr+k)
        if k==6: mk=(dd.hole==hh)&(dd.year>=yr+6)
        if k==-6: mk=(dd.hole==hh)&(dd.year<=yr-6)
        dd.loc[mk,nm]+=yd_/10
terms=[f'k{k}'.replace('-','m') for k in K_ if k!=-1]
me=smf.ols('rel ~ '+' + '.join(terms)+' + green + trees_added + trees_removed + bunker_fairway + C(h) + C(y) + '+tt,data=dd).fit(cov_type='cluster',cov_kwds={'groups':dd.hole})
ks=[k for k in K_ if k!=-1]; b=[me.params[f'k{k}'.replace('-','m')] for k in ks]; se=[me.bse[f'k{k}'.replace('-','m')] for k in ks]
fig,ax=plt.subplots(figsize=(6.5,2.4)); ax.errorbar(ks,b,yerr=[1.96*s for s in se],fmt='o-',color=K,ms=3,capsize=2,lw=0.8)
ax.plot([-1],[0],'o',color='white',mec=K,ms=3); ax.axhline(0,color=G,lw=0.5); ax.axvline(-0.5,color=G,lw=0.5,ls=':')
ax.set_xlabel('Masters relative to the lengthening (k = −6 and 6 binned)'); ax.set_ylabel('Strokes on the hole per 10 yards,\nrelative to k = −1'); ax.set_xticks(K_); save(fig,'fig5')

# ---- counterfactual
tot=d.groupby('year').rel.sum(); cum=d.groupby('year').dy.sum()/10
post3=np.mean([me.params[f'k{k}'] for k in [3,4,5,6]])
fig,ax=plt.subplots(figsize=(6.5,2.6))
ax.plot(tot.index,tot.values,color=K,lw=1.2,label='Course as played (sum of hole scores relative to par)')
ax.plot(tot.index,(tot-mB.params.dy10*cum).values,color=R,lw=1,ls='--',label=f'Without lengthening, two-way FE rate ({mB.params.dy10:.3f}/10 yd)')
ax.plot(tot.index,(tot-post3*cum).values,color=B,lw=1,ls=':',label=f'Without lengthening, post-reversion rate ({post3:.3f}/10 yd)')
ax.axhline(0,color=G,lw=0.5); ax.set_xlabel('Masters'); ax.set_ylabel('Strokes over par'); ax.legend(frameon=False,loc='lower left'); save(fig,'fig6')
print('post3',post3,'twfe',mB.params.dy10)
