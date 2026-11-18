import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRID="#e6e5e1"; BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YELLOW="#eda100"; RED="#e34948"
rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.edgecolor":GRID,"axes.linewidth":1,"axes.facecolor":SURF,"figure.facecolor":SURF,"savefig.facecolor":SURF,"xtick.color":INK2,"ytick.color":INK2,"axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.titleweight":"bold","axes.titlesize":11,"axes.titlelocation":"left","axes.spines.top":False,"axes.spines.right":False,"grid.color":GRID,"grid.linewidth":1,"axes.grid":True,"axes.grid.axis":"y","axes.axisbelow":True,"legend.frameon":False,"legend.fontsize":8.5,"xtick.major.size":0,"ytick.major.size":0})
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6),gridspec_kw=dict(wspace=0.25))
# left: prime-age year effects vs raw
aug=pd.read_csv('../osf_in/masters_fsa_1934_2026.csv',index_col=0).fsa.loc[1983:]
riv=pd.read_csv('../riv/riviera_fsa_series_1926_2026.csv',index_col=0); riv=riv[(riv.venue=='Riviera')&(riv.index>=1973)&(riv.index<=2023)].fsa_all
ya=pd.read_csv('year_effects_primeage_Augusta_27_38.csv',index_col=0).year_effect; yr=pd.read_csv('year_effects_primeage_Riviera_27_38.csv',index_col=0).year_effect; yr=yr.loc[:2023]
def sm5(s): return s.rolling(5,center=True,min_periods=3).mean()
a1.axhline(0,color=GRID,lw=1)
a1.plot(aug.index,sm5(aug-aug.loc[1983:1987].mean()),color=AQUA,lw=1.2,ls=(0,(3,2)),label='Augusta, field average (5-yr mean, re-based 1983–87)')
a1.plot(ya.index,sm5(ya-ya.loc[1983:1987].mean()),color=AQUA,lw=2.2,label='Augusta, same prime-age players (year effects net of player FE)')
a1.plot(riv.index,sm5(riv-riv.loc[1983:1987].mean()),color=BLUE,lw=1.2,ls=(0,(3,2)),label='Riviera, field average')
a1.plot(yr.index,sm5(yr-yr.loc[1983:1987].mean()),color=BLUE,lw=2.2,label='Riviera, same prime-age players')
a1.set_xlim(1982,2027); a1.set_ylim(-2.3,1.9); a1.set_ylabel('Strokes vs 1983–87 level'); a1.legend(loc='upper right',fontsize=6.8)
a1.set_title('Holding the golfer fixed (ages 27–38)',fontsize=10)
# right: synthetic Riviera
d=pd.read_csv('tour_panel.csv'); d=d[d.rounds>=100]; d=d[~d.suf.isin([535,536,544,545,547,548,549])]; d=d[~((d.suf==7)&(d.year.isin([1983,1998,2025])))]
pv=d.pivot_table(index='year',columns='suf',values='mean'); yrs=list(range(1975,1998)); pv=pv.loc[yrs]; donors=[c for c in pv.columns if c!=7 and pv[c].notna().all()]; pv[7]=pv[7].interpolate()
from scipy.optimize import minimize
Y=pv[7].values; X=pv[donors].values; pre=[yrs.index(y) for y in range(1975,1988)]
k=X.shape[1]; obj=lambda w: np.sum((Y[pre]-X[pre]@w)**2); r=minimize(obj,np.ones(k)/k,bounds=[(0,1)]*k,constraints=({'type':'eq','fun':lambda w: w.sum()-1},),method='SLSQP',options={'maxiter':500}); synth=X@r.x
a2.axhline(0,color=GRID,lw=1); a2.axvline(1987.5,color=RED,ls=':',lw=1)
a2.plot(yrs,Y,'o-',color=BLUE,ms=4,lw=1.4,label='Riviera, field average (raw)')
a2.plot(yrs,synth,'o-',color=MUTED,ms=4,lw=1.4,label='Synthetic Riviera (51 tour events, fitted 1975–87)')
a2.plot(yrs,pv[donors].mean(axis=1).values,color=INK2,lw=1,ls=(0,(3,2)),label='Mean of all donor events')
a2.set_xlim(1974,1998); a2.set_ylabel('Field average vs par'); a2.legend(loc='upper right',fontsize=6.8)
a2.set_title('Riviera 1987–88 against a synthetic Riviera',fontsize=10)
fig.text(0.01,0.01,'Left: year effects from a player-fixed-effects regression restricted to ages 27–38 (birth dates from the PGA Tour archive), where published aging curves are flat;\n5-year centred means. Right: synthetic control from 51 tour events with complete 1975–97 series; placebo p ≈ 0.79. Analysis: Mohiuddin (2026), Fixed Venues.',fontsize=7,color=MUTED)
plt.tight_layout(rect=(0,0.07,1,1)); plt.savefig('/home/claude/work/figs/fig15_aging_synth.png',dpi=200); print('ok')
