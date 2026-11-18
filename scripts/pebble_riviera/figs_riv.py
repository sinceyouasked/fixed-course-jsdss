import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRID="#e6e5e1"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YELLOW="#eda100"; RED="#e34948"
rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.edgecolor":GRID,"axes.linewidth":1,
    "axes.facecolor":SURF,"figure.facecolor":SURF,"savefig.facecolor":SURF,"xtick.color":INK2,"ytick.color":INK2,
    "axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.titleweight":"bold","axes.titlesize":11,"axes.titlelocation":"left",
    "axes.spines.top":False,"axes.spines.right":False,"grid.color":GRID,"grid.linewidth":1,"axes.grid":True,"axes.grid.axis":"y",
    "axes.axisbelow":True,"legend.frameon":False,"legend.fontsize":8.5,"xtick.major.size":0,"ytick.major.size":0})
NAVY=BLUE; GREY=MUTED; GOLD=YELLOW; GREEN=AQUA; AQUA=ORANGE
SRC='Analysis: Mohiuddin (2026), Fixed Venues, SportRxiv preprint.'
s=pd.read_csv('riviera_fsa_series_1926_2026.csv',index_col=0); s=s[s.rounds>=40]
riv=s[s.venue=='Riviera']; rp=s[s.venue=='Rancho Park']; oth=s[s.venue=='other']
# ---------- Fig A: century with championships
fig,ax=plt.subplots(figsize=(12,5.4)); ax.axhline(0,color=GRID,lw=1)
ax.scatter(oth.index,oth.fsa_all,color=GREY,s=18,label='L.A. Open at other courses')
ax.scatter(rp.index,rp.fsa_all,color=AQUA,s=22,label='Rancho Park (17 editions)')
yrs=list(riv.index); vals=list(riv.fsa_all); sx=[];sy=[]
for i,y in enumerate(yrs):
    if i>0 and y-yrs[i-1]>1: ax.plot(sx,sy,color=NAVY,lw=1.1,alpha=0.5); sx=[];sy=[]
    sx.append(y); sy.append(vals[i])
ax.plot(sx,sy,color=NAVY,lw=1.1,alpha=0.5)
ax.scatter(riv.index,riv.fsa_all,color=NAVY,s=26,zorder=3,label='Riviera, L.A. Open / Genesis (63 editions)')
for lo,hi in [(1949,1953),(1973,1982),(1984,1997),(1999,2009),(2010,2018),(2019,2023)]:
    e=riv.loc[lo:hi]; m=e.fsa_all.mean(); ax.hlines(m,lo-0.4,hi+0.4,color=RED,lw=2.2,zorder=4); ax.text((lo+hi)/2,m+0.22,f'{m:+.1f}',ha='center',color=RED,fontsize=9,fontweight='bold')
ch=[(1948,4.51,'1948 U.S. Open'),(1983,2.29,'1983 PGA'),(1995,0.09,'1995 PGA'),(1998,6.65,'1998 U.S. Senior Open'),(2026,2.35,"2026 U.S. Women's Open")]
ax.scatter([c[0] for c in ch],[c[1] for c in ch],marker='D',s=55,color=GOLD,edgecolor='k',zorder=5,label='Championships at Riviera')
for y,v,t in ch: ax.annotate(t,(y,v),xytext=(6,6),textcoords='offset points',fontsize=8,color=INK2)
ax.annotate('limited field,\nno cut',(2025,-1.1),fontsize=8,ha='center',color=NAVY)
ax.set_xlim(1924,2029); ax.set_ylim(-2,7.4); ax.set_ylabel('Field scoring average vs par (FSA+/−)')
ax.set_title('Riviera, 1929–2026: 63 L.A. Opens and five championships on the same eighteen holes',fontsize=12,loc='left')
ax.legend(loc='lower left',frameon=False,fontsize=8.5); ax.spines[['top','right']].set_visible(False)
fig.text(0.01,0.01,'Sources: PGA Tour past-results archive (every player\'s rounds); ESPN (2026 U.S. Women\'s Open). Red bars: era means. 1945–48 omitted (archive incomplete).\n'+SRC,fontsize=6.8,color=MUTED,wrap=True)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig('/home/claude/work/figs/fig12_riviera_century.png',dpi=200); plt.close()
# ---------- Fig B: raw vs weather-adjusted, 1973-2026, with step and limited field
W=pd.read_csv('riviera_series_weather_1973_2026.csv',index_col=0)
fig,ax=plt.subplots(figsize=(12,4.8)); ax.axhline(0,color=GRID,lw=1)
ax.plot(W.index,W.fsa_all,'o-',color=GREY,ms=4,lw=0.8,label='Raw FSA+/−')
adj=W.fsa_all-(0.146*(W.windP-W.windP.mean()))-(-0.531*(W.precP-W.precP.mean()))
ax.plot(W.index,adj,'o-',color=NAVY,ms=4.5,lw=1.0,label='Wind- and rain-adjusted')
for lo,hi,c in [(1973,1987,RED),(1988,2023,RED),(2024,2026,RED)]:
    e=adj.loc[lo:hi]; e=e[e.index.isin(W.index)]; ax.hlines(e.mean(),lo-0.4,hi+0.4,color=c,lw=2.4,zorder=4); ax.text((lo+hi)/2,e.mean()+0.18,f'{e.mean():+.2f}',ha='center',color=c,fontsize=9,fontweight='bold')
ax.axvline(1987.5,color=RED,ls=':',lw=1); ax.axvline(2023.5,color=RED,ls=':',lw=1)
ax.text(1988,2.9,'step −1.0 (p<.001)',color=RED,fontsize=8.5); ax.text(2019.3,2.9,'limited field\n−0.8 to −1.0',color=RED,fontsize=8.5)
ax.set_xlim(1972,2027); ax.set_ylim(-1.8,3.9); ax.set_ylabel('FSA+/− (strokes vs par)')
ax.set_title('Riviera with playing-hours weather removed: a step down in the late 1980s, then level for 35 years, then the limited field',fontsize=11.5,loc='left')
ax.legend(loc='lower left',frameon=False,fontsize=9); ax.spines[['top','right']].set_visible(False)
fig.text(0.01,0.01,'Weather: LAX hourly (1973–2025), Santa Monica airport daily (2026), Thu–Sun 08:00–17:59. Temperature has no effect; wind +0.15/mph, rain −0.53/inch. 2025 (Torrey Pines) excluded.\n'+SRC,fontsize=6.8,color=MUTED,wrap=True)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig('/home/claude/work/figs/fig13_riviera_weather.png',dpi=200); plt.close()
# ---------- Fig C: three venues 1983-2026
aug=pd.read_csv('../osf_in/masters_fsa_1934_2026.csv',index_col=0).fsa
peb=pd.read_csv('../osf_in/weather_pebble_hourly_windows.csv',index_col=0).fsa
fig,ax=plt.subplots(figsize=(12,4.8)); ax.axhline(0,color=GRID,lw=1)
a=aug.loc[1983:]; ax.plot(a.index,a.values,'o-',color=GREEN,ms=4,lw=0.9,label=f'Augusta National, Masters (mean {a.mean():+.2f}, SD {a.std():.2f})')
r=riv.loc[1983:].fsa_all; ax.plot(r.index,r.values,'o-',color=NAVY,ms=4,lw=0.9,label=f'Riviera, L.A. Open/Genesis (mean {r.loc[:2023].mean():+.2f}, SD {r.loc[:2023].std():.2f}, full fields)')
p=peb.loc[2004:2023]; ax.plot(p.index,p.values,'o-',color=AQUA,ms=4,lw=0.9,label=f'Pebble Beach, AT&T (mean {p.mean():+.2f}, SD {p.std():.2f})')
ax.set_xlim(1982,2027); ax.set_ylabel('FSA+/− (strokes vs par)'); ax.set_title('Three fixed venues, same yardstick: the year-to-year amplitude is the same everywhere; the level and the drift are not',fontsize=11.5,loc='left')
ax.legend(loc='upper right',frameon=False,fontsize=8.5); ax.spines[['top','right']].set_visible(False)
fig.text(0.01,0.01,'Augusta added 660 yards and held level; Riviera added about 370 and eased by a stroke; Pebble under tour conditions sits at par.\nMean |year-to-year change|: Augusta 0.86, Riviera 0.85, Pebble 0.9. '+SRC,fontsize=6.8,color=MUTED,wrap=True)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig('/home/claude/work/figs/fig14_three_venues.png',dpi=200); plt.close()
print('figs ok')
