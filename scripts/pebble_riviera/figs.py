import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm
from scipy import stats

OUT = "/home/claude/work/figs"; os.makedirs(OUT, exist_ok=True)
# palette (datavlz reference, light)
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRID="#e6e5e1"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YELLOW="#eda100"; RED="#e34948"; NEUT="#f0efec"
rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.edgecolor":GRID,"axes.linewidth":1,
    "axes.facecolor":SURF,"figure.facecolor":SURF,"savefig.facecolor":SURF,"xtick.color":INK2,"ytick.color":INK2,
    "axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.titleweight":"bold","axes.titlesize":11,"axes.titlelocation":"left",
    "axes.spines.top":False,"axes.spines.right":False,"grid.color":GRID,"grid.linewidth":1,"axes.grid":True,"axes.grid.axis":"y",
    "axes.axisbelow":True,"legend.frameon":False,"legend.fontsize":8.5,"xtick.major.size":0,"ytick.major.size":0})
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
def titles(ax, t, sub, pad=22):
    ax.set_title(t, pad=pad); ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
def src(ax, text):
    ax.text(0, -0.16, text, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")

A = pd.read_csv("fig_aug_series.csv"); H = pd.read_csv("fig_aug_weather.csv")
S = A[(A.year>=1983)&A.fsa.notna()].copy()
Hr = H[(H.year>=1983)&H.resid.notna()].copy()
adj = Hr.set_index("year").resid + S.fsa.mean()


# ---------- F0 ninety Masters ----------
F36 = pd.read_csv("masters_fsa_first36_1934_2026.csv", index_col=0)
Af = A.set_index("year").reindex(range(1934,2027))
Af.loc[[1954,1963],"yds"] = np.nan   # isolated one-year values in the published lists (6,800; 6,850) that revert the next year; treated as unlisted
fig, (ax, ax2) = plt.subplots(2,1, figsize=(9,6.6), gridspec_kw=dict(height_ratios=[3.2,1], hspace=0.26))
ax.axvspan(1942.5, 1945.5, color=NEUT, zorder=0); ax.text(1944, 1.5, "no\nMasters\n1943–45", fontsize=7, color=INK2, ha="center", va="top")
ax.axhline(0, color=GRID, lw=1)
for x,lab in [(1957,"1957\n36-hole cut\nintroduced"),(1981,"1981\nbentgrass\ngreens"),(2002,"2002/06\n+440 yds"),(2019,"2019–\ntee moves")]:
    ax.axvline(x-0.5, color=MUTED, lw=1, ls=(0,(3,3))); ax.text(x-0.2, 6.0, lab, fontsize=7.2, color=INK2, va="top")
ax.plot(F36.index, F36.first36.values, color=AQUA, lw=1.3, alpha=0.9, label="First 36 holes, full field (same instrument in every year)")
ax.plot(Af.index, Af.fsa.values, color=BLUE, lw=2, label="Field scoring average vs par (headline series; four-round mean)")
roll = Af.fsa.rolling(9, center=True, min_periods=5).mean()
ax.plot(roll.index, roll.values, color=ORANGE, lw=2.2, label="9-year centred mean, headline")
for y in (1956,1934,2007,1968,2019):
    v=Af.fsa[y]; ax.annotate(f"{y}  {v:+.2f}", (y,v), xytext=((-54,3) if y==1956 else (5,3) if v>1 else (5,-11)), textcoords="offset points", fontsize=7.8, color=INK2)
ax.set_ylim(-1.4, 6.2); ax.set_xlim(1932.5, 2027.5); ax.set_ylabel("Strokes over par, field average")
titles(ax, "Ninety Masters: Augusta National eased by one to two strokes between the 1950s and the 1980s — before any lengthening",
       "Field scoring average vs par 72, 1934–2026: 1934–82 from the Masters Media Guide (every player's rounds), 1983–2026 Data Golf.\nEra means, headline series: 1934–42 +2.98 · 1946–56 +3.38 · 1957–80 +1.98 · 1981–2001 +1.49 · 2002–18 +1.49 · 2019–26 +0.81", pad=30)
ax.legend(loc="lower left", fontsize=8)
ax2.step(Af.index, Af.yds.ffill().values, where="post", color=AQUA, lw=2)
ax2.set_ylim(6600, 7650); ax2.set_xlim(1932.5, 2027.5); ax2.set_ylabel("Yardage"); ax2.set_yticks([6700,7000,7300,7600])
ax2.text(1934, 7520, "Published yardage, carried forward when unlisted:\n6,700 (1934) → 7,565 (2026); the 1981 step is a re-measurement.", fontsize=7.8, color=INK2, va="top")
src(ax2, "Sources: 2026 Masters Media Guide, Augusta National Golf Club (results 1934–2025); Data Golf tournament archive (1983–2026); published yardages. Analysis: Mohiuddin (2026), Fixed Venues, SportRxiv preprint.")
save(fig, "fig0_ninety")

# ---------- F1 hero ----------
fig, (ax, ax2) = plt.subplots(2,1, figsize=(9,6.2), gridspec_kw=dict(height_ratios=[3,1], hspace=0.28))
ax.axhline(0, color=GRID, lw=1)
for x,lab,yy in [(2002,"2002\n+285 yds",4.1),(2006,"2006\n+155 yds",4.1),(2019,"2019–\ntee moves",4.1)]:
    ax.axvline(x-0.5, color=MUTED, lw=1, ls=(0,(3,3))); ax.text(x-0.3, yy, lab, fontsize=7.5, color=INK2, va="top")
roll = S.set_index("year").fsa.rolling(5, center=True, min_periods=3).mean()
Sg=S.set_index("year").fsa.reindex(range(1983,2027))
ax.plot(Sg.index, Sg.values, color=BLUE, lw=2, solid_joinstyle="round", label="Field scoring average vs par (raw)")
ax.plot(adj.index, adj.values, color=ORANGE, lw=2, label="Weather-adjusted (temperature + wind removed)")
ax.plot(roll.index, roll.values, color=INK2, lw=1.2, ls=(0,(1,2)), label="5-year rolling mean, raw")
for y in (2007,2019,2015,1987):
    v=S.set_index("year").fsa[y]; ax.plot(y, v, "o", ms=7, color=BLUE, mec=SURF, mew=2)
    ax.annotate(f"{y}  {v:+.2f}", (y,v), xytext=((8,-16) if y==2007 else (6, 4 if v>1 else -12)), textcoords="offset points", fontsize=8, color=INK2)
ax.set_ylim(-0.8, 4.2); ax.set_xlim(1982, 2027); ax.set_ylabel("Strokes over par, field average")
titles(ax, "Augusta National plays +1.3 on average — and moves almost a stroke a year", "Masters Tournament field scoring average vs par 72, 1983–2026 (44 tournaments; 1985 and 1988 reconstructed from leaderboards)")
ax.legend(loc="upper left", bbox_to_anchor=(0,0.9), ncol=1)
ax2.step(A[(A.year>=1983)].year, A[(A.year>=1983)].yds, where="post", color=AQUA, lw=2)
ax2.set_ylim(6800, 7650); ax2.set_xlim(1982,2027); ax2.set_ylabel("Yardage"); ax2.set_yticks([6900,7200,7500])
ax2.text(1983, 7600, "Official yardage: 6,905 → 7,565 (+660)", fontsize=8, color=INK2, va="top")
src(ax2, "Source: Data Golf tournament archive (scoring); NCEI LCD hourly, Augusta Bush Field, playing hours (weather); official yardages. Analysis: Mohiuddin (2026), Fixed Venues, SportRxiv preprint.")
save(fig, "fig1_hero")

# ---------- F2 weather ----------
fig, (a1, a2) = plt.subplots(1,2, figsize=(9,3.9), gridspec_kw=dict(wspace=0.3))
reg = Hr.copy()
m = sm.OLS(reg.fsa, sm.add_constant(reg[["tempP","windP"]])).fit()
xs = np.linspace(reg.tempP.min()-1, reg.tempP.max()+1, 50)
a1.scatter(reg.tempP, reg.fsa, s=34, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
a1.plot(xs, m.params["const"]+m.params["tempP"]*xs+m.params["windP"]*reg.windP.mean(), color=INK2, lw=1.5)
for y in (2007,2019,2015,2016):
    r=reg[reg.year==y].iloc[0]; a1.annotate(str(y),(r.tempP,r.fsa),xytext=(5,3),textcoords="offset points",fontsize=8,color=INK2)
a1.set_xlabel("Mean playing-hours temperature, Thu–Sun (°F)"); a1.set_ylabel("Field average vs par")
titles(a1, "Augusta: each 10°F colder ≈ +0.7 strokes", f"1983–2026, n={len(reg)}; r = {reg.fsa.corr(reg.tempP):.2f}; −0.072/°F (p .004)")
P = pd.read_csv("pws_shoreline_playing_hours.csv"); Pr = P[(P.year<2024)&P.gustAvg.notna()]
mp = sm.OLS(Pr.fsa, sm.add_constant(Pr[["gustAvg"]])).fit()
xs2 = np.linspace(3, 19, 20)
a2.scatter(Pr.gustAvg, Pr.fsa, s=34, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3)
a2.plot(xs2, mp.params["const"]+mp.params["gustAvg"]*xs2, color=INK2, lw=1.5)
for y in (2014,2015,2023,2022):
    r=Pr[Pr.year==y].iloc[0]; a2.annotate(str(y),(r.gustAvg,r.fsa),xytext=(5,3),textcoords="offset points",fontsize=8,color=INK2)
a2.set_xlabel("Mean playing-hours gust, shoreline station (mph)"); a2.set_ylabel("Field average vs par")
titles(a2, "Pebble Beach: each 1 mph of gust ≈ +0.17 strokes", f"AT&T Pro-Am 2013–2023, n={len(Pr)}; r = {Pr.fsa.corr(Pr.gustAvg):.2f}; +0.165/mph (p < .001)")
src(a1, "Sources: Data Golf (scoring); NCEI LCD hourly USW00003820 (Augusta); Weather Underground PWS KCAPEBBL4, 17-Mile Drive (Pebble). Playing hours = 08:00–17:59 local.")
save(fig, "fig2_weather")

# ---------- F3 Pebble regimes ----------
import openpyxl
wb=openpyxl.load_workbook("Masters_Paper4_Data_Scaffold.xlsx",data_only=True); ps=wb["Pebble_Paper5"]
rows=[]
for r in ps.iter_rows(min_row=5,max_row=30,values_only=True):
    if r[0] and r[1]: rows.append(dict(year=int(r[0]),event=r[1],peb=r[5] if isinstance(r[5],(int,float)) else None,spy=r[9] if isinstance(r[9],(int,float)) else None))
PB=pd.DataFrame(rows); att=PB[PB.event.str.startswith("AT&T")]; uso=PB[(PB.event=="U.S. Open")&PB.peb.notna()]
SUN = pd.read_csv("pebble_sunday_series_1958_2026.csv", index_col=0)
fig, (ax0, ax) = plt.subplots(2,1, figsize=(9,8.2), gridspec_kw=dict(height_ratios=[1,1], hspace=0.42))
# --- top panel: Sunday at Pebble, 1958–2026 ---
ax0.axhline(0, color=GRID, lw=1)
ax0.axvspan(2003.5, 2026.6, color=NEUT, zorder=0); ax0.text(2026.2, 7.1, "Per-course data\n(lower panel)", fontsize=8, color=INK2, va="top", ha="right")
Sg = SUN.fsa.reindex(range(1958,2027))
ax0.plot(Sg.index, Sg.values, color=BLUE, lw=1.6, marker="o", ms=3.2, mec=SURF, mew=0.8, label="Final round at Pebble Beach, field average vs par 72")
roll = SUN.fsa.rolling(11, center=True, min_periods=6).mean()
ax0.plot(roll.index, roll.values, color=ORANGE, lw=2.2, label="11-year centred mean")
for y in (1960,1964,1975,2020):
    v=SUN.fsa[y]; ax0.annotate(f"{y}  {v:+.1f}", (y,v), xytext=((5,-12) if y==1964 else (5,3) if v>0 else (-44,-13)), textcoords="offset points", fontsize=8, color=INK2)
for y,lab in [(1958,"1960s\n+4.2"),(1970,"1970s\n+2.3"),(1980,"1980s\n+0.7"),(1990,"1990s\n+0.5"),(2000,"2000s\n+0.6"),(2010,"2010s\n−0.2"),(2020,"2020s\n−0.8")]:
    ax0.text(y+ (3.5 if y<2020 else 2.5), -3.15, lab, fontsize=7.5, color=INK2, ha="center", va="bottom")
ax0.set_xlim(1956.5, 2031); ax0.set_ylim(-3.5, 7.6); ax0.set_ylabel("Strokes over par, Sunday field")
titles(ax0, "Sixty-one Sundays at Pebble Beach: the same course has eased by five strokes since the 1960s",
       "Final-round field scoring average vs par 72 at Pebble Beach, Crosby / AT&T Pro-Am 1958–2026 (cut survivors; 8 shortened years omitted). Trend −0.8 per decade; SD around trend 1.3")
ax0.legend(loc="upper left", bbox_to_anchor=(0.27,0.99)); ax0.annotate("2025  −2.3", (2025,SUN.fsa[2025]), xytext=(6,-4), textcoords="offset points", fontsize=8, color=INK2)
# --- bottom panel: per-course 2004–2026 (unchanged) ---
ax.axvspan(2023.5, 2026.6, color=NEUT, zorder=0); ax.text(2023.6, 4.05, "Signature event:\n80 players, no cut", fontsize=8, color=INK2, va="top")
ax.axhline(0, color=GRID, lw=1)
ax.plot(att.year, att.spy, color=AQUA, lw=2, label="Spyglass Hill, same week")
ax.plot(att.year, att.peb, color=BLUE, lw=2, label="Pebble Beach, AT&T Pro-Am (February)")
ax.scatter(uso.year, uso.peb, s=70, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=4, label="Pebble Beach, U.S. Open (June)")
for _,r in uso.iterrows(): ax.annotate(f"U.S. Open {r.year}  {r.peb:+.2f}", (r.year,r.peb), xytext=(8,-3), textcoords="offset points", fontsize=8, color=INK2)
ax.annotate("2026  −2.73", (2026, att[att.year==2026].peb.iloc[0]), xytext=(-42,-12), textcoords="offset points", fontsize=8, color=INK2)
ax.set_xlim(2003.4, 2026.9); ax.set_ylim(-3.2, 4.4); ax.set_ylabel("Field average vs par")
titles(ax, "Same course, three difficulties: who plays and how it's set up move Pebble by 2–4 strokes", "Per-course field scoring averages, 2004–2026. Tour-week baseline +0.02 (2004–23); U.S. Open weeks +3.98 and +1.22; signature era −2.35")
ax.legend(loc="lower left")
src(ax, "Sources: PGA Tour past-results archive (final-round scores, 1958–2026; final round always at Pebble Beach); Data Golf per-course pages (2004–2026).\nSpyglass Hill − Pebble Beach same-week mean +0.40 (95% CI +0.15 to +0.65, 23 pairs). Analysis: Mohiuddin (2026), Fixed Venues, SportRxiv preprint.")
save(fig, "fig3_pebble")

# ---------- F4 decomposition ----------
fig, ax = plt.subplots(figsize=(8.6,5.2))
labels=["Championship week: USGA, June\nU.S. Open vs AT&T week, same players (Pebble)","Championship week: PGA of America, August\nPGA vs L.A. Open, same players (Riviera, 1983 and 1995)","Field composition\nlimited field vs regular era (Pebble; Riviera)","Weather\n10°F colder at Augusta / 10 mph more wind at Pebble, Riviera","Architecture\nSpyglass Hill vs Pebble Beach, same week","Lengthening\nAugusta +440 yds (2002, 2006), weather-adjusted"]
y=np.arange(len(labels))[::-1]
# setup
ax.barh(y[0], 2.96, height=0.5, color=BLUE, zorder=3); ax.errorbar([2.96],[y[0]], xerr=[[2.96-2.35],[3.55-2.96]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
ax.text(3.62, y[0], "+2.96  [2.35, 3.55]", fontsize=8, color=INK2, va="center")
# PGA of America championships at Riviera (matched players): 1983 -0.27 (SE .32), 1995 -1.24 (SE .25)
ax.barh(y[1]+0.14, -0.27, height=0.26, color=YELLOW, zorder=3); ax.errorbar([-0.27],[y[1]+0.14], xerr=[[1.96*0.32],[1.96*0.32]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
ax.barh(y[1]-0.14, -1.24, height=0.26, color=YELLOW, zorder=3); ax.errorbar([-1.24],[y[1]-0.14], xerr=[[1.96*0.25],[1.96*0.25]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
ax.text(0.45, y[1]+0.14, "1983  −0.27 [−0.9, +0.4]", fontsize=8, color=INK2, va="center"); ax.text(0.45, y[1]-0.14, "1995  −1.24 [−1.7, −0.8]", fontsize=8, color=INK2, va="center")
# composition split: Pebble (upper) and Riviera (lower)
yp=y[2]+0.15; yr=y[2]-0.17
ax.barh(yp, 0.66, height=0.26, color="#b9cfea", zorder=3); ax.barh(yp, 1.41, left=0.66+0.02, height=0.26, color=BLUE, zorder=3); ax.barh(yp, 2.29-2.09, left=2.09+0.02, height=0.26, color=GRID, zorder=3); ax.errorbar([2.29],[yp], xerr=[[2.29-1.84],[2.74-2.29]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
ax.text(0.33, yp, "skill 0.66", fontsize=7, color=INK, va="center", ha="center"); ax.text(0.68+1.41/2, yp, "event itself 1.41", fontsize=7, color="white", va="center", ha="center")
ax.text(2.82, yp, "Pebble 2.3 total [1.8, 2.7]", fontsize=8, color=INK2, va="center")
ax.barh(yr, 0.31, height=0.26, color="#f5d9a8", zorder=3); ax.barh(yr, 0.43, left=0.31+0.02, height=0.26, color=YELLOW, zorder=3); ax.barh(yr, 1.13-0.76, left=0.76+0.02, height=0.26, color=GRID, zorder=3); ax.errorbar([0.76],[yr], xerr=[[1.96*0.17],[1.96*0.17]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
ax.text(1.25, yr, "Riviera 1.1 total (who plays 0.31; same players −0.43 [−0.76, −0.10])", fontsize=8, color=INK2, va="center")
# weather
ax.barh(y[3], 0.72, height=0.5, color=BLUE, zorder=3); ax.errorbar([0.72],[y[2]], xerr=[[0.51],[0.51]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
ax.scatter([1.65],[y[3]], s=50, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=5); ax.scatter([1.49],[y[3]-0.3], s=50, color=YELLOW, edgecolor=SURF, linewidth=1.5, zorder=5)
ax.text(0.97, y[3]+0.36, "Augusta temp (10°F)  +0.72 [0.21, 1.23]", fontsize=8, color=INK2, va="center"); ax.text(1.73, y[3], "Pebble gust (10 mph)  +1.65", fontsize=8, color=INK2, va="center"); ax.text(1.6, y[3]-0.3, "Riviera wind (10 mph)  +1.49; rain −0.56/inch", fontsize=8, color=INK2, va="center")
# architecture
ax.barh(y[4], 0.40, height=0.5, color=BLUE, zorder=3); ax.errorbar([0.40],[y[4]], xerr=[[0.25],[0.25]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
ax.text(0.72, y[4], "+0.40", fontsize=8, color=INK2, va="center")
# lengthening
ax.text(0.05, y[5], "not detectable at course level (p = 0.97, 0.27); +0.10 to +0.17 on the lengthened hole", fontsize=8, color=INK2, va="center")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-1.9, 5.4); ax.set_ylim(-0.6, 5.6); ax.grid(axis="x"); ax.grid(False, axis="y")
ax.set_xlabel("Strokes per round, field average"); titles(ax, "What moves a fixed course, ranked — and the largest lever belongs to the organizer", "Effect sizes with 95% intervals. Blue = Pebble Beach / Augusta; yellow = Riviera. Light segment = part of composition explained by who plays.")
src(ax, "Sources: this study, Sections 4, 5, 8 and 9. Championship week: 91 matched players at Pebble; 94 and 82 at Riviera. Composition: 5,212 Pebble rounds; 73 matched Riviera players. Architecture: 23 same-week pairs. Weather: regressions.")
save(fig, "fig4_levers")

# ---------- F8 club waterfall (illustrative) ----------
fig, ax = plt.subplots(figsize=(9.2,4.2))
steps=[("Course Rating\nfixed ~10 yrs",0.0,"base"),("Weather\nwet, cold spring",+0.5,"d"),("Setup & conditions\ngreens, rough",+0.4,"d"),("Field mix\nwho posted",-0.6,"d"),("Tee mix\nback-tee share",+0.2,"d"),("This season's\nactual difficulty",None,"total")]
x=np.arange(len(steps)); cum=0; base=71.8
for k,(lab,v,kind) in enumerate(steps):
    if kind=="base": ax.bar(k, 0.0001, bottom=base, color=INK2); ax.text(k, base+0.05, f"{base:.1f}", ha="center", fontsize=8.5, color=INK2); cum=base
    elif kind=="d":
        col = RED if v>0 else AQUA; bottom = cum if v>0 else cum+v
        ax.bar(k, abs(v), bottom=bottom, color=col, width=0.55, zorder=3); ax.plot([k-1+0.275,k-0.275],[cum,cum], color=MUTED, lw=1)
        ax.text(k, cum+v+(0.06 if v>0 else -0.14), f"{v:+.1f}", ha="center", fontsize=8.5, color=INK2); cum+=v
    else:
        ax.bar(k, cum-base, bottom=base, color=BLUE, width=0.55, zorder=3); ax.plot([k-1+0.275,k-0.275],[cum,cum], color=MUTED, lw=1); ax.text(k, cum+0.06, f"{cum:.1f}", ha="center", fontsize=8.5, color=INK2); ax.text(k+0.33, cum-0.25, f"+{cum-base:.1f}\nvs rating", ha="left", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=7.8)
ax.set_ylim(base-0.9, base+1.5); ax.set_ylabel("Scoring average, tee-normalized (strokes)")
ax.set_yticks([71.0,71.5,72.0,72.5,73.0]); ax.set_xlim(-0.6,5.9)
titles(ax, "Why a club's course plays off its rating in any given season — illustrative", "Championship-scaled estimates for a typical private club (Section 10); a pilot season of member scoring replaces these with measured values.")
ax.text(0, -0.2, "Red = made the course harder this season; green = easier. Weather and setup are what a club can measure quarterly;\nfield and tee mix are what a handicap-adjusted series removes.", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "fig8_club_waterfall")

# ---------- F5 benchmark ----------
bench = {"Royal St George's":2.88,"Pebble Beach (U.S. Open)":2.76,"Shinnecock Hills":1.71,"St Andrews":1.52,"Royal Liverpool":1.12,"Royal Portrush":1.09,"Quail Hollow":0.63,"Oak Hill":0.55,"Royal Birkdale":0.54,"Oakmont":0.49,"Royal Troon":0.45,"Whistling Straits":0.40,"Pinehurst No. 2":0.31,"Valhalla":0.31,"Kiawah Island":0.06}
aug_yoy = S.set_index("year").fsa.diff().abs().dropna(); aug_yoy = aug_yoy[[ (y-1) in S.year.values for y in aug_yoy.index]]
peb_yoy = att[att.year<=2023].set_index("year").peb.diff().abs().dropna()
RIV = pd.read_csv("/home/claude/work/riv/riviera_fsa_series_1926_2026.csv", index_col=0); RIV = RIV[(RIV.venue=="Riviera")&(RIV.index>=1973)&(RIV.index<=2023)].fsa_all
riv_yoy = RIV.diff().abs().dropna(); riv_yoy = riv_yoy[[ (y-1) in RIV.index for y in riv_yoy.index]]
fig, ax = plt.subplots(figsize=(9,4.1))
rows_=[("Rotating major venues\nbetween visits, 2010–26 (15 pairs)", list(bench.values()), BLUE),("Augusta National\nconsecutive Masters (43 pairs)", list(aug_yoy.values), ORANGE),("Pebble Beach, tour setup\nconsecutive AT&Ts (19 pairs)", list(peb_yoy.values), AQUA),(f"Riviera, tour setup\nconsecutive L.A. Opens 1973–2023 ({len(riv_yoy)} pairs)", list(riv_yoy.values), YELLOW)]
rng=np.random.default_rng(3)
for i,(lab,vals_,col) in enumerate(rows_):
    yy = 3-i + rng.uniform(-0.16,0.16,len(vals_))
    ax.scatter(vals_, yy, s=36, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
    med=np.median(vals_); ax.plot([med,med],[3-i-0.32,3-i+0.32], color=INK, lw=2, zorder=4)
    ax.text(med, 3-i+0.38, f"median {med:.2f}", fontsize=8, color=INK2, ha="center")
ax.set_yticks([3,2,1,0]); ax.set_yticklabels([r[0] for r in rows_], fontsize=8.5); ax.grid(False, axis="y"); ax.grid(True, axis="x")
ax.set_xlim(-0.1, 3.1); ax.set_xlabel("Absolute change in field average vs par (strokes)")
titles(ax, "A course that never changes moves as much between years as the rota moves between visits", "Each dot is one pair of observations; the bar is the median. Three fixed venues, three organizers, one amplitude")
src(ax, "Sources: Data Golf; venue pairs from Papers 1–3 (U.S. Open, PGA, Open Championship). Augusta and Riviera pairs use consecutive years with data on both sides; Riviera from the PGA Tour archive.")
save(fig, "fig5_benchmark")

# ---------- F6 heatmap ----------
df = pd.read_csv("masters_hole_rounds_1983_2026.csv"); df["strokes"]=(df.par+df.rel)*df.played
g = df.groupby(["year","hole"]).agg(par=("par","first"),played=("played","sum"),strokes=("strokes","sum")).reset_index()
g["rel"]=g.strokes/g.played-g.par; piv=g.pivot(index="year",columns="hole",values="rel")
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUT, RED])
fig, ax = plt.subplots(figsize=(9,9.0))
fig.subplots_adjust(top=0.90)
im = ax.imshow(piv.values, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(18)); ax.set_xticklabels([f"{h}\npar {int(g[g.hole==h].par.iloc[-1])}" for h in range(1,19)], fontsize=8)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7.5); ax.grid(False)
ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
for k,y in enumerate(piv.index):
    if y in (2002,2006,2019): ax.axhline(k-0.5, color=INK, lw=1)
for (h,y) in [(5,2019),(11,2022),(15,2022),(13,2023),(2,2024),(17,2026),(7,2006),(11,2006),(18,2002),(11,2002)]:
    if y in piv.index: ax.add_patch(plt.Rectangle((h-1-0.5, list(piv.index).index(y)-0.5),1,1,fill=False,edgecolor=INK,lw=1.3))
for y,lab in [(2002,"2002 lengthening"),(2006,"2006 lengthening"),(2019,"2019– tee moves")]:
    ax.text(18.0, list(piv.index).index(y)-0.5, lab, fontsize=7.5, color=INK2, va="center")
cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.12); cb.set_label("Strokes vs par, tournament average", fontsize=8.5); cb.outline.set_visible(False)
fig.text(0.01, 0.985, "Where Augusta's difficulty lives: 18 holes × 42 Masters", fontsize=11, fontweight="bold", color=INK, va="top")
fig.text(0.01, 0.962, "Hardest vs par: No. 11 (+0.29), No. 4 (+0.26), No. 1 (+0.25). Easiest: the par 5s. Boxes mark the year a hole was lengthened.", fontsize=8.5, color=INK2, va="top")
ax.text(0,-0.03,"Source: Data Golf per-round hole tables, rounds-weighted. Blue = under par, red = over par.", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
fig.savefig(f"{OUT}/fig6_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# ---------- F7 (Figure 3): hole-year panel ----------
E = pd.read_csv("panel_event_study.csv"); E["per10"] = E.res_fe/(E.d/10)
es = E.groupby("k").agg(mean=("res_fe","mean"), se=("res_fe", lambda s: s.std()/np.sqrt(len(s))), n=("res_fe","size"))
fig, (a1, a2) = plt.subplots(1,2, figsize=(9,4.0), gridspec_kw=dict(wspace=0.35, width_ratios=[1.25,1]))
a1.axhline(0, color=GRID, lw=1); a1.axvline(-0.5, color=MUTED, lw=1, ls=(0,(3,3)))
a1.errorbar(es.index, es["mean"], yerr=1.96*es.se, fmt="o-", color=BLUE, ecolor=INK2, elinewidth=1, capsize=2.5, ms=5, mec=SURF, mew=1.2, lw=1.6)
a1.text(0.2, 0.135, "change\nin play", fontsize=7.5, color=INK2, va="top")
a1.set_xlabel("Years before / after the lengthening"); a1.set_ylabel("Hole scoring vs par, net of hole and year effects")
a1.set_ylim(-0.12, 0.15); a1.set_xticks(range(-6,7))
a1.set_title("Event study: 23 lengthenings of 15+ yards (mean 28 yds)", fontsize=9.5, loc="left", pad=6, color=INK2, fontweight="normal")
coef = [("Length, per 10 yds added", 0.0304, 0.0055, BLUE), ("Green altered\n(contours, hole locations)", -0.0646, 0.0216, ORANGE), ("Trees added", 0.0349, 0.0283, MUTED), ("Bunkers / fairway", 0.0032, 0.0227, MUTED), ("Trees removed", 0.0053, 0.0228, MUTED)]
y = np.arange(len(coef))[::-1]
a2.axvline(0, color=GRID, lw=1)
for yy,(lab,b,se,col) in zip(y,coef):
    a2.errorbar(b, yy, xerr=1.96*se, fmt="o", color=col, ecolor=INK2, elinewidth=1, capsize=2.5, ms=6, mec=SURF, mew=1.2)
    a2.text(b+1.96*se+0.008, yy, f"{b:+.3f}", fontsize=7.8, color=INK2, va="center")
a2.set_yticks(y); a2.set_yticklabels([c[0] for c in coef], fontsize=8); a2.set_xlim(-0.16, 0.13); a2.set_xlabel("Strokes on the hole, relative to unchanged holes")
a2.grid(axis="x"); a2.grid(axis="y", visible=False)
a2.set_title("Two-way fixed effects, by kind of change", fontsize=9.5, loc="left", pad=6, color=INK2, fontweight="normal")
fig.suptitle("A changed hole moves relative to the other seventeen: length adds strokes, green work removes them", x=0.125, y=1.06, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.125, 0.99, "Hole-year panel, 42 Masters × 18 holes, 1983–2026, hole and year fixed effects; SE clustered by hole; 95% bands", fontsize=8.5, color=INK2)
a1.text(0, -0.2, "Sources: Data Golf per-round hole tables 1983–2026 (no tables for 1985, 1988); official scorecard yardages; change chronology from the 2026 Masters Media Guide.\nYear effects absorb the course total: the design is silent on it by construction.", transform=a1.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "fig7_teemoves")

# ---------- F9 signatures: benchmark venues sorted by amplitude ----------
bench = {"Royal St George's":(2.88,"Open"),"Pebble Beach (U.S. Open)":(2.76,"U.S. Open"),"Shinnecock Hills":(1.71,"U.S. Open"),"St Andrews":(1.52,"Open"),"Royal Liverpool":(1.12,"Open"),"Royal Portrush":(1.09,"Open"),"Quail Hollow":(0.63,"PGA"),"Oak Hill":(0.55,"PGA"),"Royal Birkdale":(0.54,"Open"),"Oakmont":(0.49,"U.S. Open"),"Royal Troon":(0.45,"Open"),"Whistling Straits":(0.40,"PGA"),"Pinehurst No. 2":(0.31,"U.S. Open"),"Valhalla":(0.31,"PGA"),"Kiawah Island":(0.06,"PGA")}
items=sorted(bench.items(), key=lambda kv: kv[1][0])
fig, ax = plt.subplots(figsize=(8.6,5.2))
yy=np.arange(len(items)); cmap={"Open":AQUA,"U.S. Open":BLUE,"PGA":ORANGE}
ax.barh(yy,[v[0] for k,v in items],height=0.55,color=[cmap[v[1]] for k,v in items],zorder=3)
for i,(k,v) in enumerate(items): ax.text(v[0]+0.05,i,f"{v[0]:.2f}",va="center",fontsize=8,color=INK2)
ax.axvline(0.86,color=INK2,lw=1.2,ls=(0,(3,3))); ax.text(0.95,6.5,"Augusta 0.86 / Pebble (tour) 0.91:\ntypical year-to-year move at the\ntwo fixed venues",fontsize=7.5,color=INK2,va="center")
ax.set_yticks(yy); ax.set_yticklabels([k for k,v in items],fontsize=8.5); ax.grid(False,axis="y"); ax.grid(True,axis="x")
ax.set_xlim(0,3.3); ax.set_xlabel("Change in field average vs par between visits (strokes)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="U.S. Open"),Patch(color=AQUA,label="Open Championship"),Patch(color=ORANGE,label="PGA Championship")],loc="lower right")
titles(ax,"Every course has its own amplitude: 0.06 to 2.88 strokes on the same instrument","Raw field scoring average vs par at each visit, elite championship fields, no adjustment for weather or field. Fifteen venues that hosted twice since 2010.")
src(ax,"Source: Data Golf field averages; venue pairs from Papers 1–3. One pair per venue (St Andrews: largest of three visits). Amplitude ≠ quality: Pebble and Kiawah are both exceptional courses.")
save(fig,"fig9_signatures")

# ---------- F10 (Figure 4): counterfactuals ----------
C = pd.read_csv("counterfactual_no_lengthening_1983_2026.csv", index_col=0)
fig, (a1, a2) = plt.subplots(1,2, figsize=(9.6,4.0), gridspec_kw=dict(wspace=0.28))
a1.axhline(0, color=GRID, lw=1)
a1.plot(C.index, C.fsa, color=BLUE, lw=1.8, label="Actual field average vs par")
a1.plot(C.index, C.cf_pooled, color=ORANGE, lw=1.6, ls=(0,(4,2)), label="Without the lengthening (panel rate, 0.030/10 yds)")
a1.plot(C.index, C.cf_event, color=ORANGE, lw=1.1, ls=(0,(1,2)), label="Without the lengthening (event-study rate, 0.020/10 yds)")
a1.fill_between(C.index, C.cf_pooled, C.cf_event, color=ORANGE, alpha=0.10)
a1.set_ylabel("Strokes over par, field average"); a1.set_ylim(-2.4, 5.2)
a1.legend(loc="upper left", fontsize=7.2, frameon=False)
a1.set_title("Had Augusta not lengthened: one to two strokes easier", fontsize=9.5, loc="left", pad=6, color=INK2, fontweight="normal")
a2.plot(C.index, C.drive_gain, color=INK2, lw=1.8, label="Tour average driving distance, gain since 1983")
a2.plot(C.index, C.yds_added/18, color=AQUA, lw=1.8, label="Augusta yardage added since 1983, per hole")
a2.set_ylabel("Yards"); a2.set_ylim(0, 50); a2.legend(loc="upper left", fontsize=7.5)
a2.set_title("The lengthening kept pace with the ball: 37 yds/hole vs 44 yds", fontsize=9.5, loc="left", pad=6, color=INK2, fontweight="normal")
fig.suptitle("Two counterfactuals: the course without its lengthening, and the lengthening against the distance gain", x=0.125, y=1.05, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.125, 0.985, "Counterfactual path = actual minus the panel's relative effect of cumulative yards added. Distance: PGA Tour season averages; 2025 from the USGA Annual Drive Distance Report; 2026 carried forward.", fontsize=8, color=INK2)
a1.text(0, -0.18, "Sources: Data Golf (scoring); official yardages; hole-year panel (Section 4); PGA Tour / USGA–R&A driving-distance series. The counterfactual takes the untouched holes as the\ncounterfactual for the lengthened ones; it is a relative-effect construction, not a forecast, and the event-study rate is its lower bound.", transform=a1.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "fig10_counterfactual")

