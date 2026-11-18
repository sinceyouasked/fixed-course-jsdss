# EIV wind correction (Section 9) and metalog backtest (Appendix A3)
import pandas as pd, numpy as np, statsmodels.api as sm
W=pd.read_csv('../riviera_series_weather_1973_2026.csv',index_col=0); o=W.dropna(subset=['lax_windP','smo_windP'])
m=sm.OLS(o.smo_windP,sm.add_constant(o.lax_windP)).fit(); lam=m.params.iloc[1]
print('reliability ratio',round(lam,3),'corrected wind coef',round(0.149/lam,3))
def metalog3(q10,q50,q90):
    ys=np.array([.1,.5,.9]); Y=np.column_stack([np.ones(3),np.log(ys/(1-ys)),(ys-.5)*np.log(ys/(1-ys))]); return np.linalg.solve(Y,np.array([q10,q50,q90]))
def Mq(a,y): return a[0]+a[1]*np.log(y/(1-y))+a[2]*(y-.5)*np.log(y/(1-y))
def crps(a,obs,n=20001):
    y=np.linspace(1/(n+1),n/(n+1),n); x=Mq(a,y); return np.mean(np.abs(x-obs))-0.5*np.mean(np.abs(x[:,None][::40]-x[None,:][:, ::40]))
opens={2000:4.31,2010:3.98,2019:1.22}; base=0.02; sd_tour=0.79
for yr,obs in opens.items():
    others=[v for k,v in opens.items() if k!=yr]; med=base+np.mean(others)-base; sd=np.sqrt(np.std(others,ddof=1)**2+sd_tour**2)
    a=metalog3(med-1.2816*sd,med,med+1.2816*sd); print(yr,'LOO median',round(med,2),'CRPS',round(crps(a,obs),2))
    print('  baseline-only CRPS',round(crps(metalog3(base-1.2816*sd_tour,base,base+1.2816*sd_tour),obs),2))
