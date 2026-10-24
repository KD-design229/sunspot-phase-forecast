#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce a one-month-ahead solar-activity benchmark on authentic WDC-SILSO v2.0 data.

Data period: 1990-01..2025-12 (432 authentic monthly sunspot numbers).
Training 1993-01..2015-12; validation 2016-01..2020-12; test 2021-01..2025-12.
Each predictor uses previous 36 *observed* months. Rolling one-step testing
is NOT a five-year recursive forecast. Preprocessing sees training targets only.

Data provenance: WDC-SILSO, Royal Observatory of Belgium, Brussels,
DOI https://doi.org/10.24414/qnza-ac80; source monthly V2.0 file
https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt, accessed 2026-10-24.
SILSO source data licensed CC BY-NC 4.0. A dated exact snapshot is bundled.
Option --verify-official downloads and compares source values; does not
substitute a possibly revised file into the published benchmark.
"""
import argparse, json, hashlib, sys, os, random, platform, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
RAW=ROOT/'SILSO_monthly_1990_2025_snapshot.csv'
OFFICIAL_URL='https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt'
SEEDS=[7,42,73]
LOOKBACK=36
TRAIN_END='2015-12';VAL_END='2020-12';TEST_END='2025-12'
COL={'obs':'#243B4A','lstm':'#BA655A','gru':'#925D8B','mlp':'#D1A25B','ridge':'#308080','persist':'#77898B'}
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'figure.dpi':125,
    'savefig.dpi':300,'axes.titlesize':11,'font.size':9,'axes.labelsize':9,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False})
torch.set_num_threads(1)

parser=argparse.ArgumentParser()
parser.add_argument('--verify-official',action='store_true',help='Compare dated snapshot with current SILSO official raw data')
args=parser.parse_args()

df=pd.read_csv(RAW)
assert df.shape[0]==432 and df.date.iloc[0]=='1990-01' and df.date.iloc[-1]=='2025-12'
assert list(df.columns)==['date','year','month','sunspot_number']
assert (pd.period_range('1990-01','2025-12',freq='M').astype(str)==df.date).all()
assert df.sunspot_number.notna().all() and (df.sunspot_number>=0).all()
sha=hashlib.sha256(RAW.read_bytes()).hexdigest()
if args.verify_official:
    txt=urllib.request.urlopen(OFFICIAL_URL,timeout=25).read().decode('utf8')
    data={f'{int(a[0]):04d}-{int(a[1]):02d}':float(a[3]) for ln in txt.splitlines() if len(a:=ln.split())>=4}
    mism=[(z.date,z.sunspot_number,data.get(z.date)) for z in df.itertuples() if data.get(z.date)!=z.sunspot_number]
    print('Official-vs-snapshot mismatches:',len(mism),'first ten:',mism[:10])
    if mism:raise RuntimeError('Snapshot differs from current SILSO: a revision may have occurred')

values=df.sunspot_number.to_numpy('float32')
dates=df.date.to_numpy()
indices=np.arange(LOOKBACK,len(df))
X=np.stack([values[i-LOOKBACK:i] for i in indices]);y=values[indices];when=dates[indices]
train=(when<=TRAIN_END);val=(when>TRAIN_END)&(when<=VAL_END);test=(when>VAL_END)&(when<=TEST_END)
assert (train.sum(),val.sum(),test.sum())==(276,60,60)
center=float(y[train].mean());spread=float(y[train].std());assert spread>0
xs=((X-center)/spread).astype('float32');ys=((y-center)/spread).astype('float32')

def E(yt,yp):
    return {'MAE':float(mean_absolute_error(yt,yp)),
            'RMSE':float(np.sqrt(mean_squared_error(yt,yp))),
            'R2':float(r2_score(yt,yp)),
            'Bias':float(np.mean(yp-yt))}

pred={}
pred['持续性基线']=X[:,-1].copy()
pred['前3月均值']=X[:,-3:].mean(axis=1)
# 11-year-cycle benchmark is available for target months starting 2001-01.
lag132=np.array([values[i-132] if i>=132 else np.nan for i in indices],dtype='float32')
pred['11年滞后']=lag132
ridge_cands=[]
for alpha in [0.01,.1,1.,10.,100.,1000.]:
    model=Ridge(alpha=alpha).fit(xs[train],ys[train]);p=np.maximum(0,model.predict(xs)*spread+center)
    ridge_cands.append((E(y[val],p[val])['RMSE'],alpha))
ra=float(min(ridge_cands)[1]);ridge=Ridge(alpha=ra).fit(xs[train],ys[train])
pred['岭回归AR(36)']=np.maximum(0,ridge.predict(xs)*spread+center)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(LOOKBACK,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).flatten()
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.LSTM(1,16,batch_first=True)
        self.out=nn.Sequential(nn.Linear(16,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):
        h,_=self.rnn(x.unsqueeze(-1));return self.out(h[:,-1,:]).flatten()
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(1,16,batch_first=True)
        self.out=nn.Sequential(nn.Linear(16,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):
        h,_=self.rnn(x.unsqueeze(-1));return self.out(h[:,-1,:]).flatten()

def fit(mclass,seed,max_epochs=260,patience=40):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=mclass();opt=torch.optim.Adam(model.parameters(),lr=.004,weight_decay=.0003)
    los=nn.MSELoss()
    tx=torch.from_numpy(xs[train]);ty=torch.from_numpy(ys[train]);vx=torch.from_numpy(xs[val]);vy=torch.from_numpy(ys[val])
    loader=DataLoader(TensorDataset(tx,ty),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');keep=None;epochs_no_imp=0;hist=[]
    for epoch in range(max_epochs):
        model.train()
        for a,b in loader:
            opt.zero_grad();l=los(model(a),b);l.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval()
        with torch.no_grad(): tl=float(los(model(tx),ty));vl=float(los(model(vx),vy))
        hist.append((epoch+1,tl,vl))
        if vl < best-0.00005:
            best=vl;keep={k:z.detach().clone() for k,z in model.state_dict().items()};epochs_no_imp=0
        else:epochs_no_imp+=1
        if epochs_no_imp>=patience:break
    model.load_state_dict(keep);model.eval()
    with torch.no_grad():allp=model(torch.from_numpy(xs)).numpy()
    p=np.maximum(0,allp*spread+center)
    return {'seed':seed,'val_rmse':E(y[val],p[val])['RMSE'],'test_rmse':E(y[test],p[test])['RMSE'],
            'epochs':len(hist),'prediction':p,'history':hist}

runs={};best_models={}
for name,model_cls in [('MLP',MLP),('LSTM',LSTM),('GRU',GRU)]:
    print('Training',name,flush=True)
    trials=[fit(model_cls,s) for s in SEEDS]
    selected=min(trials,key=lambda z:z['val_rmse'])
    pred[name]=selected['prediction'];best_models[name]=selected
    runs[name]=[{k:v for k,v in t.items() if k not in ('prediction','history')} for t in trials]
    print(name,'best seed',selected['seed'],'val RMSE',round(selected['val_rmse'],3),'test RMSE',round(selected['test_rmse'],3),flush=True)

metrics={n:{s:E(y[mask],p[mask]) for s,mask in [('train',train),('validation',val),('test',test)] if np.isfinite(p[mask]).all()} for n,p in pred.items()}

# Circular moving block bootstrap of paired test months; 12 month blocks.
rng=np.random.default_rng(20261024)
def paired_block_interval(a,b,rep=3000,block=12):
    ids=np.flatnonzero(test);n=len(ids);d=[]
    for _ in range(rep):
        selected=[]
        while len(selected)<n:
            start=int(rng.integers(n));selected.extend((start+np.arange(block))%n)
        r=ids[np.array(selected[:n],dtype=int)]
        d.append(np.sqrt(np.mean((pred[a][r]-y[r])**2))-np.sqrt(np.mean((pred[b][r]-y[r])**2)))
    return [float(i) for i in np.quantile(d,[.025,.5,.975])]
ci={f'{n}_minus_Ridge':paired_block_interval(n,'岭回归AR(36)') for n in ['MLP','LSTM','GRU']}
ci['MLP_minus_Persistence']=paired_block_interval('MLP','持续性基线')

result={'data_source':'WDC-SILSO monthly mean total sunspot number v2.0; DOI 10.24414/qnza-ac80',
        'raw_data_url':OFFICIAL_URL,'snapshot_accessed':'2026-10-24',
        'snapshot_file':RAW.name,'snapshot_sha256':sha,
        'raw_months':len(df),'span':'1990-01 to 2025-12',
        'test_period':'2021-01 to 2025-12','train_period':'1993-01 to 2015-12',
        'validation_period':'2016-01 to 2020-12',
        'counts':{'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum())},
        'lag_months':LOOKBACK,'ridge_alpha':ra,'chosen_seed_and_epochs':{k:{'seed':int(t['seed']),'epochs':int(t['epochs']),'validation_RMSE':float(t['val_rmse'])} for k,t in best_models.items()},
        'seed_experiments':runs,'scores':metrics,'bootstrap_rmse_differences':ci,
        'raw_peak_2010_2025':df[(df.date>='2010-01')].sort_values('sunspot_number',ascending=False).head(1).to_dict('records')[0],
        'software':{'python':sys.version.split()[0],'torch':torch.__version__,'sklearn':__import__('sklearn').__version__,
                    'numpy':np.__version__,'pandas':pd.__version__},
        'note':'SILSO authentic monthly series; chronological one-month-ahead with previous observed months in test; not multi-year recursive. No published DOI assigned to this manuscript.'}
(ROOT/'metrics_recent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
pt=pd.DataFrame({'date':when,'observed':y,'split':np.where(train,'train',np.where(val,'validation','test'))})
for k,v in pred.items():pt[k]=v
pt.to_csv(ROOT/'predictions_recent_all.csv',index=False,float_format='%.5f')

def save(name):plt.tight_layout();plt.savefig(FIG/name,dpi=300,bbox_inches='tight');plt.close()

recent=df[df.year>=2010]
fig,ax=plt.subplots(figsize=(7.1,3.1))
ax.axvspan(pd.Timestamp('2010-01-01'),pd.Timestamp('2015-12-31'),color='#e8eeec',label='训练期(近期部分)',alpha=.7)
ax.axvspan(pd.Timestamp('2016-01-01'),pd.Timestamp('2020-12-31'),color='#f8ecd9',label='验证期',alpha=.8)
ax.axvspan(pd.Timestamp('2021-01-01'),pd.Timestamp('2025-12-31'),color='#f1e7e5',label='独立测试期',alpha=.8)
ax.plot(pd.to_datetime(recent.date),recent.sunspot_number,color=COL['obs'],lw=1.15,label='SILSO月度实测')
ax.set(title='2010—2025年太阳黑子数真实月度观测',ylabel='月度平均太阳黑子数',xlabel='年份')
ax.legend(ncol=2,loc='upper left',frameon=False);ax.grid(alpha=.14);save('fig1_SILSO_recent_observations.png')

fig,ax=plt.subplots(figsize=(7.1,3.1))
valid=df[df.date>='2010-01']
sm=valid.sunspot_number.rolling(window=13,center=True,min_periods=13).mean()
ax.plot(pd.to_datetime(valid.date),valid.sunspot_number,color='#BDC7C5',lw=.9,label='月度原始值')
ax.plot(pd.to_datetime(valid.date),sm,color=COL['ridge'],lw=2.1,label='13个月中心滑动平均（展示用）')
ax.axhline(0,color='#a1acad',lw=.65)
ax.set(title='第24—25太阳活动周期的观测变化（平滑仅用于展示）',ylabel='月度平均太阳黑子数',xlabel='年份')
ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.14);save('fig2_cycle_comparison.png')

fig,ax=plt.subplots(figsize=(7.1,3.1))
tt=pd.to_datetime(when[test]);ax.plot(tt,y[test],color=COL['obs'],lw=1.7,label='实际观测')
for name,k in [('岭回归AR(36)','ridge'),('MLP','mlp'),('LSTM','lstm'),('GRU','gru')]:
 ax.plot(tt,pred[name][test],lw=1.1,alpha=.88,label=name,color=COL[k])
ax.set(title='2021—2025年独立测试期：真实观测与一步预测',ylabel='月度平均太阳黑子数',xlabel='预测目标月份')
ax.legend(ncol=3,loc='upper left',frameon=False);ax.grid(alpha=.15);save('fig3_2021_2025_predictions.png')

fig,ax=plt.subplots(figsize=(7.1,3.2))
names=['持续性基线','前3月均值','11年滞后','岭回归AR(36)','MLP','LSTM','GRU']
ms=[metrics[n]['test']['RMSE'] for n in names]
bars=ax.bar(range(len(names)),ms,color=['#A3ABAA','#6D8992','#ABB39D','#3C8B91','#B5A069','#AF7465','#826D9E'],width=.64)
for i,v in enumerate(ms):ax.text(i,v+.8,f'{v:.1f}',ha='center',fontsize=8)
ax.set_xticks(range(len(names)),names,rotation=25,ha='right')
ax.set(title='2021—2025年独立测试期不同模型RMSE比较',ylabel='RMSE（太阳黑子数单位）',ylim=(0,max(ms)*1.24))
ax.grid(axis='y',alpha=.15);save('fig4_test_RMSE.png')

fig,ax=plt.subplots(figsize=(7.1,3.0))
for name,col in [('MLP',COL['mlp']),('LSTM',COL['lstm']),('GRU',COL['gru'])]:
 h=np.array(best_models[name]['history'])
 ax.plot(h[:,0],h[:,2],color=col,lw=1.3,label=name+'验证损失')
ax.set(title='验证集标准化均方误差随训练轮次变化',xlabel='训练轮次',ylabel='标准化MSE')
ax.legend(frameon=False);ax.grid(alpha=.15);save('fig5_learning_curves.png')

fig,ax=plt.subplots(figsize=(7.1,3.0))
for name,col in [('岭回归AR(36)',COL['ridge']),('MLP',COL['mlp']),('LSTM',COL['lstm'])]:
 ae=np.abs(y[test]-pred[name][test]);ax.plot(tt,ae,label=name,color=col,lw=1.15)
ax.set(title='2021—2025年测试期逐月绝对误差',ylabel='|实测 - 预测|',xlabel='目标月份')
ax.legend(frameon=False);ax.grid(alpha=.14);save('fig6_monthly_absolute_errors.png')

print('FINAL TEST RESULTS:')
for name in names:print(f'{name:15s}: {metrics[name]["test"]}')
print('UNCERTAINTY intervals',ci)
print('SILSO SHA256',sha)
