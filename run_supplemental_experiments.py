#!/usr/bin/env python3
"""Replicable supplement to fixed SILSO 1990-2025 monthly benchmark.
3 disjoint 5-year evaluation periods; 12/24/36/60-month window ablation;
train-only normalization; choice via preceding validation period.
One-month-ahead rolling evaluation with latest measured x[t-1], not recursive.
Original results preserved. Centered SILSO smoother is display only.
"""
from __future__ import annotations
import hashlib,json,random,time,warnings
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore',category=FutureWarning)
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent;BASE=ROOT;FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
RAW=BASE/'SILSO_monthly_1990_2025_snapshot.csv'; df=pd.read_csv(RAW)
OLD=pd.read_csv(BASE/'predictions_recent_all.csv');v=df.sunspot_number.to_numpy('float32');date=df.date.to_numpy()
assert len(df)==432 and len(OLD.loc[OLD.split=='test'])==60
S=[7,42,73];W=[12,24,36,60]
FOLDS=[{'name':'2011-2015','tr':'2005-12','va':'2010-12','te':'2015-12'},
       {'name':'2016-2020','tr':'2010-12','va':'2015-12','te':'2020-12'},
       {'name':'2021-2025','tr':'2015-12','va':'2020-12','te':'2025-12'}]
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9,'savefig.dpi':250,'axes.spines.top':False,'axes.spines.right':False})
def score(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return {'RMSE':float(np.sqrt(np.mean((a-b)**2))),'MAE':float(np.mean(np.abs(a-b))),'Bias':float(np.mean(b-a))}
def get_data(w,f):
    ids=np.arange(w,len(df));x=np.stack([v[i-w:i] for i in ids]);y=v[ids];d=date[ids]
    cohort=d>='1995-01';tr=cohort&(d<=f['tr']);va=(d>f['tr'])&(d<=f['va']);te=(d>f['va'])&(d<=f['te'])
    assert va.sum()==60 and te.sum()==60 and tr.sum()>=132
    mu=float(y[tr].mean());sd=float(y[tr].std())
    return dict(x=((x-mu)/sd).astype('float32'),yn=((y-mu)/sd).astype('float32'),y=y,d=d,
                tr=tr,va=va,te=te,mu=mu,sd=sd)
class LSTM(nn.Module):
    def __init__(self):
        super().__init__();self.rnn=nn.LSTM(1,16,batch_first=True)
        self.fc=nn.Sequential(nn.Linear(16,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):
        z,_=self.rnn(x.unsqueeze(-1));return self.fc(z[:,-1,:]).flatten()
def fit(A,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=LSTM();opt=torch.optim.Adam(net.parameters(),lr=.004,weight_decay=.0003)
    tx=torch.from_numpy(A['x'][A['tr']]);ty=torch.from_numpy(A['yn'][A['tr']]);
    vx=torch.from_numpy(A['x'][A['va']]);vy=torch.from_numpy(A['yn'][A['va']]);
    loader=DataLoader(TensorDataset(tx,ty),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');weights=None;pat=0;epochs=0
    for ep in range(210):
        net.train()
        for x,y in loader:
            opt.zero_grad();loss=((net(x)-y)**2).mean();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        net.eval()
        with torch.no_grad():vl=float(((net(vx)-vy)**2).mean())
        epochs=ep+1
        if vl<best-0.00005:
            best=vl;weights={k:z.detach().clone() for k,z in net.state_dict().items()};pat=0
        else:pat+=1
        if pat>=32:break
    net.load_state_dict(weights);net.eval()
    with torch.no_grad():p=net(torch.from_numpy(A['x'])).numpy()
    p=np.maximum(0,p*A['sd']+A['mu'])
    return {'seed':seed,'epochs':epochs,'val':score(A['y'][A['va']],p[A['va']])['RMSE'],
            'test':score(A['y'][A['te']],p[A['te']])['RMSE'],'pred':p}
def fit_ridge(A):
    trials=[]
    for alpha in [.01,.1,1.,10.,100.,1000.]:
        reg=Ridge(alpha=alpha).fit(A['x'][A['tr']],A['yn'][A['tr']]);p=np.maximum(0,reg.predict(A['x'])*A['sd']+A['mu'])
        trials.append((score(A['y'][A['va']],p[A['va']])['RMSE'],alpha,p))
    return min(trials,key=lambda z:z[0])
start=time.time();rows=[];seeds=[];fold_preds={}
for f in FOLDS:
    fold_preds[f['name']]={}
    for w in W:
        A=get_data(w,f);rv,alpha,rp=fit_ridge(A);rs=score(A['y'][A['te']],rp[A['te']])
        rows.append(dict(test_period=f['name'],window=w,model='Ridge',train_n=int(A['tr'].sum()),validation_n=60,test_n=60,chosen_seed='',epochs='',alpha=alpha,val_RMSE=rv,**{'test_'+k:z for k,z in rs.items()}))
        runs=[fit(A,s) for s in S];pick=min(runs,key=lambda z:z['val']);ls=score(A['y'][A['te']],pick['pred'][A['te']])
        rows.append(dict(test_period=f['name'],window=w,model='LSTM',train_n=int(A['tr'].sum()),validation_n=60,test_n=60,chosen_seed=pick['seed'],epochs=pick['epochs'],alpha='',val_RMSE=pick['val'],**{'test_'+k:z for k,z in ls.items()}))
        seeds.extend([{'test_period':f['name'],'window':w,'seed':z['seed'],'epochs':z['epochs'],'validation_RMSE':z['val'],'test_RMSE':z['test']} for z in runs])
        fold_preds[f['name']][w]={'observed':A['y'][A['te']].tolist(),'date':A['d'][A['te']].tolist(),'LSTM':pick['pred'][A['te']].tolist(),'Ridge':rp[A['te']].tolist()}
        print(f"fold={f['name']} w={w} train_n={A['tr'].sum()} LSTM_RMSE={ls['RMSE']:.4f} ridge_RMSE={rs['RMSE']:.4f}, selected_seed={pick['seed']}",flush=True)
allr=pd.DataFrame(rows);allr.to_csv(ROOT/'rolling_window_ablation.csv',index=False,float_format='%.7f')
pd.DataFrame(seeds).to_csv(ROOT/'rolling_seed_runs.csv',index=False,float_format='%.7f')
chosen=[]
for f in FOLDS:
    for model in ['LSTM','Ridge']:
        selected=allr[(allr.test_period==f['name'])&(allr.model==model)].sort_values('val_RMSE').iloc[0]
        chosen.append(selected.to_dict())
chosen=pd.DataFrame(chosen);chosen.to_csv(ROOT/'rolling_selected_by_validation.csv',index=False,float_format='%.7f')
pooled=[]
for model in ['LSTM','Ridge']:
    truth=[];pred=[]
    for _,r in chosen[chosen.model==model].iterrows():
        z=fold_preds[r.test_period][int(r.window)];truth.extend(z['observed']);pred.extend(z[model])
    pooled.append(dict(model=model,months=len(truth),**score(truth,pred)))
pd.DataFrame(pooled).to_csv(ROOT/'rolling_pooled_selected.csv',index=False,float_format='%.7f')
# same fixed 2021..2025 test predictions as original, 20k circular block resamples
raw=OLD.loc[OLD.split=='test'];truth=raw.observed.to_numpy();a=raw.LSTM.to_numpy();b=raw['岭回归AR(36)'].to_numpy();n=len(truth)
obs_diff=score(truth,a)['RMSE']-score(truth,b)['RMSE'];boot=[]
for blk in [6,12,18]:
    rng=np.random.default_rng(20261024+blk);d=[]
    for j in range(20000):
        ii=[]
        while len(ii)<n:
            k=int(rng.integers(n));ii.extend(((k+np.arange(blk))%n).tolist())
        ii=ii[:n]
        d.append(np.sqrt(np.mean((a[ii]-truth[ii])**2))-np.sqrt(np.mean((b[ii]-truth[ii])**2)))
    lo,mid,hi=np.quantile(d,[.025,.5,.975]);boot.append(dict(block_months=blk,replications=20000,original_difference=obs_diff,CI_low=float(lo),CI_median=float(mid),CI_high=float(hi),includes_zero=bool(lo<=0<=hi)))
    print('bootstrap',boot[-1],flush=True)
pd.DataFrame(boot).to_csv(ROOT/'bootstrap_block_sensitivity.csv',index=False,float_format='%.8f')
# display-only SILSO 13-month tapered weighted center average
smooth=np.full(len(v),np.nan)
for i in range(6,len(v)-6):smooth[i]=float((.5*v[i-6]+np.sum(v[i-5:i+6])+.5*v[i+6])/12)
smoothdf=pd.DataFrame(dict(date=date,raw_sunspot_number=v,official_13month_weighted_smooth=smooth))
smoothdf.to_csv(ROOT/'official_silso_smoothing.csv',index=False,float_format='%.7f')
print('2024-10 smoothed',smoothdf.loc[smoothdf.date=='2024-10','official_13month_weighted_smooth'].iloc[0],flush=True)
yr=[];raw=raw.copy();raw['year']=raw.date.str[:4]
for y,gr in raw.groupby('year'):
    for model in ['LSTM','GRU','岭回归AR(36)','持续性基线']:
        yr.append(dict(year=int(y),model=model,n_months=len(gr),**score(gr.observed,gr[model])))
pd.DataFrame(yr).to_csv(ROOT/'yearly_original_test_errors.csv',index=False,float_format='%.7f')
# charts
recent=smoothdf.date>='2010-01';fig,ax=plt.subplots(figsize=(7.0,3.0))
ax.plot(pd.to_datetime(smoothdf.date[recent]),smoothdf.raw_sunspot_number[recent],color='#9FABB2',lw=.9,label='月度真实观测')
ax.plot(pd.to_datetime(smoothdf.date[recent]),smoothdf.official_13month_weighted_smooth[recent],color='#BA655A',lw=2,label='SILSO官方13个月加权平滑（仅作展示）')
ax.axvline(pd.Timestamp('2024-10-01'),color='#BA655A',lw=.8,ls='--');ax.set(ylabel='太阳黑子数',xlabel='年份');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(FIG/'fig2_official_weighted_smoothing.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3));xs=np.arange(3);width=.3
for shift,mod,col in [(-.5,'LSTM','#BA655A'),(.5,'Ridge','#308080')]:
    r=chosen[chosen.model==mod].set_index('test_period').loc[[f['name'] for f in FOLDS]]
    zz=r.test_RMSE.to_numpy();ax.bar(xs+shift*width,zz,width=width,color=col,label=mod)
    for xpos,z in zip(xs+shift*width,zz):ax.text(xpos,z+.4,f'{z:.1f}',ha='center',fontsize=8)
ax.set_xticks(xs,[f['name'] for f in FOLDS]);ax.set_ylim(0,max(chosen.test_RMSE)*1.18);ax.set_ylabel('独立测试RMSE');ax.set_title('跨活动阶段：按验证集选择窗口后的外推误差');ax.legend(frameon=False)
fig.tight_layout();fig.savefig(FIG/'fig7_rolling_fold_rmse.png',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(9,3.0),sharey=True)
for ax,f in zip(axes,FOLDS):
    for m,c in [('LSTM','#BA655A'),('Ridge','#308080')]:
        a=allr[(allr.model==m)&(allr.test_period==f['name'])].sort_values('window')
        ax.plot(a.window,a.test_RMSE,'o-',color=c,lw=1.5,ms=3.5,label=m)
    ax.set_title(f['name']);ax.set_xticks(W);ax.set_xlabel('输入月数');ax.grid(alpha=.2)
axes[0].set_ylabel('RMSE');axes[0].legend(frameon=False)
fig.suptitle('不同输入窗口下的独立测试表现',fontsize=10);fig.tight_layout();fig.savefig(FIG/'fig8_window_ablation.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,2.4))
for j,r in pd.DataFrame(boot).iterrows():
    ax.plot([r.CI_low,r.CI_high],[j,j],lw=3,color='#BA655A',solid_capstyle='round');ax.plot(r.CI_median,j,'o',color='#243B4A')
ax.axvline(0,ls='--',lw=.9,color='#77898B');ax.set(yticks=np.arange(3),yticklabels=['6个月区块','12个月区块','18个月区块'],xlabel='LSTM - 岭回归 RMSE差异')
ax.invert_yaxis();ax.set_title('95%区块自助经验区间，20,000次/区块长度',fontsize=10)
fig.tight_layout();fig.savefig(FIG/'fig9_bootstrap_sensitivity.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,2.5));yrdf=pd.DataFrame(yr)
for m,c in [('LSTM','#BA655A'),('岭回归AR(36)','#308080'),('GRU','#925D8B')]:
    a=yrdf[yrdf.model==m];ax.plot(a.year,a.RMSE,'o-',label=m,color=c,lw=1.6)
ax.set(xticks=range(2021,2026),xlabel='年份',ylabel='年度RMSE');ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(FIG/'fig10_yearly_rmse.png',bbox_inches='tight');plt.close(fig)
meta={'source':'SILSO V2.0 monthly 1990-2025 frozen snapshot','snapshot_SHA256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
'primary_original_test_results_preserved':True,'train_target_start_for_window_comparison':'1995-01',
'folds':FOLDS,'windows':W,'seeds':S,'model':'PyTorch 1-layer LSTM 16 units, fc 12, Adam LR .004, wd .0003; early stop patience=32, max epochs=210; normalization training target only',
'ridge_search':[.01,.1,1.,10.,100.,1000.],
'cross_phase_coverage':'2011-2015, 2016-2020, 2021-2025; multiple phases but not large number of complete independent solar cycles',
'protocol':'one-step, updates observed previous months; models retrained once per fold and held fixed over 5-year test',
'fold_selected':chosen.to_dict(orient='records'),'pooled':pooled,'bootstrap':boot,'runtime_seconds':time.time()-start}
(ROOT/'supplemental_results.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
print('SUCCESS elapsed sec',round(time.time()-start,1),flush=True)
