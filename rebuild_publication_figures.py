#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication image redraw from the frozen 1990-2025 SILSO study outputs.

Palette: user-supplied paper_color_library.zip / issue_01_008. No models
are retrained and no test predictions are recomputed. Chart 5 training
histories were not archived, so ONLY their existing curve colors are
re-tinted pixelwise; axes, legend, and curve geometry are preserved.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent
SRC = ROOT
DST = ROOT/'figures_publication'
DST.mkdir(parents=True,exist_ok=True)
# Exact HEX values from library palette issue_01_008 (not guessed).
COL = dict(observed='#6F6F6F', blue='#547BB4', lstm='#C0321A', ridge='#629C35',
           mlp='#DD7C4F', gru='#6C61AF', gray='#6F6F6F')
plt.rcParams.update({
 'font.family':'Noto Sans CJK JP', 'font.size':9.2, 'axes.labelsize':9.4,
 'xtick.labelsize':8.6,'ytick.labelsize':8.6,'legend.fontsize':8.0,
 'axes.edgecolor':'#686868','axes.linewidth':.75, 'axes.spines.top':False,
 'axes.spines.right':False,'xtick.major.width':.65,'ytick.major.width':.65,
 'xtick.color':'#3d4248','ytick.color':'#3d4248',
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'svg.fonttype':'none','axes.unicode_minus':False,'lines.solid_capstyle':'round'
})

def fg(size=(7.1,3.06),left=.09,right=.985,top=.965,bottom=.18):
 fig,ax=plt.subplots(figsize=size)
 fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
 return fig,ax

def finish(ax):
 ax.grid(axis='y',color='#E6E8EA',lw=.65,zorder=0)
 ax.set_axisbelow(True)
 for spine in ['left','bottom']:
  ax.spines[spine].set_color('#B3B6B9')
  ax.spines[spine].set_linewidth(.7)
 ax.tick_params(length=3.2)

def dates(ax, interval=2):
 ax.xaxis.set_major_locator(mdates.YearLocator(interval))
 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 ax.margins(x=.012)

def save(fig,name,dpi=360):
 fig.savefig(DST/name,dpi=dpi,facecolor='white')
 plt.close(fig)

raw = pd.read_csv(SRC/'SILSO_monthly_1990_2025_snapshot.csv')
prd = pd.read_csv(SRC/'predictions_recent_all.csv')
chosen = pd.read_csv(SRC/'rolling_selected_by_validation.csv')
window = pd.read_csv(SRC/'rolling_window_ablation.csv')
boot = pd.read_csv(SRC/'bootstrap_block_sensitivity.csv')
yearly = pd.read_csv(SRC/'yearly_original_test_errors.csv')
smooth = pd.read_csv(SRC/'official_silso_smoothing.csv')
assert len(raw)==432 and raw.date.iloc[0]=='1990-01' and raw.date.iloc[-1]=='2025-12'
assert len(prd.loc[prd.split=='test'])==60
assert len(chosen)==6 and len(window)==24 and len(boot)==3
te=prd[prd.split=='test'].copy(); te['datetime']=pd.to_datetime(te.date)
dt=te.datetime
rmse=lambda k: np.sqrt(np.mean((te.observed-te[k])**2))
assert abs(rmse('LSTM')-21.907)<.01,rmse('LSTM')
assert abs(rmse('岭回归AR(36)')-22.854)<.01,rmse('岭回归AR(36)')
assert abs(chosen.loc[(chosen.model=='LSTM')&(chosen.test_period=='2021-2025'),'test_RMSE'].iloc[0]-23.115)<.01

# 1 — raw observations and time intervals. No artificial smoothing.
recent=raw[raw.date>='2010-01'].copy()
fig,ax=fg()
regions=[('2010-01-01','2015-12-31',COL['blue'],'训练期（近期）'),
         ('2016-01-01','2020-12-31',COL['ridge'],'验证期'),
         ('2021-01-01','2025-12-31',COL['lstm'],'测试期')]
for x0,x1,c,lab in regions: ax.axvspan(pd.Timestamp(x0),pd.Timestamp(x1),color=c,alpha=.075,label=lab,zorder=0)
ax.plot(pd.to_datetime(recent.date),recent.sunspot_number,color=COL['observed'],lw=1.35,label='SILSO月度实测',zorder=2)
ax.set(ylabel='太阳黑子数',xlabel='年份',ylim=(0,232))
dates(ax,2);finish(ax)
ax.legend(ncol=4,loc='upper left',frameon=False,fontsize=7.8,handlelength=2.2,columnspacing=1.3)
save(fig,'fig1_SILSO_recent_observations.png')

# 2 — official weighted 13-month smoothing, display only.
sub=smooth[smooth.date>='2010-01'].copy(); td=pd.to_datetime(sub.date)
fig,ax=fg()
ax.plot(td,sub.raw_sunspot_number,color=COL['gray'],alpha=.65,lw=.95,label='月度实测')
ax.plot(td,sub.official_13month_weighted_smooth,color=COL['blue'],lw=2.0,label='SILSO 13个月加权平滑')
ax.axvline(pd.Timestamp('2024-10-01'),color=COL['lstm'],ls=(0,(4,3)),lw=1.05,label='2024年10月平滑峰值')
ax.set(ylabel='太阳黑子数',xlabel='年份',ylim=(0,232))
dates(ax,2);finish(ax);ax.legend(frameon=False,fontsize=7.8,loc='upper left',ncol=2)
save(fig,'fig2_official_weighted_smoothing.png')

# 3 — one-step predictions (held-out test).
fig,ax=fg((7.1,3.05),bottom=.19)
for k,c,lw,ls,z in [('observed',COL['observed'],1.85,'-',5),
                    ('岭回归AR(36)',COL['ridge'],1.20,'--',2),
                    ('MLP',COL['mlp'],1.10,'-.',2),
                    ('LSTM',COL['lstm'],1.48,'-',4),
                    ('GRU',COL['gru'],1.18,':',2)]:
 label={'observed':'真实观测'}.get(k,k)
 ax.plot(dt,te[k],color=c,lw=lw,linestyle=ls,label=label,zorder=z)
ax.set(xlabel='预测月份',ylabel='月度太阳黑子数',ylim=(0,235))
dates(ax,1);finish(ax)
ax.legend(loc='upper left',ncol=3,frameon=False,fontsize=7.5,handlelength=2.2,columnspacing=1.2)
save(fig,'fig3_2021_2025_predictions.png')

# 4 — observed RMSE values calculated directly from archived 60 predictions.
names=['持续性基线','前3月均值','11年滞后','岭回归AR(36)','MLP','LSTM','GRU']
vals=np.array([rmse(k) for k in names]); fig,ax=fg((7.1,3.18),bottom=.28,top=.95)
colors=[COL['gray'],COL['blue'],COL['gray'],COL['ridge'],COL['mlp'],COL['lstm'],COL['gru']]
bar=ax.bar(np.arange(len(names)),vals,width=.62,color=colors,zorder=3,edgecolor='white',lw=.55)
bar[2].set_hatch('//');bar[0].set_alpha(.68)
for b,v in zip(bar,vals):ax.text(b.get_x()+b.get_width()/2,v+.6,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
ax.set_xticks(np.arange(len(names)),['持续性','3月均值','11年滞后','岭回归','MLP','LSTM','GRU'],rotation=0)
ax.set(ylabel='RMSE（越低越好）',ylim=(0,max(vals)*1.17))
finish(ax);save(fig,'fig4_test_RMSE.png')

# 5 — render the saved training curves unchanged except for line colors.
# Training histories are not in the released metrics, so plotting new histories
# would misrepresent the original experiment. Preserve pixel coordinates.
img=np.array(Image.open(SRC/'figures'/'fig5_learning_curves.png').convert('RGB'),dtype=np.uint8)
for orig,target in [('#D1A25B',COL['mlp']),('#BA655A',COL['lstm']),('#925D8B',COL['gru'])]:
 p=np.array([int(orig[i:i+2],16) for i in (1,3,5)],dtype=np.float32)
 q=np.array([int(target[i:i+2],16) for i in (1,3,5)],dtype=np.float32)
 z=img.astype(np.float32)
 # For antialiased pixels, allow interpolation against white / near-white.
 direction=255-p
 alpha=np.clip(np.sum((255-z)*direction,axis=2)/np.dot(direction,direction),0,1)
 predicted=255-alpha[...,None]*direction
 residual=np.linalg.norm(z-predicted,axis=2)
 mask=(alpha>.23)&(residual<12)
 if mask.sum()<50:print('Warning: few pixels for',orig,int(mask.sum()))
 out=np.rint(255-alpha[...,None]*(255-q)).clip(0,255).astype('uint8')
 img[mask]=out[mask]
Image.fromarray(img).save(DST/'fig5_learning_curves.png',dpi=(300,300))

# 6 — absolute errors.
fig,ax=fg((7.1,2.95),bottom=.18)
for name,c,ls in [('岭回归AR(36)',COL['ridge'],'--'),('MLP',COL['mlp'],'-.'),('LSTM',COL['lstm'],'-')]:
 ax.plot(dt,abs(te.observed-te[name]),label=name,color=c,lw=1.25,ls=ls)
ax.set(xlabel='预测月份',ylabel='月度绝对误差')
dates(ax,1);finish(ax)
ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=7.7)
save(fig,'fig6_monthly_absolute_errors.png')

# 7 — validation-selected window, three chronological folds.
order=['2011-2015','2016-2020','2021-2025']
fig,ax=fg((7.0,2.95),bottom=.18,top=.95)
x=np.arange(3);w=.29
for off,m,c in [(-.5,'LSTM',COL['lstm']),(.5,'Ridge',COL['ridge'])]:
 z=chosen[chosen.model==m].set_index('test_period').loc[order,'test_RMSE'].to_numpy()
 bars=ax.bar(x+off*w,z,width=w,color=c,label=m,edgecolor='white',lw=.4)
 for b,v in zip(bars,z):ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.2f}',ha='center',fontsize=8)
ax.set_xticks(x,['2011—2015','2016—2020','2021—2025'])
ax.set(ylabel='RMSE',ylim=(0,28))
finish(ax);ax.legend(frameon=False,ncol=2,loc='upper right')
save(fig,'fig7_rolling_fold_rmse.png')

# 8 — 3-panel input-window ablation.
fig,axes=plt.subplots(1,3,figsize=(9,3.05),sharey=True)
fig.subplots_adjust(left=.072,right=.993,bottom=.24,top=.86,wspace=.22)
for ax,f in zip(axes,order):
 for m,c,ls in [('LSTM',COL['lstm'],'-'),('Ridge',COL['ridge'],'--')]:
  v=window[(window.test_period==f)&(window.model==m)].sort_values('window')
  ax.plot(v.window,v.test_RMSE,marker='o',ms=3.6,lw=1.45,ls=ls,color=c,label=m)
 ax.set_title(f.replace('-','—'),fontsize=9.3,pad=7)
 ax.set_xticks([12,24,36,60])
 ax.set(xlabel='输入历史（月）')
 finish(ax)
axes[0].set_ylabel('RMSE')
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
save(fig,'fig8_window_ablation.png',dpi=330)

# 9 — paired-block bootstrap reported intervals.
fig,ax=fg((6.6,2.35),left=.20,bottom=.22,top=.93)
for idx,row in boot.sort_values('block_months').reset_index(drop=True).iterrows():
 c=COL['lstm'] if not bool(row.includes_zero) else COL['blue']
 ax.plot([row.CI_low,row.CI_high],[idx,idx],lw=3.1,color=c,solid_capstyle='round')
 ax.plot([row.CI_low,row.CI_high],[idx,idx],'|',markersize=10,color=c,markeredgewidth=1.4)
 ax.plot(row.CI_median,idx,'o',ms=4,color=COL['gray'],zorder=4)
ax.axvline(0,color=COL['gray'],lw=.9,ls='--')
ax.set_yticks(np.arange(3),[f'{int(i)}个月' for i in sorted(boot.block_months)])
ax.set(xlabel='ΔRMSE = LSTM − 岭回归',ylim=(2.5,-.5),xlim=(-2.45,.4))
finish(ax);save(fig,'fig9_bootstrap_sensitivity.png')

# 10 — per-year test RMSE without cherry-picking dates.
fig,ax=fg((6.6,2.48),left=.105,bottom=.2,top=.94)
for model,clr,ls in [('LSTM',COL['lstm'],'-'),('岭回归AR(36)',COL['ridge'],'--'),('GRU',COL['gru'],'-.')]:
 z=yearly[yearly.model==model].sort_values('year')
 ax.plot(z.year,z.RMSE,marker='o',markersize=3.8,lw=1.45,ls=ls,color=clr,label=model)
ax.set(xticks=[2021,2022,2023,2024,2025],xlabel='年份',ylabel='年度RMSE',ylim=(0,35))
ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5));finish(ax)
ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=7.7)
save(fig,'fig10_yearly_rmse.png')

for f in sorted(DST.glob('*.png')):
 im=Image.open(f)
 assert im.width>=1750 and im.height>=450, (f.name,im.size)
 print(f.name,im.size,f.stat().st_size)
metadata={'palette_id':'issue_01_008','colors':COL,'original_data_changed':False,'model_predictions_changed':False,
 'fig5_method':'pixelwise color substitution of recorded curves, no training re-execution',
 'metrics':{'test_LSTM_RMSE':float(rmse('LSTM')),'test_Ridge_RMSE':float(rmse('岭回归AR(36)'))},
 'figures':sorted(p.name for p in DST.glob('*.png'))}
(DST/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
print('Saved 10 figures successfully')
