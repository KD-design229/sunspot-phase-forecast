# 太阳黑子数跨活动阶段预测稳定性研究

对应《现代物理》论文：**太阳黑子数跨活动阶段预测稳定性研究**（论文正文、作者封面分开保存）。

## 数据与许可

- 真实数据：WDC-SILSO / Royal Observatory of Belgium，月均国际太阳黑子数 Version 2.0。
- 原始来源：<https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt>
- 数据 DOI：<https://doi.org/10.24414/qnza-ac80>
- 固定实验快照：`SILSO_monthly_1990_2025_snapshot.csv`，1990年1月—2025年12月，432个月，标注获取日期2026-10-24。
- **SILSO源数据采用 CC BY-NC 4.0**：必须注明来源、保留许可信息；商业性再使用需要判断是否需要单独取得授权。论文中引用的官方2024年10月平滑峰值约161，和原始月度值216.0不是同一指标。
- 文件校验值存于 `metrics_recent.json` / `supplemental_results.json`；网站更新会导致未来下载结果可能出现修订。

## 程序环境

建议 Python 3.11+。主要依赖 `numpy, pandas, scipy, matplotlib, scikit-learn, torch, python-docx`，细节见 `requirements.txt`。CPU即可运行，PyTorch采用单线程以减少运行差异。

```bash
python -m pip install -r requirements.txt
python verify_results.py
python run_recent_sunspot_study.py
python run_supplemental_experiments.py
```

程序有随机种子和早停选择，但不同PyTorch/BLAS软件版本仍可能导致极小数值差异；CSV和JSON保存的是本次实际运行结果，不应假定完全跨平台逐位相同。

## 研究实验结构

### 实验A（原始固定36个月方案）

- 1993-01—2015-12训练，2016-01—2020-12验证，2021-01—2025-12独立测试。
- 测试方式为**每月获取前月真实观测后的单步预测**，并不是提前预测60个月的未来序列。
- 对比持续性、前3个月均值、11年滞后、岭回归、MLP、LSTM和GRU；原测试RMSE为LSTM 21.907、岭回归22.854。
- 输出：`predictions_recent_all.csv`, `metrics_recent.json`, `figures/fig1...fig6`。

### 实验B（新增时间滚动与输入窗口）

- 三个互不重叠的独立测试期：2011-2015、2016-2020、2021-2025；每期前5年验证；训练样本随时间扩展。
- 为确保窗口公平比较，所有窗口共享1995-01为训练目标起点。输入长度12/24/36/60个月；模型LSTM和Ridge；各折只依据验证RMSE选择窗口和随机种子。
- 训练目标数依次为132、192、252；验证及测试各60个月；总180个独立测试月。
- `rolling_window_ablation.csv`是**所有窗口**的测试结果，`rolling_selected_by_validation.csv`是按验证期选择后的结果，勿按测试分数挑选窗口。
- `rolling_seed_runs.csv`含随机种子实验细节，`rolling_pooled_selected.csv`含180个月误差合并结果。

### 实验C（区块自助敏感性）

- 固定实验A中2021—2025年保存的60个月逐月预测；使用6/12/18月循环移动区块，三种方案各20,000次成对重采样。
- 输出 `bootstrap_block_sensitivity.csv` 和图9。区间对区块长度敏感，因此不是模型优越性的无条件证明。

### 实验D（官方平滑和年度误差）

- `official_silso_smoothing.csv`：首尾0.5权重、中间11月各1权重、除以12的SILSO 13个月中心加权平滑。该平滑使用前后各6个月的观测，仅用于事后趋势可视化，绝不进入预测输入。
- `yearly_original_test_errors.csv`：实验A在2021—2025年的逐年误差；图10。

## 文件说明

- `现代物理_太阳黑子跨阶段预测_修订投稿正文.docx`：不含作者身份信息的正文。
- `现代物理_作者信息封面_待填写.docx`：需由作者自行填写的封面。
- `run_recent_sunspot_study.py`：原固定基准全部实测计算。
- `run_supplemental_experiments.py`：新增研究实验、数据统计、真实图片生成。
- `verify_results.py`：无需训练即可验证固定数据快照及原始独立测试指标。
- `figures/`：10张图，图2已改为官方加权平滑，其他图3-6保存原基准结果。

**注意：**对近代36年月度数据进行多个五年折次评价，属于“跨活动阶段的时间滚动验证”，不能等同于完整25个太阳活动周期的独立泛化检验。正式投稿仍需补充真实作者身份信息，核实图表参考文献和SILSO再使用许可要求。
