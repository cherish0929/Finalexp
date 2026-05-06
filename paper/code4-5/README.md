# paper/code5 — 4.5 节结果讨论配套代码

本目录提供 ch4.md 4.5 节"结果讨论"所需的全部分析代码。代码统一基于
`inference_v1.AeroGtoPredictor` 调用现有训练好的模型权重，不重训。

## 文件组织

| 文件 | 服务于 | 主要输出 |
|------|--------|----------|
| `predictor.py` | 公用工具：构建 Predictor、单样本指标、E_v 解析 | — |
| `error_spatial.py` | 4.5.1 误差空间分布热力图 | `figures/4_5_1/error_spatial_*.png/.pdf` |
| `sensitivity.py` | 4.5.2 工况敏感性分析 | `figures/4_5_2/sensitivity_*.png/.pdf` + CSV |
| `failure_case.py` | 4.5.3 失败案例剖析（top-K 最大 L2） | `figures/4_5_3/failure_top*_*.png/.pdf` + CSV/TXT |

所有绘图代码默认使用 `DejaVu Serif` 字体、400 dpi 输出，并同时导出
`.png`（嵌入论文）与 `.pdf`（打印质量），符合学术发表要求。

## 运行（在 `GTO` conda 环境中）

```bash
conda activate GTO
cd /home/ubuntu/PhysGTO/Contrastexp

# 4.5.1 — 选取传导/匙孔模式各一个代表样本，绘制误差热力图
python paper/code5/error_spatial.py \
  --config config/easypool_scale/other_models/GTO_attnres_3_ep_scale_s.json \
  --mode_label conduction \
  --config config/keyhole_scale/other_models/GTO_attnres_3_kh_scale_s_v2.json \
  --mode_label keyhole \
  --time_step 15 --slice_axis z

# 4.5.2 — 工况敏感性散点图（含 E_v 总图与逐参数子图）
python paper/code5/sensitivity.py \
  --config config/easypool_scale/other_models/GTO_attnres_3_ep_scale_s.json \
  --mode_label conduction \
  --config config/keyhole_scale/other_models/GTO_attnres_3_kh_scale_s_v2.json \
  --mode_label keyhole

# 4.5.3 — 失败案例剖析
# 第一次运行（全量扫描，生成 CSV 排名表与汇总）
python paper/code5/failure_case.py \
  --config config/easypool_scale/other_models/GTO_attnres_3_ep_scale_s.json \
  --config config/keyhole_scale/other_models/GTO_attnres_3_kh_scale_s_v2.json \
  --mode_label conduction --mode_label keyhole \
  --topk 5 \
  --out_dir paper/code5/figures/4_5_3

# 后续重绘（从已有 CSV 直接选样本，跳过全量推理；--outlier_pct 去除极端离群值）
python paper/code5/failure_case.py \
  --config config/easypool_scale/other_models/GTO_attnres_3_ep_scale_s.json \
  --config config/keyhole_scale/other_models/GTO_attnres_3_kh_scale_s_v2.json \
  --mode_label conduction --mode_label keyhole \
  --from_csv --outlier_pct 4 \
  --out_dir paper/code5/figures/4_5_3
```

## 说明

- **指标计算**：`per_sample_metrics` 在物理量级（已反归一化）上计算逐通道
  rL2 与 RMSE；活跃区/非活跃区掩码沿用训练时的 `build_active_mask`，在
  归一化空间下按阈值（默认 `T>800K` 或 `α∈[0.4,0.6]`）确定，与 4.1.3 节
  评估指标定义一致。
- **代表样本选择**（4.5.1）：默认在测试集中扫描至多 30 个样本，挑选 mean rL2
  最接近中位数者作为"代表性工况"；如需指定，可传 `--sample_idx`。
- **E_v 定义**（4.5.2/4.5.3）：`E_v = A·P / (v · r0² · h)`，层厚 `h` 默认
  30 μm，与 ch4.md 4.5.2 节定义一致；输出单位 J/mm³。
- **失败样本归因**（4.5.3）：除可视化外，`failure_summary_<label>.txt` 给出
  top-K 工况参数的均值/标准差及与全集对比，便于在论文里写入"失败样本
  集中分布于 E_v > X 的高能量密度匙孔工况"等定量结论。

如果需要切换到其他模型（例如 baseline PhysGTO），把 `--config` 换成对应
config 即可，脚本根据 `model.name` 自动适配 `spatial_inform` 输入与 EMA
权重加载逻辑。
