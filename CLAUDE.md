# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位与维护优先级

本项目是面向 LPBF（激光粉末床熔融）多物理场预测/重构的研究型深度学习代码库。代码维护优先级为：

1. **正确性** — 模型结构、loss 计算、数据处理必须正确
2. **可复现性** — 已有实验必须能用相同 config 重现
3. **模块化** — 不同职责放在不同目录/文件
4. **可扩展性** — 新方法/新实验可以低摩擦接入
5. **论文实验可追踪** — config + checkpoint + log 完整对应

核心要求：
- 新增功能前**必须先判断职责归属**（见下方放置规则表）
- 不要为了快速实现把代码堆到当前正在编辑的文件
- 大型改动应先提出拆分方案，再执行修改

---

## Environment

All commands should be run inside the `GTO` conda environment:
```bash
conda activate GTO
```

Key dependencies: PyTorch, torch_scatter, h5py, pyvista, tensorboard.

---

## Training

Training is fully config-driven. Use `main_v2.py` (preferred — includes warmup LR, pushforward training, EMA) or `main.py` (legacy):

```bash
python main_v2.py --config config/easypool/GTO_easypool_stronger.json
python main_v2.py --config config/keyhole/GTO_keyhole_stronger.json
python main_v2.py --config config/lpbf/GTO_lpbf_easypool.json
```

Resume from checkpoint by setting `model.load_path` in the config JSON to the `save_path` directory (it loads `{save_path}/nn/{name}_best.pt`).

---

## Evaluation & Inference

```bash
# Evaluate multiple trained models on test sets (edit CONFIG_LIST inside evaluate.py first)
python evaluate.py

# Inference with visualization (saves VTK files and plots)
python inference.py       # standard
python inference_v1.py    # variant with additional options
python inference_cut.py   # for cut-domain datasets
python inference_air.py   # for air-field specific inference
python inference_max.py   # for gto_attnres_max / gto_lpbf models
```

---

## 目录职责边界

| 目录 | 职责 | 禁止放入 |
|---|---|---|
| `config/` | 实验超参数 JSON 文件 | 代码、数据 |
| `data/` | 数据列表 `.txt`、数据划分脚本 | 模型代码、训练逻辑 |
| `result_*/` | 实验输出（checkpoint、log、评估报告） | 源代码 |
| `src/model/` | 模型结构：Encoder、Decoder、GNN、Attention、各变体 | 训练循环、数据读取、绘图 |
| `src/dataset/` | Dataset、DataLoader、HDF5 读取、归一化、图构建、时间窗口采样 | 模型结构、loss 计算 |
| `src/train/` | 训练循环、验证、loss、metrics、pushforward、EMA | 模型定义、数据集定义、绘图 |
| `src/paint/` | 绘图、可视化、论文图生成 | 训练主流程不应依赖它 |
| `src/utils/` | 通用工具：种子、调度器、EMA、checkpoint probe、logging | 模型结构、训练逻辑 |
| `src/utils.py` | **Legacy 工具文件，不应继续扩展** | 新功能 |
| `utils/` (根目录) | 实验计划、任务说明、shell 脚本等非核心文件 | 核心代码 |
| `main.py` / `main_v2.py` | 运行入口：参数解析、配置读取、环境初始化、调用训练流程 | 模型结构、数据处理、loss、绘图逻辑 |

---

## 新增代码放置规则

| 新增功能 | 应放置位置 |
|---|---|
| 新 Dataset / DataLoader | `src/dataset/` 新建文件 |
| HDF5 / VTK / 数据读取逻辑 | `src/dataset/` |
| 图结构构建、边特征、节点特征 | `src/dataset/` |
| 新模型结构 | `src/model/` 新建文件，注册到 `MODEL_REGISTRY` |
| GNN / Attention / MLP / Decoder / Encoder | `src/model/` |
| LPBF 物理先验模块 | `src/model/`，必要时单独建文件 |
| 训练循环 | `src/train/` |
| 新 loss 函数 | `src/train/losses.py` |
| 新 metric | `src/train/metrics.py` |
| checkpoint / EMA / scheduler | `src/utils/` |
| 绘图、误差图、3D 可视化、Pareto 图 | `src/paint/` |
| 配置读取、随机种子、路径处理 | `src/utils/` |
| 实验计划、任务脚本 | 根目录 `utils/` |

**不要把所有函数都塞进 `src/utils.py`，也不要为了方便直接追加到当前文件末尾。**

---

## Config System（配置优先原则）

All hyperparameters live in JSON files under `config/`. Subdirectories: `easypool/`, `keyhole/`, `lpbf/`, `new_liquid/`. The `src/utils.py:load_json_config()` function loads them as nested dicts (config values are accessed via `args.data.get("key")` dict-style, not dot-access).

规则：
- 所有实验超参数优先写入 `config/*.json`
- 不要在代码中硬编码数据路径、模型超参数、loss 权重、batch size、learning rate、horizon 等
- 新实验优先新增或复制修改 config 文件，而不是复制训练代码
- 新增功能需要新参数时，必须同步更新 config template 或在此文件中说明
- 保持 nested dict 风格访问（`.get()`），不要随意改成 dot-access

Key config sections:

- `data.fields`: list of physical fields to predict, e.g. `["T", "alpha.air"]`
- `data.horizon_train` / `data.horizon_test`: autoregressive rollout steps
- `data.cut`: if `true`, uses `CutAeroGtoDataset` (spatial sub-domain cropping)
- `model.name`: selects which model class to instantiate via `MODEL_REGISTRY` in `src/model/__init__.py`
- `model.load_path`: directory to resume from (`{name}_best.pt` is loaded)
- `train.weight_loss`: configures weighted / focal loss per field, optional gradient / chamfer / peak / normal losses
- `train.pushforward`: enables pushforward training (extra rollout steps during training)
- `train.scheduler`: warmup + cosine annealing schedule (`warmup_epochs`, `T_0`, `T_mult`)
- `train.check_point`: enables gradient checkpointing (or set to `true` to let `ckpt_probe` auto-detect threshold)

---

## Architecture

**PhysGTO** is a graph-based neural operator for time-series PDE simulation (additive manufacturing / LPBF / fluid dynamics).

The forward pass is autoregressive: given initial state `x_0`, the model rolls out `T` steps predicting `[x_1, ..., x_T]`. Stepper scheme (`model.stepper_scheme`) is either `"euler"` (x_{t+1} = x_t + dt * f(x_t)) or `"delta"` (x_{t+1} = x_t + f(x_t)).

### Core model components (shared across variants)
- **Encoder**: embeds `(node_pos, state, time, conditions) → (V, E)`. Node features combine state + positional Fourier embeddings + time encoding + condition encoding. Edge features encode relative displacement + distance.
- **Mixer** (`N_block` stacked blocks): each block runs GNN message-passing → cross-attention (`Atten` module with learnable token queries) → FFN with residual connections.
- **Atten**: 3-step attention — cross-attend from learnable token queries to node features, self-attend tokens, cross-attend back to nodes.
- **Decoder**: aggregates all block outputs, projects through MLP to predict `delta_state`.

### Model variants (selected by `model.name` in config)
| Config name | Module | Description |
|---|---|---|
| `PhysGTO` | `src/model/physgto.py` | Base GTO model |
| `PhysGTO_v2` | `src/model/physgto_v2.py` | Anisotropic pos encoding, FiLM conditioning, spatial_inform |
| `gto_res` | `src/model/physgto_res.py` | With residual connections |
| `gto_attnres_multi` | `src/model/physgto_attnres_multi.py` | Multi-field attention residual |
| `gto_attnres_multi_v2` | `src/model/physgto_attnres_multi_v2.py` | v2 with `attn_res_mode` |
| `gto_res_attnres` | `src/model/physgto_res_attnres.py` | Combined residual + attention residual |
| `gto_attnres_multi_v3` | `src/model/physgto_attnres_multi_v3.py` | Block AttnRes + multi-field cross-attention |
| `gto_attnres_max` | `src/model/physgto_attnres_max.py` | Max variant with latent cross-attention, spatial gating |
| `gto_lpbf` | `src/model/physgto_lpbf.py` | Physics-informed LPBF model (laser field, source-term decoder) |
| `gto_lnn` | `src/model/gto_lnn.py` | Lagrangian neural network variant |

Models are instantiated via `build_model()` in `src/model/__init__.py`, which reads `model.name` from config and passes the appropriate kwargs per variant.

### 模型扩展规范

- 每个主要模型变体应有独立文件
- 公共模块（MLP、GNN、Atten、FourierEmbedding 等）目前集中在 `physgto_attnres_multi_v3.py` 中被其他文件 import；后续如果继续增长应考虑抽象到独立的 `modules.py`
- 不要把新模型追加到已有大型模型文件（当前最大的 `physgto_attnres_max.py` 已 1388 行）
- 模型文件只负责网络结构和 forward，不写训练循环、数据读取、checkpoint 保存、绘图
- 新模型通过 `MODEL_REGISTRY` / `build_model()` 机制接入
- 如果新模型需要额外 batch 字段，必须在此文件中记录

### `gto_lpbf` — the physics-informed variant
The LPBF model (`src/model/physgto_lpbf.py`) adds physics-structured components on top of the v3 mixer:
- **LaserFieldModule**: computes per-node Gaussian laser intensity from physical parameters
- **ScaleAwareEncoder**: dual positional encoding (normalized + absolute) with spatially-varying FiLM
- **SourceTermDecoder**: separate physics branches (diffusion, laser absorption, radiation, latent heat, recoil pressure, evaporation, interface evolution) combined via per-node spatial gating
- Requires extra batch fields: `node_pos_abs`, `laser_params`, `laser_traj`, `abs_time_seq`
- Uses `LPBFLaserDataset` (selected automatically when `model.name == "gto_lpbf"`)

---

## Dataset classes (`src/dataset/`)

| Class | Module | Use case |
|---|---|---|
| `AeroGtoDataset` | `dataset_fast.py` | 3D mesh data, standard loader with metadata caching |
| `AeroGtoDataset2D` | `dataset_2d.py` | 2D version |
| `CutAeroGtoDataset` | `dataset_cut_fast.py` | Spatially cropped sub-domain sampling |
| `LPBFSlotDataset` | `dataset_lpbf.py` | Slot-based LPBF (field-agnostic fixed-K design) |
| `LPBFLaserDataset` | `dataset_lpbf.py` | Extends LPBFSlotDataset with laser trajectory and physical params |

Datasets read HDF5 file paths from `.txt` lists under `data/` (e.g. `data/train_new_easypool-1.txt`). Use `data/data_split.py` to generate new train/test splits. Normalization statistics are cached to `norm_cache.json` (path set via `data.norm_cache` in config).

### Batch 字段约定

标准 batch（`AeroGtoDataset`）返回：
```python
{
    "dt":             float tensor,
    "state":          [T+1, N, C],      # 含初始帧
    "time_seq":       [T, 1],           # 相对时间
    "node_pos":       [N, 3],           # 归一化坐标
    "edges":          [E, 2],
    "node_type":      [N],
    "spatial_inform": [dim],            # 网格范围等空间信息
    "conditions":     [cond_dim],
    "grid_shape":     [3],
    "active_mask":    [T, N, C],        # optional
}
```

`LPBFLaserDataset` 额外增加：
```python
{
    "abs_time_seq":   [T+1],            # 绝对时间
    "laser_traj":     [T+1, laser_dim], # 激光轨迹参数
    "laser_params":   [param_dim],      # 激光物理参数
    "node_pos_abs":   [N, 3],           # 绝对坐标（物理单位）
}
```

### 数据集扩展规范

- 数据读取、归一化、数据划分、时间窗口采样、空间裁剪、图构建集中在 `src/dataset/`
- 不要在训练循环或入口文件里临时写数据处理逻辑
- 新增 Dataset 时应说明它读取哪些 HDF5 字段、返回哪些 batch key
- batch 字段命名应与上方约定保持一致

---

## Training pipeline (`src/train/`)

The training package is split into focused modules:
- `trainer.py` → `train_v2()`: main training loop with configurable grad_loss_weight, NaN guards, EMA
- `pushforward.py` → `train_pushforward()`: extends rollout beyond `horizon_train` by extra steps
- `losses.py`: value loss (MSE / weighted focal MSE / Huber for VOF fields), gradient loss, chamfer loss, peak loss, normal consistency loss, sharpness loss
- `metrics.py`: per-field L2, RMSE, active/inactive region metrics
- `autoregressive.py`: LPBF-specific autoregressive helper (extracts laser params from batch)
- `legacy.py`: original `train()` / `validate()` from pre-refactor

Loss details:
- **Value loss**: MSE by default; weighted focal MSE when `weight_loss.enable = true`; Huber loss for VOF fields (`alpha.*`, `gamma.*`)
- **Gradient loss**: 3D finite-difference spatial gradient loss (`weight_loss.gradient = true`)
- **Geometry losses** (LPBF): chamfer, peak (top-K error), normal consistency — require `node_pos` in loss
- **Region metrics**: active/inactive region L2/RMSE tracked separately when `data.active_mask` is configured
- AMP (mixed precision) via `train.use_amp`; gradient checkpointing via `train.check_point`

### 训练与评估规范

- 训练逻辑集中在 `src/train/`
- loss 与 metric 分离（`losses.py` vs `metrics.py`）
- pushforward、EMA、AMP、gradient checkpointing 等功能应保持模块化
- 新增 loss 优先放入 `src/train/losses.py`
- 新增 metric 优先放入 `src/train/metrics.py`
- `evaluate.py`、`inference.py` 等脚本应尽量只做流程编排，核心逻辑放入可复用模块

---

## Utilities (`src/utils/`)

- `ema.py`: Exponential Moving Average wrapper for model weights (used for validation)
- `scheduler.py`: `WarmupCosineScheduler` (linear warmup → CosineAnnealingWarmRestarts)
- `ckpt_probe.py`: adaptive checkpoint threshold probing — binary-searches for max non-checkpointed steps before OOM, caches results to `{save_path}/record/`
- `data.py`: DataLoader worker init
- `logging.py`: error logging to `{save_path}/record/bug.txt`
- Legacy utilities in `src/utils.py`: `set_seed`, `init_weights`, `ChannelNormalizer`, `build_active_mask`, `collate_variable_nodes`, `load_json_config`, `parse_args`

**注意**: `src/utils.py` 是历史遗留文件（296 行），不应继续扩展。新增通用工具应放入 `src/utils/` 包中合适的模块。

---

## Plotting (`src/paint/`)

- 绘图代码放在 `src/paint/`
- 训练主流程不应依赖绘图代码
- 论文图、切片图、误差图、3D VTK 可视化应作为独立脚本
- 绘图脚本从 `result_*/` 或配置指定路径读取数据
- 不要在训练循环中写复杂绘图逻辑
- 输出路径应通过参数或 config 控制

---

## Output structure

Saved under `save_path/` (configured per experiment):
- `nn/`: model checkpoints (`{name}_best.pt`, `{name}_{epoch}.pt`)
- `logs/`: TensorBoard event files (`tensorboard --logdir result_*/logs`)
- `record/`: text training logs (`{name}_training_log.txt`), error logs (`bug.txt`), checkpoint probe cache

---

## `main.py` / `main_v2.py` 入口约束

- `main_v2.py` 是当前推荐训练入口（741 行，含 dataloader 构建和主循环编排）
- `main.py` 是 legacy 入口（408 行）
- 入口文件只负责：参数解析 → 配置读取 → 环境初始化 → 调用 `src/` 中的训练/测试流程
- **不要**在入口文件中新增大段模型结构、数据处理、loss、metric、可视化逻辑
- 如果需要新增逻辑，应放到 `src/dataset/`、`src/model/`、`src/train/`、`src/paint/` 或 `src/utils/` 的合适位置

---

## 代码质量规范

- 文件名、函数名、变量名使用 `snake_case`；类名使用 `PascalCase`；常量使用 `UPPER_SNAKE_CASE`
- 新增核心函数尽量添加类型标注
- 涉及张量的函数应在 docstring 中说明 shape
- 单个函数尽量不超过 80 行
- 单个文件过长时应主动建议拆分（参考阈值：400 行）
- 避免复制粘贴大段重复代码
- 删除无用代码前应确认不会影响旧实验复现

---

## 修改前后 Checklist

### 修改前

- [ ] 当前要新增的逻辑属于哪个模块？（参考放置规则表）
- [ ] 是否已经有类似函数可以复用？
- [ ] 是否应该新增配置项？
- [ ] 是否会影响已有 config 的解析？
- [ ] 是否会破坏已有 checkpoint 加载？
- [ ] 是否会改变模型输入输出接口？
- [ ] 是否会覆盖历史实验结果？

### 修改后

- [ ] 是否能正常 import？（`python -c "from src.model import MODEL_REGISTRY"`）
- [ ] 是否能运行最小 forward？
- [ ] 输入输出 shape 是否一致？
- [ ] 是否出现重复函数？
- [ ] 是否把临时代码放进核心模块？
- [ ] 是否保持 config-driven？
- [ ] 是否需要更新本文件？

---

## 禁止事项

- **不要**把大量逻辑堆到 `main.py` 或 `main_v2.py`
- **不要**继续无限扩展 `src/utils.py`
- **不要**把模型、数据集、训练、绘图混在一个文件
- **不要**在模型文件中读取数据或保存结果
- **不要**在训练循环中写论文图生成逻辑
- **不要**硬编码绝对路径
- **不要**修改 `data/` 和历史 `result_*/`，除非明确要求
- **不要**随意更改已有 public API（`MODEL_REGISTRY`、Dataset `__getitem__` 返回格式、`build_model()` 签名）
- **不要**引入大型新依赖，除非说明必要性
- **不要**为了临时实验破坏整体结构
