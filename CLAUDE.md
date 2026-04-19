# LPBF 多物理场重构 — 对比实验工作区

## 项目背景

本项目基于算子学习范式对激光粉末床熔融（LPBF）过程中的多物理场（温度场、应力场、熔池形貌等）进行代理模型重构。核心贡献模型为 **PhysGTO-AttnRes-Multi**，在原版 PhysGTO 基础上引入了：
- **Attention Residual**：增强特征提取的稳定性与深度；
- **跨物理场 Cross-Attention**：显式建模不同物理场之间的耦合关系。

当前目录为**对比实验专用工作区**，目标是通过与业界主流模型的系统性比较，验证本方法在 LPBF 多物理场代理建模领域的先进性与创新性。

---

## 目录结构

```
.
├── CLAUDE.md                         # 本文件（项目规范与任务说明）
├── main_contrast.py                  # 对比实验主入口（待完善）
├── evaluate_contrast.py              # 对比实验评估脚本（待完善）
├── main_v2.py                        # 参考主文件（已有，勿修改）
├── evaluate.py                       # 参考评估文件（已有，勿修改）
└── src/
    ├── model/
    │   ├── physgto.py                # 原版 PhysGTO（基线之一）
    │   └── physgto_attnres_multi_v3.py  # 本文核心模型（PhysGTO-AttnRes-Multi）
    └── contrast/
        ├── mgn_model.py              # MeshGraphNets
        ├── transolver_model.py       # Transolver
        ├── gnot_model.py             # GNOT
        ├── baseline_models.py        # FNO3D / UNet3D / GraphViT
        └── lpbf_baseline_models.py   # MeltPoolResNet / ConvLSTMModel / ResNet3DModel
```

---

## 对比模型清单

### A. 通用 PDE 代理模型（7 个）

| 模型 | 文件 | 类型 |
|---|---|---|
| PhysGTO | `src/model/physgto.py` | 图-Transformer 混合 |
| MeshGraphNets (MGN) | `src/contrast/mgn_model.py` | 图神经网络 |
| Transolver | `src/contrast/transolver_model.py` | Transformer |
| GNOT | `src/contrast/gnot_model.py` | 图神经算子 |
| FNO3D | `src/contrast/baseline_models.py` | 傅里叶神经算子 |
| UNet3D | `src/contrast/baseline_models.py` | 卷积编解码器 |
| GraphViT | `src/contrast/baseline_models.py` | 图-ViT 混合 |

### B. LPBF 专用代理模型（3 个）

| 模型 | 文件 | 来源背景 |
|---|---|---|
| MeltPoolResNet | `src/contrast/lpbf_baseline_models.py` | 熔池形貌预测 |
| ConvLSTMModel | `src/contrast/lpbf_baseline_models.py` | 时序热场预测 |
| ResNet3DModel | `src/contrast/lpbf_baseline_models.py` | 3D 残差结构 |

> **我们的方法**：`PhysGTO-AttnRes-Multi`（`src/model/physgto_attnres_multi_v3.py`）

---

## 主要任务

### Task 1 — 模型适配性检查（`src/contrast/` & `src/model/physgto.py`）

逐一检查上述 10 个对比模型，确保：

1. **前向传播和自回归流程一致**：自回归推理逻辑（时间步展开方式、隐状态传递）须与 PhysGTO 的实现对齐；
2. **特殊输入兼容**：若某模型需要额外输入（如邻接矩阵、网格坐标、边特征），在 `main_contrast.py` 中单独构造，不改动模型本身；
3. **无静默错误**：确保维度匹配，避免 shape 静默广播掩盖问题。

### Task 2 — 主文件 `main_contrast.py`

以 `main_v2.py` 为模板，扩展支持所有 10 个对比模型 + 本文模型，要求：

- 通过配置文件控制调用模型（参考`config/template.json`，可以为每个模型打造特定的配置文件）
- 为需要特殊输入的模型（MGN、GNOT 等图模型）添加专属的数据预处理分支（必要的时候可以修改数据集文件`src/dataset/dataset_fast.py`，但不能影响已有模型的正常使用）；
- 训练、验证、测试流程与 `main_v2.py` 保持一致；

### Task 3 — 评估脚本 `evaluate_contrast.py`

以 `evaluate.py` 为模板，在原有精度指标基础上，增加**推理效率统计**：

| 新增指标 | 说明 |
|---|---|
| 平均推理时间（ms/sample） | 含 warm-up，取多次均值 |
| GPU 显存峰值（MB） | `torch.cuda.max_memory_allocated()` |
| 模型参数量（M） | `sum(p.numel() for p in model.parameters())` |
| FLOPs（GFLOPs） | 使用 `fvcore` 或 `ptflops` 估算 |

---

## 注意事项

> 在开始任何修改之前，请先完整阅读 `main_v2.py` 和 `evaluate.py`，以理解现有的数据流、接口约定、训练循环结构以及编码规范，再进行扩展，**不要重写已有逻辑**；
> 修改或新建的代码需要做好注释和签名。
