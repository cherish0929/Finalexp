请基于当前原版模型文件进行**定向重构**，重点修改 `FieldCrossAttention` 和 `AttnResMixerBlock`，不要随意改动其余模块的整体接口与训练/推理流程；代码要保持整洁、稳定、可直接运行，并尽量兼容当前 `Model / MultiFieldMixer / Decoder` 的调用方式。

## 1. 升级 `FieldCrossAttention`

将当前较轻量的 cross-attention 改成更强的 **bi-conditioned projection cross-attention**，要求：

* `query token` 不能只依赖固定可学习参数，而要同时与 `V_self` 和 `V_other` 相关；
* 采用：`Q_base + self_summary_offset + other_summary_offset + pair_summary_offset` 的思路生成 query；
* `self_summary / other_summary / pair_summary` 由当前输入动态提取，`pair_summary` 可用 `mean(V_self * V_other)`、`mean(abs(V_self - V_other))` 或其他合理实现；
* **对 query 向量做好 LayerNorm**，保证注意力更稳定；
* `V_other` 在被读取前，先经过由 `V_self` 条件化的调制/门控，使 cross 读取是“self-aware”的；
* token 内部不要只有单层 attention，改成多层 `self-attn + FFN + residual + norm` 的 latent refinement；
* 输出阶段使用 `V_self` 作为 query、refined tokens 作为 key/value；
* 最终输出不要直接覆盖 `V_self`，而是采用 **gated residual fusion**；
* 多场情况下，不要简单把所有 `other fields` 在节点维度直接拼接后一起做 cross-attention，而是：

  * 对每个 `other field` 分别做独立的 token 压缩；
  * 再在 token 层面对来自不同物理场的 tokens 做融合；
  * 尽量保留“不同物理场身份不被混淆”的信息；
* 接口尽量保持与当前版本一致，复杂度仍然保持为线性投影式 cross-attention，而不是退化成全节点 O(N^2) cross-attention。

## 2. 扩充 `AttnResMixerBlock`

将当前 block 架构明确改成：

`GNN -> Intra -> GNN -> Cross -> GNN -> Post -> GNN -> FFN`

要求：

* 每个物理场仍然保留独立分支；
* `Intra` 和 `Post` 都表示该场内部的 attention 模块；
* `Cross` 表示场间交互，调用升级后的 `FieldCrossAttention`；
* block 内一共变成 8 个子层，因此 **跨 block 的 AttnRes** 也要同步扩展，覆盖每一个子层，而不只是部分子层；
* 所有子层尽量统一采用 **pre-norm + gated residual + optional layer scale** 的稳定结构；
* GNN 不要只是简单重复：建议前面的 GNN 偏“编码/聚合”，后面的 GNN 偏“传播/回灌”，可以轻重有别；
* edge 的更新逻辑需要保留，但写法要清晰，不要让 block 结构混乱；
* 代码中写清楚每个阶段的注释，便于后续继续改。

## 3. 在 block 内部新增“子层级注意力残差”

除了原本**跨 block 的注意力残差**之外，我还希望在单个 `AttnResMixerBlock` 内部，对**节点表示**和**边表示**也引入独立的注意力残差聚合机制，用于在 block 内部不同子层之间传递信息。

要求：

* 不要和原本的跨 block `AttnRes` 混为一谈，要**明确区分两类残差机制**：

  1. **Block-level AttnRes**：用于跨历史 block 聚合；
  2. **Intra-block AttnRes**：用于当前 block 内部各子层之间的历史聚合；
* `node` 和 `edge` 的 Intra-block AttnRes 也要**彼此独立**，不要共用同一套参数；
* 对于当前 block 内部，每个子层输出后：

  * 节点表示进入一个 `node intra-block history`
  * 边表示进入一个 `edge intra-block history`
* 后续子层的输入，不再只是上一层输出，而是可以对**当前 block 内部已有历史子层输出**做 softmax attention 聚合后再输入下一层；
* 这里的注意力残差形式可以参考 block-level AttnRes，但需要单独实现，命名清晰，例如：

  * `block_attn_res_*`
  * `node_intra_attn_res_*`
  * `edge_intra_attn_res_*`
* 节点和边的 pseudo-query / attention weight 参数要分开；
* 如果某些子层没有边更新，也要保证边历史逻辑是清楚且一致的，不要写乱。

## 4. edge 更新策略分层化

如果 block 内有多次 GNN，不要每次都同等强度地更新 edge；请一并实现更合理的分层策略，例如：

* 前面的 GNN 可以执行完整的 node + edge 更新；
* 中间某些 GNN 可以是相对轻量 node + edge 更新；
* 后面的 GNN 可以只更新 node，不更新 edge，或仅做 very light residual edge update；
* 通过显式配置或清晰注释说明每一层 GNN 的 edge 更新策略；
* 目标是让 edge 演化更稳定，避免后期过度扰动。

## 5. 实现要求

* 这次请把上述内容尽量一并实现，而不是只停留在设计；
* 但整体仍要以**稳定、清晰、可训练**为优先，不要无节制堆复杂度；
* 保持 `n_fields=1` 和 `n_fields>1` 两种情况都能正常工作；
* 保留当前 `AttnRes` 的核心思想，但使其与新的 8 子层结构、以及 block 内部 node/edge 历史聚合机制严格对应；
* 尽量通过清晰的辅助函数/小模块组织代码，避免 `forward` 过于臃肿；
* 给出修改后的完整模块代码，而不是只给 patch；
* 最后补一个最小可运行的 shape 检查/快速验证，并检查单场/多场、edge 更新分层逻辑、以及新增残差模块是否都能正常工作。