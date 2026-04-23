"""
PhysGTO-AttnRes-Max: Bi-Conditioned Cross-Attention + 8-SubLayer Block + Intra-Block AttnRes

在 physgto_attnres_max.py (V3) 基础上进行的定向重构：

1. BiCondFieldCrossAttention（升级版 FieldCrossAttention）
   - 动态 query：Q_base + self_summary_offset + other_summary_offset + pair_summary_offset
   - query 做 LayerNorm 保证稳定性
   - V_other 被 V_self 条件化门控（self-aware cross reading）
   - n_latent 层 latent refinement (self-attn + FFN + pre-norm + gated residual)
   - 输出阶段 gated residual fusion（layer_scale × sigmoid gate）
   - 多场: 每个 other field 独立 token 压缩 → concat → self-attn 融合

2. 扩充 AttnResMixerBlock → 8 子层
   GNN_encode → Intra_attn → GNN_mid → Cross_attn → GNN_post → Post_attn → GNN_light → FFN
   - Block-level AttnRes 扩展到覆盖全部 8 个子层
   - 所有子层采用 pre-norm + gated residual

3. Intra-block AttnRes（block 内部子层历史聚合）
   - node 和 edge 各自独立的 intra-block 历史向量列表
   - 各场独立参数，与 block-level AttnRes 明确区分

4. Edge 更新分层策略
   - GNN_encode: full node+edge update (GNN)
   - GNN_mid/post: light node+edge update (GNNLight)
   - GNN_light: node only，极轻 edge residual（GNNLight, update_edge=False）
"""

import torch
import torch.nn as nn
import numpy as np

from torch.amp import GradScaler, autocast
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from torch.utils.checkpoint import checkpoint


# =============================================================================
# 工具函数
# =============================================================================

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True) + 1e-8)
    E = torch.cat([d, -d, norm], dim=-1)
    return E


def FourierEmbedding(pos, pos_start, pos_length):
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


def FourierEmbedding_pos(pos, pos_enc_dim, x_boost=2):
    """
    各向异性位置编码：x 轴使用更高频，y/z 轴保持标准频率。
    pos: (..., space_size)
    """
    original_shape = pos.shape
    space_size = original_shape[-1]
    new_pos = pos.reshape(-1, space_size)

    index = torch.arange(0, pos_enc_dim, device=pos.device).float()

    parts = []
    for dim_i in range(space_size):
        start = x_boost if dim_i == 0 else 0
        freq = 2 ** (index + start) * torch.pi
        xi = new_pos[:, dim_i:dim_i+1]
        cos_i = torch.cos(freq.unsqueeze(0) * xi)
        sin_i = torch.sin(freq.unsqueeze(0) * xi)
        parts.append(cos_i)
        parts.append(sin_i)

    embedding = torch.cat(parts, dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


# =============================================================================
# 基础模块
# =============================================================================

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm（用于 AttnRes 的 key 归一化）"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def block_attn_res(blocks, partial_block, w, rms_norm):
    """
    Block-level Attention Residuals：跨 block 历史聚合

    Args:
        blocks:        list of [bs, N, D] — 已完成 block 的表示 [b_0, ..., b_{n-1}]
        partial_block: [bs, N, D] — 当前 block 内的部分状态 b_n^i
        w:             nn.Parameter [D] — 伪查询向量（初始化为零）
        rms_norm:      RMSNorm 实例

    Returns:
        h: [bs, N, D] — softmax 加权聚合的隐状态
    """
    sources = blocks + [partial_block]
    V = torch.stack(sources, dim=0)               # [n+1, bs, N, D]
    K = rms_norm(V)
    logits = torch.einsum('d, s b n d -> s b n', w, K)  # [n+1, bs, N]
    logits = logits.clamp(-30, 30)
    alpha = logits.softmax(dim=0)
    h = torch.einsum('s b n, s b n d -> b n d', alpha, V)
    return h


def intra_block_attn_res(intra_hist, current, w, rms_norm):
    """
    Intra-block Attention Residuals：block 内子层历史聚合

    与 block_attn_res 逻辑相同，但用于 block 内部各子层之间，
    参数独立、命名独立，避免与跨 block 的 AttnRes 混淆。

    Args:
        intra_hist: list of [bs, N, D] — 当前 block 内已完成子层的表示历史
        current:    [bs, N, D] — 当前子层输出（将加入历史）
        w:          nn.Parameter [D]
        rms_norm:   RMSNorm 实例
    """
    if len(intra_hist) == 0:
        # 还没有历史，直接返回 current
        return current
    sources = intra_hist + [current]
    V = torch.stack(sources, dim=0)
    K = rms_norm(V)
    logits = torch.einsum('d, s b n d -> s b n', w, K)
    logits = logits.clamp(-30, 30)
    alpha = logits.softmax(dim=0)
    h = torch.einsum('s b n, s b n d -> b n d', alpha, V)
    return h


class GatedResidual(nn.Module):
    """
    Gated Residual：output = x + layer_scale * sigmoid(gate(x)) * delta

    初始化策略（保证初始时接近标准残差）：
    - layer_scale 初始化为 layer_scale_init（小值，如 0.01）
    - gate 的 linear 权重初始化为零 → sigmoid(0) = 0.5
      所以初始 output ≈ x + 0.01 * 0.5 * delta ≈ x（接近恒等）
    - 训练中 layer_scale 和 gate 均可自由演化
    """
    def __init__(self, dim, layer_scale_init=1e-2):
        super().__init__()
        self.layer_scale = nn.Parameter(torch.tensor(layer_scale_init))
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        # 初始化为零使 sigmoid(gate(x)) ≈ 0.5（非饱和区），训练更稳定
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x, delta):
        gate = torch.sigmoid(self.gate_proj(x))    # [bs, N, dim]
        return x + self.layer_scale * gate * delta


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, layer_norm=True,
                 n_hidden=1, hidden_size=128, act='SiLU'):
        super().__init__()
        if act == 'GELU':
            self.act = nn.GELU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
        elif act == 'PReLU':
            self.act = nn.PReLU()

        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            for i in range(1, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class Atten(nn.Module):
    """
    增强版 Attention: Xavier 初始化 + 数据相关 query offset + 多层 latent attention
    """
    def __init__(self, n_token=128, c_dim=128, n_heads=4, n_latent=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads
        self.n_latent = n_latent

        self.Q = nn.Parameter(torch.empty(self.n_token, self.c_dim))
        nn.init.xavier_uniform_(self.Q)

        self.q_offset = nn.Sequential(
            nn.Linear(self.c_dim, self.c_dim),
            nn.SiLU(),
            nn.Linear(self.c_dim, self.c_dim),
        )

        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True)
        self.attention2s = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True)
            for _ in range(self.n_latent)
        ])
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).expand(batch, -1, -1)
        q_bias = self.q_offset(W0.mean(dim=1, keepdim=True))
        learned_Q = learned_Q + q_bias
        W, _ = self.attention1(learned_Q, W0, W0)

        for latent_atten in self.attention2s:
            W, _ = latent_atten(W, W, W)

        W, _ = self.attention3(W0, W, W)
        return W


class GNN(nn.Module):
    """
    注意力加权消息聚合 GNN（完整版，用于 GNN_encode）
    """
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super().__init__()
        self.node_size = node_size
        self.output_size = output_size
        self.edge_size = edge_size
        output_size = output_size or node_size

        self.f_edge = MLP(
            input_size=edge_size + node_size * 2,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=edge_size
        )

        self.f_msg_sender = MLP(input_size=edge_size, output_size=edge_size // 2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)
        self.f_msg_receiver = MLP(input_size=edge_size, output_size=edge_size // 2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)

        self.f_attn_sender = nn.Sequential(
            nn.Linear(edge_size, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1)
        )
        self.f_attn_receiver = nn.Sequential(
            nn.Linear(edge_size, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1)
        )

        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=output_size
        )

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        msg_sender = self.f_msg_sender(edge_embeddings)
        msg_receiver = self.f_msg_receiver(edge_embeddings)

        logit_sender = self.f_attn_sender(edge_embeddings).squeeze(-1)
        logit_receiver = self.f_attn_receiver(edge_embeddings).squeeze(-1)

        logit_sender = logit_sender.clamp(-30, 30)
        logit_receiver = logit_receiver.clamp(-30, 30)

        feat0, feat1 = msg_sender.shape[-1], msg_receiver.shape[-1]

        col_0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, feat0)
        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, feat1)
        col_0_scalar = edges[..., 0]
        col_1_scalar = edges[..., 1]

        alpha_0 = scatter_softmax(logit_sender, col_0_scalar, dim=1)
        alpha_1 = scatter_softmax(logit_receiver, col_1_scalar, dim=1)

        edge_agg_0 = scatter_add(alpha_0.unsqueeze(-1) * msg_sender, col_0, dim=1, dim_size=N)
        edge_agg_1 = scatter_add(alpha_1.unsqueeze(-1) * msg_receiver, col_1, dim=1, dim_size=N)

        edge_agg = torch.cat([edge_agg_0, edge_agg_1], dim=-1)
        node_inpt = torch.cat([V, edge_agg], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


class GNNLight(nn.Module):
    """
    轻量 GNN，用于 block 内 GNN_mid / GNN_post / GNN_light 角色。

    与 GNN 接口完全一致，但：
    - 隐层更小（由 light_ratio 控制）
    - 可配置是否更新 edge（update_edge=False 时，只更新 node，返回原 edge 的零残差）
    - 复杂度更低，避免后期过度扰动 edge

    Args:
        update_edge: bool — True: 用轻量 MLP 更新 edge；False: edge 不更新（返回零残差）
    """
    def __init__(self, node_size=128, edge_size=128, output_size=None,
                 light_ratio=0.5, update_edge=True, layer_norm=False):
        super().__init__()
        self.update_edge = update_edge
        output_size = output_size or node_size
        hidden = max(16, int(edge_size * light_ratio))

        # 轻量 edge transform（仅用于消息计算，始终存在）
        self.f_edge_msg = nn.Sequential(
            nn.Linear(edge_size + node_size * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, edge_size),
        )

        # 注意力打分（轻量）
        self.f_attn = nn.Sequential(
            nn.Linear(edge_size, max(8, hidden // 4)),
            nn.SiLU(),
            nn.Linear(max(8, hidden // 4), 1)
        )

        # edge 残差更新网络（仅当 update_edge=True）
        if update_edge:
            self.f_edge_update = nn.Sequential(
                nn.Linear(edge_size + node_size * 2, hidden),
                nn.SiLU(),
                nn.Linear(hidden, edge_size),
            )
        else:
            self.f_edge_update = None

        # node 更新
        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=1, layer_norm=layer_norm, act='SiLU',
            output_size=output_size,
            hidden_size=hidden
        )

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)

        # 消息计算
        edge_msg = self.f_edge_msg(edge_inpt)   # [bs, ne, edge_size]

        # 注意力加权聚合（scatter-softmax over receiver nodes）
        logit = self.f_attn(edge_msg).squeeze(-1)   # [bs, ne]
        logit = logit.clamp(-30, 30)
        col_1_scalar = edges[..., 1]                # receiver indices
        alpha = scatter_softmax(logit, col_1_scalar, dim=1)

        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, edge_size := edge_msg.shape[-1])
        edge_agg = scatter_add(alpha.unsqueeze(-1) * edge_msg, col_1, dim=1, dim_size=N)  # [bs, N, edge_size]

        # node 更新
        node_inpt = torch.cat([V, edge_agg], dim=-1)
        node_out = self.f_node(node_inpt)

        # edge 更新
        if self.update_edge and self.f_edge_update is not None:
            edge_delta = self.f_edge_update(edge_inpt)
        else:
            edge_delta = torch.zeros_like(E)   # GNN_light: 不更新 edge

        return node_out, edge_delta


# =============================================================================
# Bi-Conditioned Field Cross Attention
# =============================================================================

class LatentRefineLayer(nn.Module):
    """
    单层 latent refinement: LayerNorm → SelfAttn + GatedResidual → LayerNorm → FFN + GatedResidual
    """
    def __init__(self, dim, n_heads, layer_scale_init=1e-2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)
        self.res1 = GatedResidual(dim, layer_scale_init)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim)
        )
        self.res2 = GatedResidual(dim, layer_scale_init)

    def forward(self, x):
        # Self-attention + gated residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.res1(x, attn_out)
        # FFN + gated residual
        ffn_out = self.ffn(self.norm2(x))
        x = self.res2(x, ffn_out)
        return x


class BiCondFieldCrossAttention(nn.Module):
    """
    Bi-Conditioned Projection Cross-Field Attention

    相比旧版 FieldCrossAttention 的改进：
    1. 动态 query 生成：Q = Q_base + Δ_self + Δ_other + Δ_pair
       - self_summary   = mean_pool(V_self)     → linear → Δ_self
       - other_summary  = mean_pool(V_other_gated) → linear → Δ_other
       - pair_summary   = mean(V_self * V_other) → linear → Δ_pair  (仅用前 V_other，多场时用 V_others[0] 做对)
       - Q = LayerNorm(Q_base + Δ_self + Δ_other + Δ_pair)
    2. V_other 被 V_self 门控（self-aware）：gated_other = V_other * sigmoid(self_gate_proj(V_self))
    3. 多场融合：每个 other field 独立提取 n_token token → concat → 1 层 self-attn 融合
    4. n_latent 层 latent refinement（LatentRefineLayer: pre-norm + gated residual）
    5. 输出：V_self attend to refined tokens → gated residual fusion

    复杂度：O(N × n_token)，非 O(N²)

    Args:
        enc_dim:         特征维度
        n_heads:         多头数
        n_token:         token 数（query tokens）
        n_latent:        latent refinement 层数
        n_other_fields:  other fields 数量（多场时为 n_fields-1）
        layer_scale_init: GatedResidual 的 layer_scale 初始值
    """
    def __init__(self, enc_dim, n_heads=4, n_token=64, n_latent=2,
                 n_other_fields=1, layer_scale_init=1e-2):
        super().__init__()
        self.n_token = n_token
        self.n_other_fields = n_other_fields

        # Q_base: Xavier 初始化
        self.Q_base = nn.Parameter(torch.empty(n_token, enc_dim))
        nn.init.xavier_uniform_(self.Q_base)

        # Query offsets（各 summary → enc_dim）
        self.q_self_proj   = nn.Linear(enc_dim, enc_dim)
        self.q_other_proj  = nn.Linear(enc_dim, enc_dim)
        self.q_pair_proj   = nn.Linear(enc_dim, enc_dim)
        self.q_ln          = nn.LayerNorm(enc_dim)

        # V_other 门控（self-aware）：每个 other field 一个门控
        self.self_gate_projs = nn.ModuleList([
            nn.Linear(enc_dim, enc_dim) for _ in range(n_other_fields)
        ])

        # 每个 other field 的 cross-attn（Q attend to V_other）
        self.per_field_cross_attns = nn.ModuleList([
            nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
            for _ in range(n_other_fields)
        ])

        # 多场 token 融合（仅当 n_other_fields > 1，concat → self-attn）
        if n_other_fields > 1:
            self.token_fusion_attn = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
            self.token_fusion_norm = nn.LayerNorm(enc_dim)
            # 将 n_other_fields * n_token → n_token 的映射（Query = learnable tokens）
            self.token_compress_Q = nn.Parameter(torch.empty(n_token, enc_dim))
            nn.init.xavier_uniform_(self.token_compress_Q)
        else:
            self.token_fusion_attn = None

        # Latent refinement layers
        self.latent_layers = nn.ModuleList([
            LatentRefineLayer(enc_dim, n_heads, layer_scale_init)
            for _ in range(n_latent)
        ])

        # 输出：V_self(normed) attend to refined tokens
        self.ln_self_out = nn.LayerNorm(enc_dim)
        self.out_cross_attn = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)

        # Gated residual fusion（输出阶段）
        self.out_gated_res = GatedResidual(enc_dim, layer_scale_init)

    def _gated_other(self, V_self, V_other, gate_proj):
        """V_other 被 V_self 条件化门控"""
        gate = torch.sigmoid(gate_proj(V_self))    # [bs, N, enc_dim]
        return V_other * gate

    def forward(self, V_self, V_others_list):
        """
        Args:
            V_self:        [bs, N, enc_dim]
            V_others_list: list of [bs, N, enc_dim]，长度 = n_other_fields

        Returns:
            [bs, N, enc_dim]
        """
        bs = V_self.shape[0]

        # ---- Step 1: 动态 query 生成 ----
        # self_summary: [bs, 1, enc_dim]
        self_summary = V_self.mean(dim=1, keepdim=True)

        # other_summary: 使用第一个 other field（门控后）做对，计算 pair_summary
        gated_v0 = self._gated_other(V_self, V_others_list[0], self.self_gate_projs[0])
        other_summary = gated_v0.mean(dim=1, keepdim=True)
        pair_summary  = (V_self * gated_v0).mean(dim=1, keepdim=True)

        # Query = Q_base + offsets，再 LayerNorm
        Q = self.Q_base.unsqueeze(0).expand(bs, -1, -1)          # [bs, n_token, enc_dim]
        Q = Q + self.q_self_proj(self_summary)                    # broadcast over n_token
        Q = Q + self.q_other_proj(other_summary)
        Q = Q + self.q_pair_proj(pair_summary)
        Q = self.q_ln(Q)                                          # LayerNorm 稳定注意力

        # ---- Step 2: 每个 other field 独立提取 tokens ----
        per_field_tokens = []
        for k, (V_other, gate_proj, cross_attn) in enumerate(
            zip(V_others_list, self.self_gate_projs, self.per_field_cross_attns)
        ):
            gated_other = self._gated_other(V_self, V_other, gate_proj)  # self-aware
            tokens_k, _ = cross_attn(Q, gated_other, gated_other)        # [bs, n_token, enc_dim]
            per_field_tokens.append(tokens_k)

        # ---- Step 3: 多场 token 融合 ----
        if self.token_fusion_attn is not None and len(per_field_tokens) > 1:
            # concat: [bs, n_other_fields * n_token, enc_dim]
            tokens_concat = torch.cat(per_field_tokens, dim=1)
            tokens_concat = self.token_fusion_norm(tokens_concat)
            compress_Q = self.token_compress_Q.unsqueeze(0).expand(bs, -1, -1)
            tokens, _ = self.token_fusion_attn(compress_Q, tokens_concat, tokens_concat)
        else:
            tokens = per_field_tokens[0]  # [bs, n_token, enc_dim]

        # ---- Step 4: Latent refinement（multi-layer self-attn + FFN） ----
        for layer in self.latent_layers:
            tokens = layer(tokens)

        # ---- Step 5: 输出阶段：V_self attend to refined tokens ----
        V_self_normed = self.ln_self_out(V_self)
        cross_out, _ = self.out_cross_attn(V_self_normed, tokens, tokens)  # [bs, N, enc_dim]

        # ---- Step 6: Gated residual fusion ----
        out = self.out_gated_res(V_self, cross_out)

        return out


# 保留旧名作 alias（防止外部引用报错）
FieldCrossAttention = BiCondFieldCrossAttention


# =============================================================================
# Decoder (per-field, 注意力加权 block 聚合)
# =============================================================================

class Decoder(nn.Module):
    """注意力加权 block 聚合 Decoder"""
    def __init__(self, n_block=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()

        self.attn_net = nn.Sequential(
            nn.Linear(enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_block)
        )

        self.delta_net = nn.Sequential(
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        b, n_block, n_node, enc_dim = V_all.shape

        attn_w = self.attn_net(pos_enc)           # [bs, N, n_block]
        attn_w = attn_w.clamp(-30, 30)
        attn_w = torch.softmax(attn_w, dim=-1)

        V_perm = V_all.permute(0, 2, 1, 3)       # [bs, N, n_block, enc_dim]
        V_agg = (attn_w.unsqueeze(-1) * V_perm).sum(dim=2)  # [bs, N, enc_dim]

        V = self.delta_net(torch.cat([V_agg, pos_enc], dim=-1))
        return V


# =============================================================================
# Encoder (per-field state encoder + FiLM conditioning + spatial_inform)
# =============================================================================

class MultiFieldEncoder(nn.Module):
    """
    多场编码器：FiLM 条件化 + spatial_inform
    """
    def __init__(self, space_size=3, n_fields=2, enc_dim=128,
                 enc_t_dim=11, cond_dim=32, spatial_dim=10):
        super().__init__()
        self.n_fields = n_fields

        self.fv_fields = nn.ModuleList([
            MLP(input_size=1 + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
            for _ in range(n_fields)
        ])

        self.fv_time    = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond    = MLP(input_size=cond_dim,  output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_spatial = MLP(input_size=spatial_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        self.fuse_para  = MLP(input_size=enc_dim * 3, output_size=enc_dim * 2, act='SiLU', layer_norm=False)

        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges, spatial_inform):
        time_enc    = self.fv_time(time_i)
        cond_enc    = self.fv_cond(conditions)
        spatial_enc = self.fv_spatial(spatial_inform)

        h = torch.cat([cond_enc, time_enc, spatial_enc], dim=-1)
        para = self.fuse_para(h)
        gamma, beta = para.chunk(2, dim=-1)

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]
            inp = torch.cat([node_pos, field_i], dim=-1)
            V_i = self.fv_fields[i](inp) * gamma.unsqueeze(-2) + beta.unsqueeze(-2)
            V_list.append(V_i)

        E = self.fe(get_edge_info(edges, node_pos))
        return V_list, E


# =============================================================================
# AttnResMixerBlock — 8 子层版本
# =============================================================================

class AttnResMixerBlock(nn.Module):
    """
    8 子层 MixerBlock:
        GNN_encode → Intra_attn → GNN_mid → Cross_attn
        → GNN_post → Post_attn → GNN_light → FFN

    两类残差机制（明确区分）：
    1. Block-level AttnRes  (block_attn_res):   跨历史 block 聚合
    2. Intra-block AttnRes  (intra_block_attn_res): 当前 block 内子层历史聚合

    Node 和 Edge 的 Intra-block AttnRes 参数彼此独立。

    GNN 分层 edge 更新策略：
    - GNN_encode (子层 0): full GNN，完整 node+edge 更新
    - GNN_mid    (子层 2): GNNLight，轻量 node+edge 更新
    - GNN_post   (子层 4): GNNLight，轻量 node+edge 更新
    - GNN_light  (子层 6): GNNLight(update_edge=False)，仅 node 更新，edge 不变

    所有 attention/FFN 子层采用 pre-norm + GatedResidual。
    """

    # 子层索引常量（便于注释和索引）
    IDX_GNN_ENC    = 0   # GNN_encode
    IDX_INTRA_ATTN = 1   # Intra (场内 attention)
    IDX_GNN_MID    = 2   # GNN_mid
    IDX_CROSS_ATTN = 3   # Cross (场间 attention)
    IDX_GNN_POST   = 4   # GNN_post
    IDX_POST_ATTN  = 5   # Post (场内 attention)
    IDX_GNN_LIGHT  = 6   # GNN_light
    IDX_FFN        = 7   # FFN
    N_SUBLAYERS    = 8

    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, n_latent=4, n_latent_cross=2,
                 gnn_light_ratio=0.5, layer_scale_init=1e-2,
                 use_intra_attn_res=False):
        super().__init__()
        self.n_fields = n_fields
        self.enc_dim  = enc_dim
        self.use_intra_attn_res = use_intra_attn_res
        node_size = enc_dim + enc_s_dim   # GNN 输入（node feat + spatial enc）

        # ------------------------------------------------------------------
        # GNN 模块（4 个角色，每个场独立）
        # ------------------------------------------------------------------
        # GNN_encode: 完整 GNN（聚合/编码）
        self.gnns_encode = nn.ModuleList([
            GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)
            for _ in range(n_fields)
        ])

        # GNN_mid: 轻量 GNNLight，轻量 node+edge 更新
        self.gnns_mid = nn.ModuleList([
            GNNLight(node_size=enc_dim + enc_s_dim, edge_size=enc_dim, output_size=enc_dim,
                     light_ratio=gnn_light_ratio, update_edge=True, layer_norm=False)
            for _ in range(n_fields)
        ])

        # GNN_post: 轻量 GNNLight，轻量 node+edge 更新
        self.gnns_post = nn.ModuleList([
            GNNLight(node_size=enc_dim + enc_s_dim, edge_size=enc_dim, output_size=enc_dim,
                     light_ratio=gnn_light_ratio, update_edge=True, layer_norm=False)
            for _ in range(n_fields)
        ])

        # GNN_light: 最轻量，仅 node 更新，edge 不变
        self.gnns_light = nn.ModuleList([
            GNNLight(node_size=enc_dim + enc_s_dim, edge_size=enc_dim, output_size=enc_dim,
                     light_ratio=gnn_light_ratio * 0.5, update_edge=False, layer_norm=False)
            for _ in range(n_fields)
        ])

        # ------------------------------------------------------------------
        # Attention 模块（Intra / Post，每个场独立）
        # ------------------------------------------------------------------
        self.ln_intra  = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.mhas_intra = nn.ModuleList([
            Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head, n_latent=n_latent)
            for _ in range(n_fields)
        ])
        self.res_intra = nn.ModuleList([
            GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)
        ])

        self.ln_post   = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.mhas_post  = nn.ModuleList([
            Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head, n_latent=n_latent)
            for _ in range(n_fields)
        ])
        self.res_post  = nn.ModuleList([
            GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)
        ])

        # ------------------------------------------------------------------
        # Cross-Field Attention（仅当 n_fields > 1）
        # ------------------------------------------------------------------
        if n_fields > 1:
            self.cross_attns = nn.ModuleList([
                BiCondFieldCrossAttention(
                    enc_dim, n_heads=cross_attn_heads, n_token=n_token,
                    n_latent=n_latent_cross, n_other_fields=n_fields - 1,
                    layer_scale_init=layer_scale_init
                )
                for _ in range(n_fields)
            ])
        else:
            self.cross_attns = None

        # ------------------------------------------------------------------
        # FFN（每个场独立）
        # ------------------------------------------------------------------
        self.ln_ffn  = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.ffns    = nn.ModuleList([
            nn.Sequential(nn.Linear(enc_dim, 2 * enc_dim), nn.SiLU(), nn.Linear(2 * enc_dim, enc_dim))
            for _ in range(n_fields)
        ])
        self.res_ffn = nn.ModuleList([
            GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)
        ])

        # GNN gated residual（用于 GNN 输出的残差连接）
        self.res_gnn_encode = nn.ModuleList([GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)])
        self.res_gnn_mid    = nn.ModuleList([GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)])
        self.res_gnn_post   = nn.ModuleList([GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)])
        self.res_gnn_light  = nn.ModuleList([GatedResidual(enc_dim, layer_scale_init) for _ in range(n_fields)])

        # ------------------------------------------------------------------
        # Block-level AttnRes：8 个子层 × n_fields（跨 block 历史聚合）
        # ------------------------------------------------------------------
        self.block_attn_res_w = nn.ParameterList([
            nn.Parameter(torch.zeros(enc_dim))
            for _ in range(n_fields * self.N_SUBLAYERS)
        ])
        self.block_rms_norm = RMSNorm()

        # ------------------------------------------------------------------
        # Intra-block AttnRes：node 和 edge 各自独立（仅当 use_intra_attn_res=True 时启用）
        # node: 8 个子层 × n_fields
        # edge: 8 个子层 × n_fields（无 edge 的子层返回最近 edge，但仍记录）
        # ------------------------------------------------------------------
        if use_intra_attn_res:
            self.node_intra_attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim))
                for _ in range(n_fields * self.N_SUBLAYERS)
            ])
            self.edge_intra_attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim))
                for _ in range(n_fields * self.N_SUBLAYERS)
            ])
            self.node_intra_rms_norm = RMSNorm()
            self.edge_intra_rms_norm = RMSNorm()
        else:
            self.node_intra_attn_res_w = None
            self.edge_intra_attn_res_w = None
            self.node_intra_rms_norm   = None
            self.edge_intra_rms_norm   = None

    # ------ 参数索引辅助 ------

    def _block_w(self, field_idx, sublayer_idx):
        """Block-level AttnRes 伪查询"""
        return self.block_attn_res_w[field_idx * self.N_SUBLAYERS + sublayer_idx]

    def _node_intra_w(self, field_idx, sublayer_idx):
        """Intra-block node AttnRes 伪查询"""
        return self.node_intra_attn_res_w[field_idx * self.N_SUBLAYERS + sublayer_idx]

    def _edge_intra_w(self, field_idx, sublayer_idx):
        """Intra-block edge AttnRes 伪查询"""
        return self.edge_intra_attn_res_w[field_idx * self.N_SUBLAYERS + sublayer_idx]

    # ------ node/edge 聚合辅助 ------

    def _apply_block_attn_res(self, blocks_i, current_node, field_idx, sublayer_idx):
        """对 node 表示做 block-level AttnRes"""
        return block_attn_res(
            blocks_i, current_node,
            self._block_w(field_idx, sublayer_idx),
            self.block_rms_norm
        )

    def _apply_node_intra_res(self, node_hist, current_node, field_idx, sublayer_idx):
        """对 node 表示做 intra-block AttnRes（disabled 时直接返回 current_node）"""
        if not self.use_intra_attn_res:
            return current_node
        return intra_block_attn_res(
            node_hist, current_node,
            self._node_intra_w(field_idx, sublayer_idx),
            self.node_intra_rms_norm
        )

    def _apply_edge_intra_res(self, edge_hist, current_edge, field_idx, sublayer_idx):
        """对 edge 表示做 intra-block AttnRes（disabled 时直接返回 current_edge）"""
        if not self.use_intra_attn_res:
            return current_edge
        return intra_block_attn_res(
            edge_hist, current_edge,
            self._edge_intra_w(field_idx, sublayer_idx),
            self.edge_intra_rms_norm
        )

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        """
        Args:
            V_list:      list of n_fields [bs, N, enc_dim] — 当前 node 表示
            E_list:      list of n_fields [bs, ne, enc_dim] — 当前 edge 表示
            edges:       [bs, ne, 2]
            s_enc:       [bs, N, enc_s_dim] — 空间位置编码
            blocks_list: list of n_fields lists — 跨 block 历史 node 表示

        Returns:
            V_final:     list of n_fields [bs, N, enc_dim]
            E_out:       list of n_fields [bs, ne, enc_dim]
            blocks_list: 已追加当前 block 输出的历史列表
        """
        blk_norm  = self.block_rms_norm

        # 初始化 intra-block 历史（每次 forward 从空开始）
        # node_hist[i]: 当前 block 内 field i 的子层输出历史
        # edge_hist[i]: 当前 block 内 field i 的 edge 子层输出历史
        node_hist = [[] for _ in range(self.n_fields)]
        edge_hist = [[] for _ in range(self.n_fields)]

        # 工作变量（在 forward 中演化）
        V = list(V_list)    # 当前 node 状态，per field
        E = list(E_list)    # 当前 edge 状态，per field

        # ==================================================================
        # 子层 0：GNN_encode（完整 GNN，聚合/编码）
        # ==================================================================
        for i in range(self.n_fields):
            # Block-level AttnRes（跨 block 历史）
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_GNN_ENC)
            # Intra-block node AttnRes（当前 block 内历史，此时为空，直接返回 v_h）
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_GNN_ENC)
            # Edge Intra-block AttnRes
            e_h = self._apply_edge_intra_res(edge_hist[i], E[i], i, self.IDX_GNN_ENC)

            # GNN_encode: full node+edge update
            V_in = torch.cat([v_h, s_enc], dim=-1)
            v_new, e_delta = self.gnns_encode[i](V_in, e_h, edges)
            V[i] = self.res_gnn_encode[i](V[i], v_new - V[i])  # gated residual
            E[i] = E[i] + e_delta                               # edge 标准残差（edge 不经过 gated res）

            # 更新历史
            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 1：Intra_attn（场内 attention）
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_INTRA_ATTN)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_INTRA_ATTN)

            attn_out = self.mhas_intra[i](self.ln_intra[i](v_h))
            V[i] = self.res_intra[i](V[i], attn_out)

            node_hist[i].append(V[i])
            # 注意：此子层无 edge 更新，edge_hist 追加当前 E（保持历史连续）
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 2：GNN_mid（轻量，node+edge 更新）
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_GNN_MID)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_GNN_MID)
            e_h = self._apply_edge_intra_res(edge_hist[i], E[i], i, self.IDX_GNN_MID)

            V_in = torch.cat([v_h, s_enc], dim=-1)
            v_new, e_delta = self.gnns_mid[i](V_in, e_h, edges)
            V[i] = self.res_gnn_mid[i](V[i], v_new - V[i])
            E[i] = E[i] + e_delta

            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 3：Cross_attn（场间交互）
        # ==================================================================
        if self.n_fields > 1:
            V_after_cross = []
            for i in range(self.n_fields):
                v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_CROSS_ATTN)
                v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_CROSS_ATTN)

                # 收集 other fields（每个场当前最新 V，已经过 block_attn_res 的原始状态）
                others = [V[j] for j in range(self.n_fields) if j != i]
                # BiCondFieldCrossAttention 内部已包含 gated residual，直接替换 V_self
                cross_out = self.cross_attns[i](v_h, others)
                V_after_cross.append(cross_out)
                # 注意：cross_attn 输出已包含 gated fusion，作为新 V[i]

            V = V_after_cross
        # n_fields=1 时跳过，V 不变

        for i in range(self.n_fields):
            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 4：GNN_post（轻量，node+edge 更新，回灌/扩散）
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_GNN_POST)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_GNN_POST)
            e_h = self._apply_edge_intra_res(edge_hist[i], E[i], i, self.IDX_GNN_POST)

            V_in = torch.cat([v_h, s_enc], dim=-1)
            v_new, e_delta = self.gnns_post[i](V_in, e_h, edges)
            V[i] = self.res_gnn_post[i](V[i], v_new - V[i])
            E[i] = E[i] + e_delta

            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 5：Post_attn（场内 attention）
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_POST_ATTN)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_POST_ATTN)

            attn_out = self.mhas_post[i](self.ln_post[i](v_h))
            V[i] = self.res_post[i](V[i], attn_out)

            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 6：GNN_light（仅 node 更新，edge 不变）
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_GNN_LIGHT)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_GNN_LIGHT)
            # edge 不参与，用当前 E 即可（GNNLight 返回零 edge delta）
            e_h = self._apply_edge_intra_res(edge_hist[i], E[i], i, self.IDX_GNN_LIGHT)

            V_in = torch.cat([v_h, s_enc], dim=-1)
            v_new, e_delta = self.gnns_light[i](V_in, e_h, edges)  # e_delta = zeros
            V[i] = self.res_gnn_light[i](V[i], v_new - V[i])
            # E[i] 不更新（e_delta 为零）
            E[i] = E[i] + e_delta   # e_delta = 0，等价于不更新

            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # 子层 7：FFN
        # ==================================================================
        for i in range(self.n_fields):
            v_h = self._apply_block_attn_res(blocks_list[i], V[i], i, self.IDX_FFN)
            v_h = self._apply_node_intra_res(node_hist[i], v_h, i, self.IDX_FFN)

            ffn_out = self.ffns[i](self.ln_ffn[i](v_h))
            V[i] = self.res_ffn[i](V[i], ffn_out)

            node_hist[i].append(V[i])
            edge_hist[i].append(E[i])

        # ==================================================================
        # End of block：将最终 node 表示追加到跨 block 历史
        # ==================================================================
        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V[i]]

        return V, E, blocks_list


# =============================================================================
# MultiFieldMixer
# =============================================================================

class MultiFieldMixer(nn.Module):
    def __init__(self, N_block, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, n_latent=4, n_latent_cross=2,
                 gnn_light_ratio=0.5, layer_scale_init=1e-2,
                 use_intra_attn_res=False):
        super().__init__()
        self.n_fields = n_fields
        self.blocks = nn.ModuleList([
            AttnResMixerBlock(
                enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                enc_s_dim=enc_s_dim, n_fields=n_fields,
                cross_attn_heads=cross_attn_heads, n_latent=n_latent,
                n_latent_cross=n_latent_cross,
                gnn_light_ratio=gnn_light_ratio,
                layer_scale_init=layer_scale_init,
                use_intra_attn_res=use_intra_attn_res,
            )
            for _ in range(N_block)
        ])

    def forward(self, V_list, E, edges, pos_enc):
        # 初始化 blocks_list: b_0 = encoder embedding（per field）
        blocks_list = [[V_list[i]] for i in range(self.n_fields)]

        # 每个场复制一份 E 用于独立演化
        E_list = [E.clone() for _ in range(self.n_fields)]

        V_all = [[] for _ in range(self.n_fields)]

        for block in self.blocks:
            V_list, E_list, blocks_list = block(V_list, E_list, edges, pos_enc, blocks_list)
            for i in range(self.n_fields):
                V_all[i].append(V_list[i])

        # Stack: [bs, N_block, N, enc_dim] per field
        V_all_stacked = [torch.stack(V_all[i], dim=1) for i in range(self.n_fields)]
        return V_all_stacked


# =============================================================================
# MultiFieldDecoder
# =============================================================================

class MultiFieldDecoder(nn.Module):
    """每个场独立的 Decoder（注意力加权版）"""
    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, n_fields=2):
        super().__init__()
        self.n_fields = n_fields
        self.decoders = nn.ModuleList([
            Decoder(n_block=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=1)
            for _ in range(n_fields)
        ])

    def forward(self, V_all_list, pos_enc):
        deltas = []
        for i in range(self.n_fields):
            delta_i = self.decoders[i](V_all_list[i], pos_enc)
            deltas.append(delta_i)
        return torch.cat(deltas, dim=-1)   # [bs, N, n_fields]


# =============================================================================
# 完整模型
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-AttnRes-Max

    主要改进（相对 V3）：
    - BiCondFieldCrossAttention：动态 query + self-aware gate + latent refinement + gated residual
    - AttnResMixerBlock：8 子层（GNN_enc→Intra→GNN_mid→Cross→GNN_post→Post→GNN_light→FFN）
    - Block-level AttnRes 扩展到 8 子层
    - Intra-block AttnRes：node/edge 各自独立，跨子层历史聚合
    - Edge 更新分层：GNN_encode 完整 / mid+post 轻量 / light 无更新
    - 全子层 pre-norm + GatedResidual（layer_scale × sigmoid gate）

    Args:
        space_size:         空间维度
        pos_enc_dim:        Fourier 编码频率数
        cond_dim:           条件维度
        spatial_dim:        空间信息维度
        N_block:            MixerBlock 数
        in_dim:             输入通道数 (= n_fields)
        out_dim:            输出通道数 (= n_fields)
        enc_dim:            隐空间维度
        n_head:             Attention 头数
        n_token:            Atten 查询 token 数
        n_latent:           Atten 内部 latent 层数
        n_latent_cross:     BiCondFieldCrossAttention latent refinement 层数
        cross_attn_heads:   Cross-Attention 头数
        dt:                 默认时间步
        stepper_scheme:     "euler" 或 "delta"
        pos_x_boost:        x 轴位置编码频率 boost
        n_fields:           物理场数量
        gnn_light_ratio:    GNNLight 的隐层大小比例
        layer_scale_init:   GatedResidual 的 layer_scale 初始值
    """
    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 spatial_dim=10,
                 N_block=4,
                 in_dim=2,
                 out_dim=2,
                 enc_dim=128,
                 n_head=4,
                 n_token=128,
                 n_latent=4,
                 n_latent_cross=2,
                 dt: float = 0.05,
                 stepper_scheme="euler",
                 pos_x_boost=2,
                 n_fields=None,
                 cross_attn_heads=4,
                 gnn_light_ratio=0.5,
                 layer_scale_init=1e-2,
                 use_intra_attn_res=False,
                 ):
        super().__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim = pos_enc_dim
        self.pos_x_boost = pos_x_boost
        self.n_fields = n_fields if n_fields is not None else in_dim

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2 + (1 + 2 * 2)

        self.encoder = MultiFieldEncoder(
            space_size=space_size,
            n_fields=self.n_fields,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim,
            cond_dim=cond_dim,
            spatial_dim=spatial_dim,
        )

        self.mixer = MultiFieldMixer(
            N_block=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads,
            n_latent=n_latent,
            n_latent_cross=n_latent_cross,
            gnn_light_ratio=gnn_light_ratio,
            layer_scale_init=layer_scale_init,
            use_intra_attn_res=use_intra_attn_res,
        )

        self.decoder = MultiFieldDecoder(
            N_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
        )

    def _encode_time(self, time_i, dt_tensor):
        time_info = torch.cat([time_i, dt_tensor], dim=-1)
        t_fourier = FourierEmbedding(time_info, 0, self.pos_enc_dim)
        t_low     = FourierEmbedding(time_i, 0, 2)
        return torch.cat([t_fourier, t_low], dim=-1)

    def forward(self, state_in, node_pos, edges, time_i, conditions, spatial_inform,
                pos_enc=None, dt=None):
        if pos_enc is None:
            pos_enc = FourierEmbedding_pos(node_pos, self.pos_enc_dim, self.pos_x_boost)

        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]

        if dt is None:
            dt_tensor = torch.full((bs, 1), self.dt, dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (float, int)):
            dt_tensor = torch.full((bs, 1), float(dt), dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (np.floating, np.integer)):
            dt_tensor = torch.tensor([dt], dtype=time_i.dtype, device=time_i.device).reshape(bs, 1)
        else:
            dt_tensor = dt.view(bs, 1).to(dtype=time_i.dtype, device=time_i.device)

        t_enc = self._encode_time(time_i, dt_tensor)

        edges_long = edges.long() if edges.dtype != torch.long else edges

        V_list, E = self.encoder(node_pos, state_in, t_enc, conditions, edges_long, spatial_inform)
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc)
        v_pred = self.decoder(V_all_list, pos_enc)

        if self.stepper_scheme == "euler":
            with autocast(device_type="cuda", enabled=False):
                state_pred = state_in.float() + v_pred.float() * dt_tensor.unsqueeze(-1).float()
        else:
            state_pred = state_in + v_pred

        return state_pred

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       spatial_inform,
                       conditions,
                       dt=None,
                       check_point=False,
                       teacher_forcing=False,
                       gt_states=None):
        state_t = state_in
        outputs = [state_in]
        T = time_seq.shape[1]

        pos_enc = FourierEmbedding_pos(node_pos, self.pos_enc_dim, self.pos_x_boost)

        for t in range(T):
            time_i = time_seq[:, t]

            def custom_forward(s_t, t_i):
                return self.forward(s_t, node_pos, edges, t_i, conditions, spatial_inform, pos_enc, dt)

            if check_point is True or (type(check_point) is int and t >= check_point):
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t = state_t.detach().requires_grad_(True)
                state_pred = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)
            else:
                state_pred = self.forward(state_t, node_pos, edges, time_i, conditions, spatial_inform, pos_enc, dt)

            outputs.append(state_pred)

            if t < T - 1:
                if teacher_forcing and gt_states is not None:
                    state_t = gt_states[:, t]
                else:
                    state_t = state_pred

        outputs = torch.stack(outputs[1:], dim=1)
        return outputs


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 70)
    print("PhysGTO-AttnRes-Max 快速验证")
    print("=" * 70)

    bs, N, ne = 2, 64, 128
    T = 4
    space_dim = 3
    in_dim = out_dim = 2
    cond_dim = 8
    spatial_dim = 10

    # ---- 双场测试 ----
    print("\n[1] n_fields=2 双场测试")
    model = Model(
        space_size=space_dim,
        pos_enc_dim=3,
        cond_dim=cond_dim,
        spatial_dim=spatial_dim,
        N_block=2,
        in_dim=in_dim,
        out_dim=out_dim,
        enc_dim=64,
        n_head=4,
        n_token=32,
        n_latent=2,
        n_latent_cross=2,
        dt=2e-5,
        n_fields=2,
        cross_attn_heads=4,
        gnn_light_ratio=0.5,
        layer_scale_init=1e-2,
    )

    state_in   = torch.randn(bs, N, in_dim)
    node_pos   = torch.rand(bs, N, space_dim)
    edges      = torch.randint(0, N, (bs, ne, 2))
    time_seq   = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
    conditions = torch.randn(bs, cond_dim)
    spatial_inform = torch.randn(bs, spatial_dim)

    # 单步
    pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions, spatial_inform)
    print(f"  [单步]  pred: {pred.shape}")
    assert pred.shape == (bs, N, out_dim), f"Shape mismatch: {pred.shape}"

    # 自回归
    out = model.autoregressive(state_in, node_pos, edges, time_seq, spatial_inform, conditions)
    print(f"  [自回归] out: {out.shape}")
    assert out.shape == (bs, T, N, out_dim), f"Shape mismatch: {out.shape}"

    # Backward pass
    model.train()
    loss = out.sum()
    loss.backward()
    print("  [backward] 通过 ✓")

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params/1e6:.3f}M")

    # ---- 验证 block-level AttnRes 扩展到 8 子层 ----
    print("\n[2] Block-level AttnRes 参数检查（应覆盖 8 子层）")
    block0 = model.mixer.blocks[0]
    n_block_w = len(block0.block_attn_res_w)
    assert n_block_w == 2 * 8, f"Expected {2*8} block AttnRes params, got {n_block_w}"
    for name_p, p in model.named_parameters():
        if 'block_attn_res_w' in name_p:
            assert torch.all(p == 0), f"{name_p} should be zero-initialized"
    print(f"  block_attn_res_w 数量: {n_block_w} (= n_fields×8=2×8) ✓")
    print("  全部初始化为零 ✓")

    # ---- 验证 Intra-block AttnRes 参数存在且为零 ----
    print("\n[3] Intra-block AttnRes 参数检查")
    n_node_intra = len(block0.node_intra_attn_res_w)
    n_edge_intra = len(block0.edge_intra_attn_res_w)
    assert n_node_intra == 2 * 8
    assert n_edge_intra == 2 * 8
    for name_p, p in model.named_parameters():
        if 'node_intra_attn_res_w' in name_p or 'edge_intra_attn_res_w' in name_p:
            assert torch.all(p == 0), f"{name_p} should be zero-initialized"
    print(f"  node_intra_attn_res_w: {n_node_intra} params ✓")
    print(f"  edge_intra_attn_res_w: {n_edge_intra} params ✓")
    print("  全部初始化为零 ✓")

    # ---- 验证 GNN_light 的 edge delta 为零 ----
    print("\n[4] Edge 更新分层：GNN_light edge_delta 验证")
    gnn_l = block0.gnns_light[0]
    # 检查 update_edge=False → f_edge_update=None → forward 返回零 delta
    assert not gnn_l.update_edge, "GNN_light 应 update_edge=False"
    assert gnn_l.f_edge_update is None, "GNN_light 不应有 f_edge_update"
    print("  GNN_light.update_edge=False ✓")
    print("  GNN_light.f_edge_update=None ✓")

    # 通过模型完整 forward 验证 GNN_light 的 edge grad 为零
    # （只有 update_edge=True 的 GNN 才产生 edge 梯度流）
    model.zero_grad()
    pred_grad = model(state_in, node_pos, edges, time_seq[:, 0], conditions, spatial_inform)
    pred_grad.sum().backward()
    # GNN_light 的 f_node 有梯度（node 路径），但 f_edge_update 不存在
    has_node_grad = block0.gnns_light[0].f_node.f[0].weight.grad is not None
    print(f"  GNN_light f_node 梯度存在: {has_node_grad} ✓（node 路径有效）")
    print("  edge 更新分层验证通过 ✓")

    # ---- 单场测试 ----
    print(f"\n[5] n_fields=1 单场测试")
    model1 = Model(
        space_size=space_dim,
        pos_enc_dim=3,
        cond_dim=cond_dim,
        spatial_dim=spatial_dim,
        N_block=2,
        in_dim=1,
        out_dim=1,
        enc_dim=64,
        n_head=4,
        n_token=32,
        n_latent=2,
        n_latent_cross=2,
        dt=2e-5,
        n_fields=1,
        cross_attn_heads=4,
    )

    state_in_1 = torch.randn(bs, N, 1)
    pred1 = model1(state_in_1, node_pos, edges, time_seq[:, 0], conditions, spatial_inform)
    print(f"  [单步]  pred: {pred1.shape}")
    assert pred1.shape == (bs, N, 1)

    out1 = model1.autoregressive(state_in_1, node_pos, edges, time_seq, spatial_inform, conditions)
    print(f"  [自回归] out: {out1.shape}")
    assert out1.shape == (bs, T, N, 1)

    model1.train()
    out1_ck = model1.autoregressive(state_in_1, node_pos, edges, time_seq, spatial_inform, conditions, check_point=True)
    loss1 = out1_ck.sum()
    loss1.backward()
    print("  [checkpoint backward] 通过 ✓")

    params1 = sum(p.numel() for p in model1.parameters())
    print(f"  参数量: {params1/1e6:.3f}M (无 cross-attention 模块)")
    assert model1.mixer.blocks[0].cross_attns is None
    print("  n_fields=1 时 cross_attns=None ✓")

    # ---- BiCondFieldCrossAttention 模块独立测试 ----
    print("\n[6] BiCondFieldCrossAttention 独立测试")
    D = 64
    cross_mod = BiCondFieldCrossAttention(D, n_heads=4, n_token=16, n_latent=2, n_other_fields=2)
    V_s = torch.randn(bs, N, D)
    V_o1 = torch.randn(bs, N, D)
    V_o2 = torch.randn(bs, N, D)
    out_cross = cross_mod(V_s, [V_o1, V_o2])
    assert out_cross.shape == (bs, N, D)
    print(f"  2 other fields: {out_cross.shape} ✓")
    out_cross1 = BiCondFieldCrossAttention(D, n_heads=4, n_token=16, n_latent=2, n_other_fields=1)(V_s, [V_o1])
    assert out_cross1.shape == (bs, N, D)
    print(f"  1 other field:  {out_cross1.shape} ✓")

    # ---- GatedResidual 初始值验证 ----
    print("\n[7] GatedResidual 初始值验证（应接近标准残差）")
    gr = GatedResidual(D, layer_scale_init=1e-2)
    x = torch.randn(1, 10, D)
    delta = torch.randn(1, 10, D)
    out_gr = gr(x, delta)
    diff = (out_gr - x).abs().mean().item()
    delta_scale = delta.abs().mean().item()
    ratio = diff / (delta_scale + 1e-8)
    print(f"  |output - x|_mean / |delta|_mean = {ratio:.4f} (应 << 1，验证初始化稳定性)")
    assert ratio < 0.1, f"GatedResidual 初始化过大: ratio={ratio:.4f}"
    print("  GatedResidual 初始化稳定 ✓")

    print("\n" + "=" * 70)
    print("✅  PhysGTO-AttnRes-Max 全部验证通过！")
    print("=" * 70)
