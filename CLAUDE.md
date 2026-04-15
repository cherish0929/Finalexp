# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands should be run inside the `GTO` conda environment:
```bash
conda activate GTO
```

## Training

Training is fully config-driven. Use `main_v2.py` (preferred — includes warmup LR, pushforward training, EMA) or `main.py` (legacy):

```bash
python main_v2.py --config config/easypool/GTO_easypool_stronger.json
python main_v2.py --config config/keyhole/GTO_keyhole_stronger.json
```

See `easypool.sh` and `keyhole.sh` for example experiment commands.

## Evaluation & Inference

```bash
# Evaluate multiple trained models on test sets (edit CONFIG_LIST inside evaluate.py first)
python evaluate.py

# Inference with visualization (saves VTK files and plots)
python inference.py       # standard
python inference_v1.py    # variant with additional options
python inference_cut.py   # for cut-domain datasets
python inference_air.py   # for air-field specific inference
```

## Config System

All hyperparameters live in JSON files under `config/`. The `src/utils.py:load_json_config()` function loads them as a `SimpleNamespace` for dot-access. Key config sections:

- `data.fields`: list of physical fields to predict, e.g. `["T", "alpha.air"]`
- `data.horizon_train` / `data.horizon_test`: autoregressive rollout steps
- `data.cut`: if `true`, uses `CutAeroGtoDataset` (spatial sub-domain cropping)
- `model.name`: selects which model class to instantiate (see architecture section)
- `model.load_path`: directory to resume from (`{name}_best.pt` is loaded)
- `train.weight_loss`: configures weighted / focal loss per field and optional gradient loss
- `train.pushforward`: enables pushforward training (extra rollout steps during training)

## Architecture

**PhysGTO** is a graph-based neural operator for time-series PDE simulation (additive manufacturing / fluid dynamics).

The forward pass is autoregressive: given initial state `x_0`, the model rolls out `T` steps predicting `[x_1, ..., x_T]`.

### Core model components (`src/physgto.py` and variants)
- **Encoder**: embeds `(node_pos, state, time, conditions) → (V, E)`. Node features combine state + positional Fourier embeddings + time encoding + condition encoding. Edge features encode relative displacement + distance.
- **Mixer** (`N_block` stacked `MixerBlock`s): each block runs GNN message-passing → cross-attention (`Atten` module with learnable queries) → FFN with residual connections.
- **Atten**: 3-step attention — cross-attend from learnable token queries to node features, self-attend tokens, cross-attend back to nodes.
- **Decoder**: concatenates all block outputs (`[B, N_block, N, enc_dim]`), projects through MLP to predict `delta_state`.

### Model variants (selected by `model.name` in config)
| Config name | Module | Description |
|---|---|---|
| `PhysGTO` | `src/physgto.py` | Base GTO model |
| `gto_res` | `src/physgto_res.py` | With residual connections |
| `gto_attnres_multi` | `src/physgto_attnres_multi.py` | Multi-field attention residual |
| `gto_attnres_multi_v2` | `src/physgto_attnres_multi_v2.py` | v2 with `attn_res_mode` |
| `gto_res_attnres` | `src/physgto_res_attnres.py` | Combined residual + attention residual |
| `v3` | `src/gto_res_attnres_v3_self.py` | Latest self-attention variant |
| `gto_lnn` | `src/gto_lnn.py` | Lagrangian neural network variant |

### Dataset classes (`src/dataset*.py`)
- `AeroGtoDataset` / `dataset_fast.py`: 3D mesh data, standard loader with metadata caching
- `AeroGtoDataset2D` / `dataset_2d.py`: 2D version
- `CutAeroGtoDataset` / `dataset_cut_fast.py`: spatially cropped sub-domain sampling

Datasets read file lists from `.txt` files (e.g. `data/train_new_easypool-1.txt`). Normalization statistics are cached to `norm_cache.json`.

### Training loss (`src/train.py`)
- **Value loss**: MSE by default; weighted focal MSE or Huber loss for VOF fields (`alpha.*`, `gamma.*`) when `weight_loss.enable = true`
- **Gradient loss**: optional 3D finite-difference spatial gradient loss (requires `weight_loss.gradient = true` and `grid_shape`)
- **Region metrics**: active/inactive region L2/RMSE tracked separately when `data.active_mask` is configured
- AMP (mixed precision) controlled via `train.use_amp`; gradient checkpointing via `train.check_point`

### Output structure
Saved under `save_path/` (configured per experiment):
- `nn/`: model checkpoints (`{name}_best.pt`, `{name}_{epoch}.pt`)
- `logs/`: TensorBoard event files
- `record/`: text training logs (`{name}_training_log.txt`)
