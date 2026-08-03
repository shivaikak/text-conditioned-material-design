# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

Research scratch space for molecular generation ML work. The primary artifact is `text-conditioned-material-design.ipynb` — a Jupyter notebook training a character-level GRU language model on SMILES strings from the ChEBI chemical database, aimed at text-conditioned material design.

## Running the notebook

```bash
jupyter notebook text-conditioned-material-design.ipynb
# or headlessly
jupyter nbconvert --to notebook --execute text-conditioned-material-design.ipynb
```

WandB login is required before training:
```bash
wandb login
```

## Key data files

| File | Description |
|------|-------------|
| `chebi_material_enriched.jsonl` | Pre-processed dataset (~43k molecules, SMILES + RDKit descriptors + text DEFINITION) — start here, do not re-parse from SDF unless refreshing |
| `chebi.sdf` / `chebi_lite_3_stars.sdf` | Raw ChEBI database (unzipped from `.gz` files) |
| `checkpoints/` | PyTorch Lightning `.ckpt` files; best run reached `val_loss ≈ 0.38` around epoch 60–64 |

## Architecture overview

**Data pipeline** (notebook cells, in order):
1. Parse `chebi.sdf` with RDKit → keep molecules with a `DEFINITION` field → compute 9 material descriptors (MW, LogP, TPSA, etc.) → save to `chebi_material_enriched.jsonl`
2. Build char-level vocab (72 tokens: 69 SMILES chars + `<pad>=0`, `<bos>=1`, `<eos>=2`)
3. `SmilesDataset` (PyTorch `Dataset`) wraps records; `collate_fn_factory` encodes SMILES with BOS/EOS and pads batches

**Model** (`SmilesRNN` inside `LitSmilesModel`):
- `nn.Embedding(vocab_size=72, embedding_dim=256, padding_idx=0)`
- Single-layer `nn.GRU(256 → 512, batch_first=True)`
- Linear head `(512 → 72)` — next-token prediction (language model)
- Trained with `CrossEntropyLoss(ignore_index=0)` on teacher-forced targets (`x[:, 1:]`)
- `LitSmilesModel` wraps the model for PyTorch Lightning; logs `train_loss`, `train_ppl`, `val_loss`, `val_ppl` to WandB project `"smiles-materials"`

**Training setup**:
- AdamW, `lr=1e-4`; `ModelCheckpoint` (top-3 by `val_loss`) + `EarlyStopping(patience=5)`
- Checkpoints written to `./checkpoints/`; reload with `LitSmilesModel.load_from_checkpoint(path)`

**Sampling** (`sample_smiles` function):
- Autoregressive: feed prefix → take last-position logits → softmax → `torch.multinomial` → append token
- Stops at `<eos>` or `max_len=200`; results decoded with `inv_map` back to SMILES strings
- Validity checked via `Chem.MolFromSmiles` (RDKit); invalid SMILES return `None`

## Environment

Conda/Anaconda environment with Python 3.13. Key packages: `torch==2.11.0`, `lightning==2.6.1`, `rdkit==2026.3.1`, `wandb==0.25.1`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`.

## `structure_guided_pllms/` sub-repo

A git repo scaffolded with `src/` layout and Hydra config directories (`hydra_config/datamodule`, `model`, `trainer`, etc.) — all source directories are currently empty. Future work intended to build structure-guided protein language model experiments using Hydra for config management.
