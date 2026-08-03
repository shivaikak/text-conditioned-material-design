"""
SMILES Generative Transformer (GPT-style, character-level)

Improvements over the notebook GRU:
- Decoder-only Transformer with causal self-attention (handles long-range ring
  closures and branch dependencies better than GRU)
- Weight tying between input embedding and output head
- AdamW + cosine LR warmup instead of fixed LR
- Gradient clipping
- Nucleus (top-p) sampling alongside temperature
- Three-metric evaluation: validity, uniqueness, novelty

Run:
    python train_smiles_gpt.py               # train from scratch
    python train_smiles_gpt.py --sample-only --ckpt checkpoints_gpt/last.ckpt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH  = Path("/scratch/sk10731/chebi_material_enriched.jsonl")
CKPT_DIR   = Path("/scratch/sk10731/checkpoints_gpt")
MAX_HEAVY  = 80    # matches notebook filter
MAX_LEN    = 256   # covers p99 SMILES length (226) + BOS/EOS with margin

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(smiles_list: list[str]) -> dict[str, int]:
    vocab = {PAD: 0, BOS: 1, EOS: 2}
    for ch in sorted({ch for s in smiles_list for ch in s}):
        vocab[ch] = len(vocab)
    return vocab

def encode(smiles: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> Tensor:
    # truncate body so BOS + body + EOS fits within max_len
    body = [vocab[ch] for ch in smiles][: max_len - 2]
    return torch.tensor([vocab[BOS]] + body + [vocab[EOS]], dtype=torch.long)

def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> str:
    chars = []
    for tok in token_ids:
        if tok == 1:   # BOS — skip
            continue
        if tok in (0, 2):  # PAD or EOS — stop
            break
        chars.append(inv_vocab.get(tok, ""))
    return "".join(chars)

# ── Dataset ───────────────────────────────────────────────────────────────────

class SmilesDataset(Dataset):
    def __init__(self, smiles_list: list[str], vocab: dict[str, int]):
        self.smiles_list = smiles_list
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tensor:
        return encode(self.smiles_list[idx], self.vocab)

def make_collate(pad_idx: int):
    def collate(batch: list[Tensor]) -> Tensor:
        return pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return collate

# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.dropout  = dropout
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash Attention via PyTorch 2.x scaled_dot_product_attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_drop(self.proj(y))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffn  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class SmilesGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        n_embd:  int   = 256,
        n_head:  int   = 4,
        n_layer: int   = 6,
        dropout: float = 0.1,
        max_len: int   = MAX_LEN,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))   # (B, T, vocab_size)

# ── Lightning Module ───────────────────────────────────────────────────────────

class LitSmilesGPT(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        n_embd:        int   = 256,
        n_head:        int   = 4,
        n_layer:       int   = 6,
        dropout:       float = 0.1,
        max_len:       int   = MAX_LEN,
        lr:            float = 3e-4,
        weight_decay:  float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model   = SmilesGPT(vocab_size, pad_idx, n_embd, n_head, n_layer, dropout, max_len)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _step(self, batch: Tensor) -> Tensor:
        logits  = self(batch)[:, :-1, :].contiguous()   # (B, T-1, V)
        targets = batch[:, 1:].contiguous()              # (B, T-1)
        B, T, V = logits.shape
        return self.loss_fn(logits.view(B * T, V), targets.view(B * T))

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        loss = self._step(batch)
        ppl  = torch.exp(loss.detach().clamp(max=20))
        self.log("train_loss", loss, prog_bar=True, on_step=True,  on_epoch=True)
        self.log("train_ppl",  ppl,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tensor, _: int) -> Tensor:
        loss = self._step(batch)
        ppl  = torch.exp(loss.detach().clamp(max=20))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ppl",  ppl,  prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Separate weight-decay groups: no decay for biases and LayerNorm params
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "ln" not in name and "bias" not in name:
                decay.append(p)
            else:
                no_decay.append(p)

        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr,
            total_steps=total_steps, pct_start=0.05,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_smiles(
    model:      LitSmilesGPT,
    vocab:      dict[str, int],
    *,
    batch_size: int   = 100,
    temperature:float = 1.0,
    top_p:      float = 0.95,   # nucleus sampling; set to 1.0 to disable
    max_len:    int   = MAX_LEN,
) -> list[str]:
    model.eval()
    device   = next(model.parameters()).device
    inv_vocab = {v: k for k, v in vocab.items()}
    bos, eos, pad = vocab[BOS], vocab[EOS], vocab[PAD]

    seq      = torch.full((batch_size, max_len), pad, dtype=torch.long, device=device)
    seq[:, 0] = bos
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(max_len - 1):
        logits = model(seq[:, : t + 1])[:, -1, :]   # (B, V)
        logits = logits / temperature

        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens where cumulative prob exceeds top_p (keep at least 1)
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            # Scatter back to original ordering
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, 1).squeeze(1)
        next_tok  = torch.where(finished, torch.tensor(pad, device=device), next_tok)
        seq[:, t + 1] = next_tok
        finished     |= next_tok == eos
        if finished.all():
            break

    return [decode(row, inv_vocab) for row in seq.tolist()]

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(smiles_list: list[str], train_set: set[str]) -> dict:
    valid = [s for s in smiles_list if s and Chem.MolFromSmiles(s) is not None]
    unique  = set(valid)
    novel   = unique - train_set
    total   = len(smiles_list)
    return {
        "total":      total,
        "validity":   len(valid)  / total        if total  else 0.0,
        "uniqueness": len(unique) / len(valid)   if valid  else 0.0,
        "novelty":    len(novel)  / len(unique)  if unique else 0.0,
        "valid_novel": sorted(novel),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-only", action="store_true",
                   help="Skip training; load checkpoint and sample.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path for --sample-only.")
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def load_data() -> tuple[list[str], set[str]]:
    records     = [json.loads(l) for l in DATA_PATH.open()]
    records     = [r for r in records if r.get("HeavyAtomCount", 999) < MAX_HEAVY]
    smiles_list = [r["SMILES"] for r in records]
    # Deduplicate while preserving order
    seen, unique_smiles = set(), []
    for s in smiles_list:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)
    print(f"Loaded {len(unique_smiles)} unique molecules (from {len(smiles_list)} records)")
    return unique_smiles, set(unique_smiles)


def main() -> None:
    args = parse_args()
    seed_everything(42, workers=True)

    smiles_list, train_set = load_data()
    vocab      = build_vocab(smiles_list)
    inv_vocab  = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    pad_idx    = vocab[PAD]
    print(f"Vocab size: {vocab_size}")

    # ── Sample-only mode ──────────────────────────────────────────────────────
    if args.sample_only:
        ckpt = args.ckpt
        if ckpt is None:
            candidates = sorted(CKPT_DIR.glob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}")
            ckpt = str(candidates[-1])
        print(f"Loading checkpoint: {ckpt}")
        model = LitSmilesGPT.load_from_checkpoint(ckpt)
        _run_sampling(model, vocab, train_set, args)
        return

    # ── Build datasets ────────────────────────────────────────────────────────
    dataset    = SmilesDataset(smiles_list, vocab)
    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate  = make_collate(pad_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LitSmilesGPT(
        vocab_size=vocab_size, pad_idx=pad_idx,
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
        dropout=args.dropout, max_len=MAX_LEN, lr=args.lr,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Callbacks & Logger ────────────────────────────────────────────────────
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="gpt-{epoch:03d}-{val_loss:.3f}",
        save_top_k=3, monitor="val_loss", save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    wandb_logger  = WandbLogger(project="smiles-materials-gpt", log_model=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        accelerator="auto", devices="auto",
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, early_stop_cb],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ── Post-training sampling ────────────────────────────────────────────────
    best_model = LitSmilesGPT.load_from_checkpoint(ckpt_cb.best_model_path)
    _run_sampling(best_model, vocab, train_set, args)


def _run_sampling(
    model:     LitSmilesGPT,
    vocab:     dict[str, int],
    train_set: set[str],
    args:      argparse.Namespace,
) -> None:
    print(f"\n=== Sampling {args.n_sample} molecules "
          f"(T={args.temperature}, top_p={args.top_p}) ===")

    batch = 100
    all_smiles: list[str] = []
    for _ in range(args.n_sample // batch):
        all_smiles.extend(
            sample_smiles(model, vocab,
                          batch_size=batch,
                          temperature=args.temperature,
                          top_p=args.top_p)
        )
    remainder = args.n_sample % batch
    if remainder:
        all_smiles.extend(
            sample_smiles(model, vocab,
                          batch_size=remainder,
                          temperature=args.temperature,
                          top_p=args.top_p)
        )

    m = evaluate(all_smiles, train_set)
    print(f"Validity:   {m['validity']:.1%}  ({int(m['validity']*m['total'])}/{m['total']})")
    print(f"Uniqueness: {m['uniqueness']:.1%}")
    print(f"Novelty:    {m['novelty']:.1%}")

    novel = m["valid_novel"]
    print(f"\nFirst 20 valid novel SMILES (out of {len(novel)}):")
    for s in novel[:20]:
        print(f"  {s}")

    out = Path("/scratch/sk10731/generated_novel_smiles.txt")
    out.write_text("\n".join(novel))
    print(f"\nAll {len(novel)} novel valid SMILES written to {out}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
