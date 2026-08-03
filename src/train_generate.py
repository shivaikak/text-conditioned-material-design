from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from rdkit import Chem, RDLogger
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from common import load_yaml, utc_now, write_json

RDLogger.DisableLog("rdApp.*")
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def build_vocab(smiles_list: list[str]) -> dict[str, int]:
    vocab = {PAD: 0, BOS: 1, EOS: 2}
    for char in sorted({char for smiles in smiles_list for char in smiles}):
        vocab[char] = len(vocab)
    return vocab


def encode(smiles: str, vocab: dict[str, int], max_len: int) -> Tensor:
    body = [vocab[char] for char in smiles][: max_len - 2]
    return torch.tensor([vocab[BOS], *body, vocab[EOS]], dtype=torch.long)


def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> str:
    chars: list[str] = []
    for token in token_ids:
        if token == 1:
            continue
        if token in (0, 2):
            break
        chars.append(inv_vocab.get(token, ""))
    return "".join(chars)


class SmilesDataset(Dataset[Tensor]):
    def __init__(self, smiles: list[str], vocab: dict[str, int], max_len: int):
        self.smiles = smiles
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Tensor:
        return encode(self.smiles[index], self.vocab, self.max_len)


def make_collate(pad_idx: int):
    def collate(batch: list[Tensor]) -> Tensor:
        return pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    return collate


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_drop(self.proj(output))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SmilesGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        _, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class LitSmilesGPT(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        max_len: int,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SmilesGPT(
            vocab_size, pad_idx, n_embd, n_head, n_layer, dropout, max_len
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _step(self, batch: Tensor) -> Tensor:
        logits = self(batch)[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        batch_size, seq_len, vocab_size = logits.shape
        return self.loss_fn(
            logits.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len)
        )

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ppl", torch.exp(loss.detach().clamp(max=20)), on_epoch=True)
        return loss

    def validation_step(self, batch: Tensor, _: int) -> Tensor:
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ppl", torch.exp(loss.detach().clamp(max=20)), on_epoch=True)
        return loss

    def configure_optimizers(self):
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and "ln" not in name and "bias" not in name:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@torch.no_grad()
def sample_smiles(
    model: LitSmilesGPT,
    vocab: dict[str, int],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_len: int,
) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    inv_vocab = {value: key for key, value in vocab.items()}
    bos, eos, pad = vocab[BOS], vocab[EOS], vocab[PAD]
    sequence = torch.full((batch_size, max_len), pad, dtype=torch.long, device=device)
    sequence[:, 0] = bos
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for position in range(max_len - 1):
        logits = model(sequence[:, : position + 1])[:, -1, :] / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > top_p
            sorted_logits[remove] = float("-inf")
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            logits = filtered
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1).squeeze(1)
        next_token = torch.where(finished, torch.tensor(pad, device=device), next_token)
        sequence[:, position + 1] = next_token
        finished |= next_token == eos
        if finished.all():
            break
    return [decode(row, inv_vocab) for row in sequence.tolist()]


def load_data(path: Path, max_heavy_atoms: int) -> tuple[list[str], set[str]]:
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    filtered = [record for record in records if record.get("HeavyAtomCount", 999) < max_heavy_atoms]
    seen: set[str] = set()
    unique_smiles: list[str] = []
    for record in filtered:
        smiles = record["SMILES"]
        if smiles not in seen:
            seen.add(smiles)
            unique_smiles.append(smiles)
    return unique_smiles, set(unique_smiles)


def simple_generation_metrics(smiles: list[str], train_set: set[str]) -> dict[str, Any]:
    valid = [item for item in smiles if item and Chem.MolFromSmiles(item) is not None]
    unique = set(valid)
    novel = unique - train_set
    return {
        "total": len(smiles),
        "valid_count": len(valid),
        "validity": len(valid) / len(smiles) if smiles else 0.0,
        "unique_count": len(unique),
        "uniqueness": len(unique) / len(valid) if valid else 0.0,
        "novel_count": len(novel),
        "novelty": len(novel) / len(unique) if unique else 0.0,
        "valid_novel": sorted(novel),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    experiment = config["experiment"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    sampling_cfg = config["sampling"]
    filters_cfg = config["filters"]
    wandb_cfg = config.get("wandb", {})
    seed = int(experiment["seed"])
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data_path = Path(config["paths"]["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    smiles_list, train_set = load_data(data_path, int(filters_cfg["max_heavy_atoms"]))
    vocab = build_vocab(smiles_list)
    pad_idx = vocab[PAD]
    max_len = int(model_cfg["max_len"])

    dataset = SmilesDataset(smiles_list, vocab, max_len)
    val_size = max(1, int(len(dataset) * float(training_cfg["val_fraction"])))
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
    collate = make_collate(pad_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(training_cfg["num_workers"]) > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(training_cfg["num_workers"]) > 0,
    )

    model = LitSmilesGPT(
        vocab_size=len(vocab),
        pad_idx=pad_idx,
        n_embd=int(model_cfg["n_embd"]),
        n_head=int(model_cfg["n_head"]),
        n_layer=int(model_cfg["n_layer"]),
        dropout=float(model_cfg["dropout"]),
        max_len=max_len,
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="model-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=int(training_cfg["early_stopping_patience"]),
        mode="min",
    )

    logger: WandbLogger | bool
    if bool(wandb_cfg.get("enabled", True)):
        logger = WandbLogger(
            project=str(wandb_cfg.get("project", "smiles-materials-autolab")),
            name=str(experiment["id"]),
            save_dir=str(output_dir / "wandb"),
            log_model=False,
            offline=str(wandb_cfg.get("mode", "offline")) == "offline",
        )
    else:
        logger = False

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=int(training_cfg["max_epochs"]),
        gradient_clip_val=float(training_cfg["gradient_clip"]),
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
    )

    started_at = utc_now()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    best_path = Path(checkpoint_callback.best_model_path)
    if not best_path.exists():
        raise RuntimeError("Training finished without a best checkpoint")

    best_model = LitSmilesGPT.load_from_checkpoint(str(best_path))
    best_model = best_model.to("cuda" if torch.cuda.is_available() else "cpu")

    requested = int(sampling_cfg["n_sample"])
    sampling_batch = int(sampling_cfg["batch_size"])
    generated: list[str] = []
    while len(generated) < requested:
        current_batch = min(sampling_batch, requested - len(generated))
        generated.extend(
            sample_smiles(
                best_model,
                vocab,
                batch_size=current_batch,
                temperature=float(sampling_cfg["temperature"]),
                top_p=float(sampling_cfg["top_p"]),
                max_len=max_len,
            )
        )

    raw_path = output_dir / "generated_all_smiles.txt"
    raw_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
    metrics = simple_generation_metrics(generated, train_set)
    novel_path = output_dir / "generated_novel_smiles.txt"
    novel_path.write_text("\n".join(metrics.pop("valid_novel")) + "\n", encoding="utf-8")

    training_summary = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "seed": seed,
        "parameter_count": parameter_count,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "validation_size": val_size,
        "vocab_size": len(vocab),
        "best_checkpoint": str(best_path),
        "best_val_loss": float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else None,
        "best_val_perplexity": math.exp(float(checkpoint_callback.best_model_score))
        if checkpoint_callback.best_model_score is not None
        else None,
        "generation_metrics": metrics,
    }
    write_json(output_dir / "training_summary.json", training_summary)
    write_json(output_dir / "vocab.json", vocab)
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
