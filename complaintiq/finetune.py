"""Fine-tune a small transformer classifier (DistilBERT) on the GPU, with on-disk caching.

Low-impact settings: one run, at most two epochs, max 256 tokens, fp16 autocast,
dynamic padding, no DataLoader workers. Weights, the epoch history and predictions are
cached under a key built from the config and the exact training ids.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from complaintiq.stats import macro_f1


@dataclass(frozen=True)
class FinetuneConfig:
    model_id: str = "distilbert-base-uncased"
    max_length: int = 256
    epochs: int = 2
    batch_size: int = 32
    lr: float = 5e-5
    warmup_frac: float = 0.06
    weight_decay: float = 0.01
    seed: int = 0


def cache_key(cfg: FinetuneConfig, train_ids: list[str], labels: list[str]) -> str:
    """Stable key over everything that changes the trained weights."""
    payload = json.dumps({"cfg": asdict(cfg), "ids": train_ids, "labels": labels}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chunks(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


def _bucketed(lengths: np.ndarray, batch_size: int, rng: np.random.Generator, pool: int = 50) -> list[np.ndarray]:
    """Shuffled batches of similar-length documents: sort inside random pools of `pool` batches."""
    perm = rng.permutation(len(lengths))
    batches: list[np.ndarray] = []
    for s in range(0, len(perm), batch_size * pool):
        block = perm[s : s + batch_size * pool]
        batches += _chunks(block[np.argsort(lengths[block], kind="stable")], batch_size)
    return [batches[i] for i in rng.permutation(len(batches))]


def _batches(enc: dict[str, list[list[int]]], batches: list[np.ndarray], pad_id: int):
    """Yield dynamically padded (input_ids, attention_mask, index) batches."""
    for idx in batches:
        seqs = [enc["input_ids"][i] for i in idx]
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, : len(x)] = torch.tensor(x)
            mask[r, : len(x)] = 1
        yield ids, mask, idx


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 128) -> np.ndarray:
    """Class probabilities; sorts by length so padding is minimal, then restores the input order."""
    device = _device()
    model.eval()
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    order = np.argsort([len(x) for x in enc["input_ids"]], kind="stable")
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for ids, mask, idx in _batches(enc, _chunks(order, batch_size), tokenizer.pad_token_id):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def finetune(
    cfg: FinetuneConfig,
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    labels: list[str],
    out_dir: Path,
    log=print,
) -> tuple[Path, list[dict]]:
    """Train with class-balanced cross-entropy and keep the epoch with the best validation macro-F1.

    Returns the directory with the saved model and the per-epoch history. If the directory
    already holds a finished run, nothing is recomputed.
    """
    done = out_dir / "history.json"
    if done.exists():
        return out_dir, json.loads(done.read_text())

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = _device()
    n_classes = len(labels)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=n_classes,
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
    ).to(device)

    enc = tokenizer(train_texts, truncation=True, max_length=cfg.max_length)
    lengths = np.array([len(x) for x in enc["input_ids"]])
    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.tensor(len(y_train) / (n_classes * counts.clip(min=1)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    y_t = torch.tensor(y_train, dtype=torch.long)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr)
    steps = cfg.epochs * len(_bucketed(lengths, cfg.batch_size, np.random.default_rng(0)))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_frac * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best = [], -1.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        batches = _bucketed(lengths, cfg.batch_size, rng)
        for step, (ids, mask, idx) in enumerate(_batches(enc, batches, tokenizer.pad_token_id)):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), y_t[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
            if step % 100 == 0:
                log(f"  epoch {epoch} step {step}: loss {loss.item():.3f}")
        train_s = time.perf_counter() - t0
        val_pred = predict_proba(model, tokenizer, val_texts, cfg.max_length).argmax(axis=1)
        score = macro_f1(y_val, val_pred, n_classes)
        history.append({"epoch": epoch, "train_loss": total / (step + 1), "val_macro_f1": score, "train_s": train_s})
        log(f"  epoch {epoch}: val macro-F1 {score:.4f} ({train_s:.0f} s)")
        if score > best:
            best = score
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    done.write_text(json.dumps(history, indent=2))
    return out_dir, history


def load_finetuned(model_dir: Path):
    """Model (on the GPU when available, eval mode) and tokenizer from a finished run."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device()).eval()
    return model, tokenizer
