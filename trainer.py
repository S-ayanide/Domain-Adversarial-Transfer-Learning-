"""
Generic training engine for all domain adaptation models.

Handles:
  - Batched source + target data loading
  - Epoch loop with GRL alpha schedule
  - Pseudo-label injection (DATL only)
  - Evaluation on labeled target samples
  - Metric logging (accuracy, F1-macro, AUC-ovr)
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report)
from torch.utils.data import DataLoader, TensorDataset

from models import DATL, DANN, CDAN, FixBi, ToAlign, grl_alpha

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def make_loader(X, y=None, batch_size=256, shuffle=True) -> DataLoader:
    X_t = to_tensor(X)
    if y is not None:
        ds = TensorDataset(X_t, to_tensor(y, torch.long))
    else:
        ds = TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    model.eval()
    X_t  = to_tensor(X)
    with torch.no_grad():
        logits = model.predict(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)

    # AUC: suppress sklearn's single-class warning; return nan when undefined
    present = np.unique(y)
    if len(present) < 2:
        auc = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if n_classes == 2:
                    auc = roc_auc_score(y, probs[:, 1])
                else:
                    auc = roc_auc_score(y, probs, multi_class="ovr",
                                        average="macro",
                                        labels=list(range(n_classes)))
            except ValueError:
                auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


# ──────────────────────────────────────────────────────────────────────────────
# DATL Trainer (proposed model)
# ──────────────────────────────────────────────────────────────────────────────

def train_datl(
    model:         DATL,
    X_src:         np.ndarray,
    y_src:         np.ndarray,
    X_tgt_all:     np.ndarray,
    y_tgt_all:     np.ndarray,
    tgt_labeled:   np.ndarray,   # boolean mask: which target rows are labeled
    n_classes:     int,
    epochs:        int  = 80,
    lr:            float = 1e-3,
    batch_size:    int  = 256,
    pseudo_update_freq: int = 10,   # recompute pseudo-labels every N epochs
    save_path:     Optional[Path] = None,
) -> dict:
    """Train DATL and return metrics on labeled target test set."""

    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tgt_ul = X_tgt_all[~tgt_labeled]   # unlabeled target (for domain alignment)
    X_tgt_lb = X_tgt_all[tgt_labeled]    # labeled target   (for evaluation)
    y_tgt_lb = y_tgt_all[tgt_labeled]

    # If all target samples are labeled, use the labeled set for domain alignment too
    if len(X_tgt_ul) == 0:
        X_tgt_ul = X_tgt_lb

    # Working source set (will be augmented with pseudo-labels)
    X_src_work = X_src.copy()
    y_src_work = y_src.copy()

    history = {"train_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}
    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = grl_alpha(epoch, epochs)

        # ── Pseudo-label update ──────────────────────────────────────────────
        if epoch % pseudo_update_freq == 0 and epoch > 10:
            model.eval()
            X_ul_t = to_tensor(X_tgt_ul)
            _, pred_pl, mask = model.get_pseudo_labels(X_ul_t)
            if mask.sum() > 0:
                X_pl = X_tgt_ul[mask.cpu().numpy()]
                y_pl = pred_pl.cpu().numpy()
                X_src_work = np.vstack([X_src, X_pl])
                y_src_work = np.concatenate([y_src, y_pl])
            model.train()

        # ── Mini-batch training ──────────────────────────────────────────────
        src_loader = make_loader(X_src_work, y_src_work, batch_size, shuffle=True)
        tgt_iter   = iter(make_loader(X_tgt_ul, batch_size=batch_size, shuffle=True))

        epoch_loss = 0.0
        n_batches  = 0
        for batch in src_loader:
            x_s, y_s = batch[0], batch[1]

            try:
                x_t = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(make_loader(X_tgt_ul, batch_size=batch_size, shuffle=True))
                x_t = next(tgt_iter)[0]

            # Align batch sizes
            min_sz = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:min_sz], y_s[:min_sz], x_t[:min_sz]

            opt.zero_grad()
            cls_logits, dom_src, dom_tgt, feat_src, feat_tgt = model(x_s, x_t, alpha)
            loss, _ = model.compute_total_loss(cls_logits, y_s, dom_src,
                                               dom_tgt, feat_src, feat_tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics  = evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc"])

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            logger.info(f"  [DATL] epoch {epoch:3d}/{epochs} | "
                        f"loss={avg_loss:.4f} | acc={metrics['accuracy']:.4f} | "
                        f"f1={metrics['f1']:.4f} | auc={metrics['auc']:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    if save_path:
        torch.save(model.state_dict(), save_path)

    final = evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Generic trainer for DANN / CDAN / ToAlign  (adversarial-only models)
# ──────────────────────────────────────────────────────────────────────────────

def train_adversarial(
    model,
    model_name:   str,
    X_src:        np.ndarray,
    y_src:        np.ndarray,
    X_tgt_all:    np.ndarray,
    y_tgt_all:    np.ndarray,
    tgt_labeled:  np.ndarray,
    n_classes:    int,
    epochs:       int   = 80,
    lr:           float = 1e-3,
    batch_size:   int   = 256,
    lambda_adv:   float = 0.1,
    save_path:    Optional[Path] = None,
) -> dict:

    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tgt_ul = X_tgt_all[~tgt_labeled]
    X_tgt_lb = X_tgt_all[tgt_labeled]
    y_tgt_lb = y_tgt_all[tgt_labeled]
    if len(X_tgt_ul) == 0:
        X_tgt_ul = X_tgt_lb

    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        alpha      = grl_alpha(epoch, epochs)
        src_loader = make_loader(X_src, y_src, batch_size, shuffle=True)
        tgt_iter   = iter(make_loader(X_tgt_ul, batch_size=batch_size, shuffle=True))

        for batch in src_loader:
            x_s, y_s = batch[0], batch[1]
            try:
                x_t = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(make_loader(X_tgt_ul, batch_size=batch_size))
                x_t = next(tgt_iter)[0]

            min_sz = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:min_sz], y_s[:min_sz], x_t[:min_sz]

            opt.zero_grad()
            out = model(x_s, x_t, alpha)
            cls_logits, dom_src, dom_tgt = out[0], out[1], out[2]
            loss = model.compute_loss(cls_logits, y_s, dom_src, dom_tgt, lambda_adv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()
        metrics = evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            logger.info(f"  [{model_name}] epoch {epoch:3d}/{epochs} | "
                        f"acc={metrics['accuracy']:.4f} | "
                        f"f1={metrics['f1']:.4f} | auc={metrics['auc']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    if save_path:
        torch.save(model.state_dict(), save_path)

    return evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# FixBi trainer (two classifiers)
# ──────────────────────────────────────────────────────────────────────────────

def train_fixbi(
    model:       FixBi,
    X_src:       np.ndarray,
    y_src:       np.ndarray,
    X_tgt_all:   np.ndarray,
    y_tgt_all:   np.ndarray,
    tgt_labeled: np.ndarray,
    n_classes:   int,
    epochs:      int   = 80,
    lr:          float = 1e-3,
    batch_size:  int   = 256,
    lambda_cons: float = 0.1,
    save_path:   Optional[Path] = None,
) -> dict:

    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tgt_ul = X_tgt_all[~tgt_labeled]
    X_tgt_lb = X_tgt_all[tgt_labeled]
    y_tgt_lb = y_tgt_all[tgt_labeled]
    if len(X_tgt_ul) == 0:
        X_tgt_ul = X_tgt_lb

    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        src_loader = make_loader(X_src, y_src, batch_size, shuffle=True)
        tgt_iter   = iter(make_loader(X_tgt_ul, batch_size=batch_size, shuffle=True))

        for batch in src_loader:
            x_s, y_s = batch[0], batch[1]
            try:
                x_t = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(make_loader(X_tgt_ul, batch_size=batch_size))
                x_t = next(tgt_iter)[0]

            min_sz = min(x_s.size(0), x_t.size(0))
            x_s, y_s, x_t = x_s[:min_sz], y_s[:min_sz], x_t[:min_sz]

            opt.zero_grad()
            logits_s_src, logits_t_src, logits_s_tgt, logits_t_tgt, _, _ = model(x_s, x_t)
            loss = model.compute_loss(logits_s_src, logits_t_src,
                                      logits_s_tgt, logits_t_tgt, y_s, lambda_cons)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()
        metrics = evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            logger.info(f"  [FixBi] epoch {epoch:3d}/{epochs} | "
                        f"acc={metrics['accuracy']:.4f} | "
                        f"f1={metrics['f1']:.4f} | auc={metrics['auc']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    if save_path:
        torch.save(model.state_dict(), save_path)

    return evaluate(model, X_tgt_lb, y_tgt_lb, n_classes)
