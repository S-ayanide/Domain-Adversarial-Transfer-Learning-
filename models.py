"""
Neural network architectures for domain adaptation fault identification.

Implements:
  - DATL   : Domain-Adversarial Transfer Learning (proposed model in paper)
             = Feature Extractor + Classifier + Domain Discriminator + MMD + Pseudo-labels
  - DANN   : Domain-Adversarial Neural Network  (Ganin et al. 2016)  [baseline]
  - CDAN   : Conditional Domain Adversarial Network (Long et al. 2018) [baseline]
  - FixBi  : Fixing the Bipartite Graph  (Na et al. 2021)             [baseline]
  - ToAlign: Task-Oriented Alignment     (Wei et al. 2021)            [baseline]

Reference paper:
  Fang & Gao, "Domain-Adversarial Transfer Learning for Fault Root Cause
  Identification in Cloud Computing Systems", RAIIC 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared building block: Gradient Reversal Layer
# ──────────────────────────────────────────────────────────────────────────────

class GRL(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Shared Feature Extractor  F(·; θ_f)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Shared feature extractor used by all models.
    Maps raw input features → latent representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Label Classifier  C(·; θ_c)
# ──────────────────────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, n_classes),
        )

    def forward(self, feat):
        return self.net(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Domain Discriminator  D(·; θ_d)
# ──────────────────────────────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, alpha: float = 1.0):
        return self.net(grad_reverse(feat, alpha))


# ──────────────────────────────────────────────────────────────────────────────
# MMD loss  (Eq. 2 in paper)
# ──────────────────────────────────────────────────────────────────────────────

def mmd_loss(src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
    """
    Maximum Mean Discrepancy loss with linear kernel.

    L_mmd = || (1/ns) Σ F(x_s) - (1/nt) Σ F(x_t) ||²
    """
    src_mean = src_feat.mean(dim=0)
    tgt_mean = tgt_feat.mean(dim=0)
    return torch.sum((src_mean - tgt_mean) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# ─── 1. DATL — Proposed model (paper Eq. 1–4) ─────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class DATL(nn.Module):
    """
    Domain-Adversarial Transfer Learning model.

    Architecture (paper Figure 1):
      F  – shared feature extractor
      C  – label classifier (source supervision)
      D  – domain discriminator (adversarial)

    Total loss (paper Eq. 4):
      L_total = L_s + λ1 * L_mmd + λ2 * L_adv

    Pseudo-label mechanism:
      High-confidence target predictions (p ≥ δ) are added to
      the training set as pseudo-labeled source samples.
    """

    def __init__(
        self,
        input_dim:  int,
        n_classes:  int,
        hidden_dim: int   = 128,
        dropout:    float = 0.3,
        lambda1:    float = 0.1,   # MMD weight
        lambda2:    float = 0.1,   # adversarial weight
        pseudo_threshold: float = 0.85,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.pseudo_threshold = pseudo_threshold

        self.F = FeatureExtractor(input_dim, hidden_dim, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = DomainDiscriminator(self.F.out_dim, dropout)

    def forward(self, x_src, x_tgt=None, alpha: float = 1.0):
        """
        Returns:
          cls_logits   – classification logits for x_src
          domain_src   – domain prediction for x_src  (0 = source)
          domain_tgt   – domain prediction for x_tgt  (1 = target)
          feat_src     – source features (for MMD)
          feat_tgt     – target features (for MMD)
        """
        feat_src    = self.F(x_src)
        cls_logits  = self.C(feat_src)
        domain_src  = self.D(feat_src, alpha)

        if x_tgt is not None:
            feat_tgt   = self.F(x_tgt)
            domain_tgt = self.D(feat_tgt, alpha)
        else:
            feat_tgt   = None
            domain_tgt = None

        return cls_logits, domain_src, domain_tgt, feat_src, feat_tgt

    def compute_total_loss(
        self,
        cls_logits:  torch.Tensor,
        y_src:       torch.Tensor,
        domain_src:  torch.Tensor,
        domain_tgt:  torch.Tensor,
        feat_src:    torch.Tensor,
        feat_tgt:    torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        L_total = L_s + λ1*L_mmd + λ2*L_adv          (paper Eq. 4)

        L_s   = cross-entropy on source labels          (paper Eq. 1)
        L_mmd = squared distance of feature means       (paper Eq. 2)
        L_adv = binary cross-entropy on domain labels   (paper Eq. 3)
        """
        # L_s: classification loss on source
        L_s = F.cross_entropy(cls_logits, y_src)

        # L_mmd: Maximum Mean Discrepancy
        L_mmd = mmd_loss(feat_src, feat_tgt)

        # L_adv: adversarial domain loss (binary cross-entropy)
        zeros = torch.zeros_like(domain_src)   # source = 0
        ones  = torch.ones_like(domain_tgt)    # target = 1
        L_adv = (F.binary_cross_entropy(domain_src, zeros)
                 + F.binary_cross_entropy(domain_tgt, ones))

        L_total = L_s + self.lambda1 * L_mmd + self.lambda2 * L_adv
        return L_total, {"L_s": L_s.item(), "L_mmd": L_mmd.item(), "L_adv": L_adv.item()}

    @torch.no_grad()
    def get_pseudo_labels(self, x_tgt: torch.Tensor):
        """
        Returns high-confidence pseudo-labeled target samples.
        Selected when max_probability ≥ pseudo_threshold (δ).
        """
        feat   = self.F(x_tgt)
        logits = self.C(feat)
        probs  = F.softmax(logits, dim=1)
        max_prob, pred = probs.max(dim=1)
        mask   = max_prob >= self.pseudo_threshold
        return x_tgt[mask], pred[mask], mask

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feat   = self.F(x)
            logits = self.C(feat)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# ─── 2. DANN — Domain-Adversarial Neural Network baseline ────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class DANN(nn.Module):
    """
    Standard DANN (Ganin et al. 2016).
    Same as DATL but without MMD and without pseudo-labels.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.F = FeatureExtractor(input_dim, hidden_dim, dropout)
        self.C = Classifier(self.F.out_dim, n_classes, dropout)
        self.D = DomainDiscriminator(self.F.out_dim, dropout)

    def forward(self, x_src, x_tgt=None, alpha: float = 1.0):
        feat_src   = self.F(x_src)
        cls_logits = self.C(feat_src)
        dom_src    = self.D(feat_src, alpha)

        dom_tgt, feat_tgt = None, None
        if x_tgt is not None:
            feat_tgt = self.F(x_tgt)
            dom_tgt  = self.D(feat_tgt, alpha)

        return cls_logits, dom_src, dom_tgt, feat_src, feat_tgt

    def compute_loss(self, cls_logits, y_src, dom_src, dom_tgt, lambda_adv=0.1):
        L_s   = F.cross_entropy(cls_logits, y_src)
        zeros = torch.zeros_like(dom_src)
        ones  = torch.ones_like(dom_tgt)
        L_adv = (F.binary_cross_entropy(dom_src, zeros)
                 + F.binary_cross_entropy(dom_tgt, ones))
        return L_s + lambda_adv * L_adv

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.C(self.F(x))


# ──────────────────────────────────────────────────────────────────────────────
# ─── 3. CDAN — Conditional Domain Adversarial Network baseline ───────────────
# ──────────────────────────────────────────────────────────────────────────────

class CDANDiscriminator(nn.Module):
    """
    CDAN domain discriminator conditioned on classifier predictions.
    Input = feature ⊗ softmax(classifier) (multilinear map)
    """

    def __init__(self, feature_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        in_dim = feature_dim * n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, cls_prob, alpha=1.0):
        # Multilinear conditioning: outer product flattened
        joint = torch.bmm(feat.unsqueeze(2), cls_prob.unsqueeze(1))
        joint = joint.view(joint.size(0), -1)
        return self.net(grad_reverse(joint, alpha))


class CDAN(nn.Module):
    """
    Conditional Domain Adversarial Network (Long et al. 2018).
    Domain discriminator is conditioned on classifier softmax output.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.F  = FeatureExtractor(input_dim, hidden_dim, dropout)
        self.C  = Classifier(self.F.out_dim, n_classes, dropout)
        self.D  = CDANDiscriminator(self.F.out_dim, n_classes, dropout)

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        feat_src   = self.F(x_src)
        cls_logits = self.C(feat_src)
        cls_prob   = F.softmax(cls_logits.detach(), dim=1)
        dom_src    = self.D(feat_src, cls_prob, alpha)

        dom_tgt, feat_tgt = None, None
        if x_tgt is not None:
            feat_tgt    = self.F(x_tgt)
            tgt_logits  = self.C(feat_tgt)
            tgt_prob    = F.softmax(tgt_logits.detach(), dim=1)
            dom_tgt     = self.D(feat_tgt, tgt_prob, alpha)

        return cls_logits, dom_src, dom_tgt, feat_src, feat_tgt

    def compute_loss(self, cls_logits, y_src, dom_src, dom_tgt, lambda_adv=0.1):
        L_s   = F.cross_entropy(cls_logits, y_src)
        zeros = torch.zeros_like(dom_src)
        ones  = torch.ones_like(dom_tgt)
        L_adv = (F.binary_cross_entropy(dom_src, zeros)
                 + F.binary_cross_entropy(dom_tgt, ones))
        return L_s + lambda_adv * L_adv

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.C(self.F(x))


# ──────────────────────────────────────────────────────────────────────────────
# ─── 4. FixBi — Fixing the Bipartite Graph baseline ──────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class FixBi(nn.Module):
    """
    FixBi simplified adaptation for tabular/1D fault data.
    (Na et al., CVPR 2021)

    Maintains two classifiers: one source-dominant (SDC), one target-dominant (TDC).
    Uses consistency regularization between them.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.F    = FeatureExtractor(input_dim, hidden_dim, dropout)
        self.C_s  = Classifier(self.F.out_dim, n_classes, dropout)   # source-dominant
        self.C_t  = Classifier(self.F.out_dim, n_classes, dropout)   # target-dominant

    def forward(self, x_src, x_tgt=None, bim_ratio=0.5):
        feat_src     = self.F(x_src)
        logits_s_src = self.C_s(feat_src)
        logits_t_src = self.C_t(feat_src)

        if x_tgt is not None:
            feat_tgt     = self.F(x_tgt)
            logits_s_tgt = self.C_s(feat_tgt)
            logits_t_tgt = self.C_t(feat_tgt)
        else:
            logits_s_tgt = logits_t_tgt = feat_tgt = None

        return logits_s_src, logits_t_src, logits_s_tgt, logits_t_tgt, feat_src, feat_tgt

    def compute_loss(self, logits_s_src, logits_t_src,
                     logits_s_tgt, logits_t_tgt, y_src, lambda_cons=0.1):
        # Classification loss (both classifiers on source)
        L_cls = (F.cross_entropy(logits_s_src, y_src)
                 + F.cross_entropy(logits_t_src, y_src)) / 2

        # Consistency regularization on target: soft-labels from SDC guide TDC
        if logits_s_tgt is not None and logits_t_tgt is not None:
            soft_s = F.softmax(logits_s_tgt.detach(), dim=1)
            soft_t = F.softmax(logits_t_tgt.detach(), dim=1)
            L_cons = F.kl_div(F.log_softmax(logits_t_tgt, dim=1), soft_s, reduction="batchmean")
            L_cons += F.kl_div(F.log_softmax(logits_s_tgt, dim=1), soft_t, reduction="batchmean")
        else:
            L_cons = torch.tensor(0.0)

        return L_cls + lambda_cons * L_cons

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feat = self.F(x)
            # Ensemble of both classifiers
            return (self.C_s(feat) + self.C_t(feat)) / 2


# ──────────────────────────────────────────────────────────────────────────────
# ─── 5. ToAlign — Task-Oriented Alignment baseline ───────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ToAlign(nn.Module):
    """
    Task-Oriented Alignment (Wei et al., NeurIPS 2021).

    Key idea: decomposes features into task-relevant and task-irrelevant parts;
    only aligns task-relevant part to avoid negative transfer.
    Simplified to 1D tabular data.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.F    = FeatureExtractor(input_dim, hidden_dim, dropout)
        self.C    = Classifier(self.F.out_dim, n_classes, dropout)
        self.D    = DomainDiscriminator(self.F.out_dim, dropout)
        # Task-relevance gate: produces per-feature weights
        self.gate = nn.Sequential(
            nn.Linear(self.F.out_dim, self.F.out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_src, x_tgt=None, alpha=1.0):
        feat_src     = self.F(x_src)
        gate_src     = self.gate(feat_src)
        task_feat_s  = feat_src * gate_src            # task-relevant part
        cls_logits   = self.C(task_feat_s)
        dom_src      = self.D(task_feat_s, alpha)

        dom_tgt, feat_tgt = None, None
        if x_tgt is not None:
            feat_tgt     = self.F(x_tgt)
            gate_tgt     = self.gate(feat_tgt)
            task_feat_t  = feat_tgt * gate_tgt
            dom_tgt      = self.D(task_feat_t, alpha)

        return cls_logits, dom_src, dom_tgt, feat_src, feat_tgt

    def compute_loss(self, cls_logits, y_src, dom_src, dom_tgt, lambda_adv=0.1):
        L_s   = F.cross_entropy(cls_logits, y_src)
        zeros = torch.zeros_like(dom_src)
        ones  = torch.ones_like(dom_tgt)
        L_adv = (F.binary_cross_entropy(dom_src, zeros)
                 + F.binary_cross_entropy(dom_tgt, ones))
        return L_s + lambda_adv * L_adv

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feat = self.F(x)
            return self.C(feat * self.gate(feat))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: alpha schedule for GRL (linearly increases from 0 → 1)
# ──────────────────────────────────────────────────────────────────────────────

def grl_alpha(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """Smooth alpha schedule used in DANN paper."""
    p = epoch / total_epochs
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0)
