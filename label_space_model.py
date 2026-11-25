"""
One-Class Label-Space Diffusion Model for Real-Time Time-Series Anomaly Detection.


Key architectural distinction vs. Latent-Space Diffusion:
1. Raw sensor window X [B, L, D] is NEVER noised or reconstructed. It is encoded once
   by a Causal Dilated TCN into context c [B, H].
2. Unsupervised Normal Operating Regimes (K prototypes) are discovered on normal
   training data only, producing soft regime label vectors y_0 in Delta^{K-1}.
3. Gaussian noise is added ONLY to the K-dimensional label vector y_0 -> y_t.
4. A lightweight FiLM-conditioned MLP denoiser predicts clean y_0 from (y_t, t, c)
   using 4-step deterministic DDIM across M parallel noise seeds.
5. Anomaly score combines Regime Consensus Mismatch (JSD to reference regime) and
   Multi-Seed Denoising Variance (scatters when X lies off the normal manifold).
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class CausalDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = F.pad(x, (self.pad, 0))
        h = F.silu(self.norm1(self.conv1(h)))
        h = F.pad(h, (self.pad, 0))
        h = F.silu(self.norm2(self.conv2(h)))
        return x + h


class ConditionEncoder(nn.Module):
    """
    Fast Causal TCN + Temporal Attention Pooling.
    Runs ONCE per incoming window X [B, L, D] -> normalized context c [B, H].
    Includes a self-supervised next-step forecasting head for stage-1 pretraining
    on normal data so c captures physical sensor-actuator dynamics.
    """
    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        if hidden % 8 != 0:
            raise ValueError("hidden must be divisible by 8")
        self.in_proj = nn.Conv1d(in_features, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalDilatedBlock(hidden, dilation=1),
            CausalDilatedBlock(hidden, dilation=2),
            CausalDilatedBlock(hidden, dilation=4),
        ])
        self.attn_weights = nn.Linear(hidden, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        # Auxiliary self-supervised head (used only during normal pretraining)
        self.forecast_head = nn.Linear(hidden, in_features)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> h: [B, L, H]
        h = self.in_proj(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)  # [B, L, H]
        w = torch.softmax(self.attn_weights(h), dim=1)  # [B, L, 1]
        pooled = (w * h).sum(dim=1)  # [B, H]
        last = h[:, -1, :]  # [B, H] (captures latest streaming state)
        c = self.out_proj(torch.cat([pooled, last], dim=-1))
        return F.normalize(c, dim=-1)


class RegimePrototypeBank(nn.Module):
    """
    Discovers K normal operating regimes on normal training embeddings and converts
    any context vector c into a soft probability label y_0 in Delta^{K-1}.
    """
    def __init__(self, num_regimes: int = 12, hidden: int = 128, temperature: float = 0.15):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = temperature
        self.register_buffer("prototypes", torch.randn(num_regimes, hidden))

    @torch.no_grad()
    def fit(self, embeddings: np.ndarray, seed: int = 42) -> None:
        # L2-normalized embeddings -> Spherical K-Means via normalized centroids
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        unit_emb = embeddings / norms
        k = min(self.num_regimes, len(unit_emb))
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(2048, len(unit_emb)),
            n_init=10,
        )
        km.fit(unit_emb)
        centers = km.cluster_centers_
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        if k < self.num_regimes:
            pad = np.repeat(centers[:1], self.num_regimes - k, axis=0)
            centers = np.vstack([centers, pad])
        self.prototypes.copy_(torch.from_numpy(centers.astype(np.float32)))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B, H] (unit-normalized), prototypes: [K, H] (unit-normalized)
        c_norm = F.normalize(c, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(c_norm, p_norm.t()) / self.temperature
        return torch.softmax(logits, dim=-1)  # [B, K] in probability simplex


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, dim * 2)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        x = self.fc1(F.silu(self.norm1(h)))
        x = x * (1.0 + scale) + shift
        x = self.fc2(F.silu(self.norm2(x)))
        return h + x


class LabelDenoiser(nn.Module):
    """
    Ultra-lightweight denoiser operating strictly on the K-dimensional label vector y_t.
    Predicts clean label logits -> softmax probabilities y_hat_0 in Delta^{K-1} (x_0 prediction).
    """
    def __init__(self, num_regimes: int = 12, cond_dim: int = 128, width: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbed(cond_dim)
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(num_regimes, width)
        self.blocks = nn.ModuleList([
            FiLMResBlock(width, cond_dim),
            FiLMResBlock(width, cond_dim),
            FiLMResBlock(width, cond_dim),
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, num_regimes),
        )

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # y_t: [B, K], t: [B], c: [B, H] -> returns logits [B, K]
        t_feat = self.time_emb(t)
        cond = self.cond_fuse(torch.cat([c, t_feat], dim=-1))
        h = self.in_proj(y_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(h)


class OneClassLabelDiffusion(nn.Module):
    """
    End-to-end One-Class Label-Space Diffusion detector.
    """
    def __init__(
        self,
        in_features: int,
        num_regimes: int = 12,
        hidden: int = 128,
        steps: int = 100,
        ddim_steps: int = 4,
        num_seeds: int = 8,
        var_weight: float = 2.0,
        temperature: float = 0.15,
    ):
        super().__init__()
        self.num_regimes = num_regimes
        self.steps = steps
        self.ddim_steps = ddim_steps
        self.num_seeds = num_seeds
        self.var_weight = var_weight

        self.encoder = ConditionEncoder(in_features, hidden=hidden)
        self.bank = RegimePrototypeBank(num_regimes, hidden=hidden, temperature=temperature)
        self.denoiser = LabelDenoiser(num_regimes, cond_dim=hidden, width=hidden)

        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, y_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(y_0)
        ab = self.alpha_bar[t, None]
        y_t = ab.sqrt() * y_0 + (1.0 - ab).sqrt() * eps
        return y_t, eps

    def diffusion_loss(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            c = self.encoder(x)
            y_0 = self.bank(c)  # [B, K] clean soft regime label
        b = x.shape[0]
        t = torch.randint(0, self.steps, (b,), device=x.device)
        y_t, _ = self.q_sample(y_0, t)
        logits = self.denoiser(y_t, t, c)
        y_hat = torch.softmax(logits, dim=-1)
        mse = F.mse_loss(y_hat, y_0)
        kl = F.kl_div(torch.log_softmax(logits, dim=-1), y_0, reduction="batchmean")
        return mse + 0.1 * kl

    @torch.no_grad()
    def ddim_sample_from_context(
        self,
        c: torch.Tensor,
        num_seeds: int | None = None,
        ddim_steps: int | None = None,
        return_trajectories: bool = False,
    ):
        """
        Runs fast deterministic DDIM sampling in K-dimensional label space.
        Condition vector c [B, H] is computed ONCE and reused across all steps and seeds.
        """
        m = num_seeds or self.num_seeds
        k_steps = ddim_steps or self.ddim_steps
        b, h = c.shape
        device = c.device

        # Expand condition across M parallel noise seeds: [B*M, H]
        c_rep = c.unsqueeze(1).expand(b, m, h).reshape(b * m, h)
        y_t = torch.randn(b * m, self.num_regimes, device=device)

        # Uniformly spaced descending timesteps from T-1 to 0
        ts = torch.linspace(self.steps - 1, 0, k_steps, device=device).long()
        traj = [y_t.view(b, m, self.num_regimes)] if return_trajectories else None

        for i in range(len(ts)):
            t_cur = ts[i]
            t_batch = torch.full((b * m,), t_cur, device=device, dtype=torch.long)
            logits = self.denoiser(y_t, t_batch, c_rep)
            y0_pred = torch.softmax(logits, dim=-1)

            if i == len(ts) - 1:
                y_t = y0_pred
            else:
                t_prev = ts[i + 1]
                ab_cur = self.alpha_bar[t_cur]
                ab_prev = self.alpha_bar[t_prev]
                eps_pred = (y_t - ab_cur.sqrt() * y0_pred) / (1.0 - ab_cur).sqrt().clamp_min(1e-6)
                y_t = ab_prev.sqrt() * y0_pred + (1.0 - ab_prev).sqrt() * eps_pred

            if return_trajectories:
                traj.append(y_t.view(b, m, self.num_regimes))

        y0_samples = y_t.view(b, m, self.num_regimes)  # [B, M, K]
        if return_trajectories:
            return y0_samples, traj
        return y0_samples

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes real-time anomaly score S(X):
        1. Regime Consensus Mismatch: mean Jensen-Shannon divergence between denoised
           samples y_hat_0^(m) and the encoder's reference regime assignment y_ref.
        2. Multi-Seed Denoising Variance: dispersion across M parallel noise trajectories.
        """
        c = self.encoder(x)
        y_ref = self.bank(c)  # [B, K]
        y_samples = self.ddim_sample_from_context(c)  # [B, M, K]

        # 1. Jensen-Shannon divergence between each seed sample and y_ref
        p = y_samples.clamp_min(1e-8)
        q = y_ref.unsqueeze(1).expand_as(p).clamp_min(1e-8)
        m_dist = 0.5 * (p + q)
        kl_pm = (p * (p.log() - m_dist.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m_dist.log())).sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm).mean(dim=1)  # [B]

        # 2. Inter-seed variance (epistemic scatter across noise seeds)
        y_mean = y_samples.mean(dim=1, keepdim=True)
        seed_var = (y_samples - y_mean).square().sum(dim=-1).mean(dim=1)  # [B]

        total_score = jsd + self.var_weight * seed_var
        details = {
            "c": c,
            "y_ref": y_ref,
            "y_samples": y_samples,
            "jsd_mismatch": jsd,
            "seed_variance": seed_var,
            "total_score": total_score,
        }
        return total_score, details