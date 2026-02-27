# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

import math


class _MaskedLoss(torch.nn.Module):
    def forward(self, estimate, output, mask=None):
        feature_mask = mask.expand_as(estimate)
        return self._loss(estimate[feature_mask], output[feature_mask])


class L1Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.L1Loss()


class L2Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.MSELoss()


class ClipLoss(torch.nn.Module):
    """CLIP (See Open AI CLIP) constrastive loss.
    """
    def __init__(self, linear=None, twin=True, pool=False, tmin=None, tmax=None,
                 tmin_train=None, tmax_train=None, dset_args=None, center=False):
        super().__init__()
        self.linear = None
        self.pool = pool
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.tmin = tmin
        self.tmax = tmax
        self.tmin_train = tmin_train
        self.tmax_train = tmax_train
        self.dset_args = dset_args

    def trim_samples(self, estimates, candidates):
        """Given estimates that is [B1, C, T] and candidates
        which is [B2, C, T], return estimates_trim of size [B1, C, T']
        and candidates_trim of size [B2, C, T'], such that T'
        corresponds to the samples between [self.tmin, self.tmax]
        """
        if self.training and (self.tmin_train is not None or self.tmax_train is not None):
            tmin, tmax = self.tmin_train, self.tmax_train
        else:
            tmin, tmax = self.tmin, self.tmax
        if (tmin is not None) or (tmax is not None):
            assert self.dset_args is not None
            assert self.dset_args.tmin is not None
            dset_tmin = self.dset_args.tmin
        if tmin is None:
            trim_min = 0
        else:
            assert tmin >= dset_tmin, 'clip.tmin should be above dset.tmin'
            trim_min = int((-dset_tmin + tmin) * self.dset_args.sample_rate)
        if tmax is None:
            trim_max = estimates.shape[-1]
        else:
            trim_max = int((-dset_tmin + tmax) * self.dset_args.sample_rate)
        estimates_trim = estimates[..., trim_min:trim_max]
        candidates_trim = candidates[..., trim_min:trim_max]
        return estimates_trim, candidates_trim

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of scores of matching.
        """
        estimates, candidates = self.trim_samples(estimates, candidates)
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        # We normalize inside the einsum, to avoid creating a copy
        # of candidates, which can be pretty big.
        scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
        """
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate, mask=None):
        """Warning: estimate and candidate are not symmetrical.
        If estimate of shape [B, C, T] and candidate of size [B', C, T]
        with B'>=B, the first B samples of candidate are targets, while
        the remaining B'-B samples of candidate are only used as negatives.
        """
        assert mask.all(), "mask is not supported for now"
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)


class FeatureDecodingLoss(torch.nn.Module):
    """
    Regresses features calculated on word stimulus using MSE, and a classification of
    a word segment with cross entropy.
    """
    def __init__(self, used_features, scaler):
        super().__init__()
        self.used_features = used_features
        self.scaler = scaler

    def forward(self, estimate, output, mask=None):
        assert estimate.shape[1] == self.used_features.output_dimension and \
               output.shape[1] == self.used_features.dimension, \
               "Invalid features dim received. Are you using the correct " \
               "features for the loss?"
        if mask is not None:
            assert mask.any()

        loss = 0
        for feature in self.used_features.values():
            feature_name = feature.name
            feature_slice = self.used_features.get_slice(feature_name)
            feature_slice_model_output = self.used_features.get_slice(
                feature_name, model_output=True)

            feature_estimate = estimate[:, feature_slice_model_output]
            feature_output = output[:, feature_slice]
            feature_mask = mask.expand_as(feature_estimate)

            if feature.categorical:
                # Classificaion loss
                assert feature_slice.stop - feature_slice.start == 1, \
                    "Supporting only single categorical cross entropy for now."
                assert feature.output_dimension > output[:, feature_slice.start].max(), \
                    f"feature output_dim is {feature.output_dimension} while output contains " \
                    f"categories up to {output[:, feature_slice.start].max()}"
                weights = self.scaler.get_categorical_feature_weights(feature_name).to(output) \
                    if self.scaler else None

                # Classes probabilities dim goes last, so feature_estimate shape is
                # [batch, seq-len, num-classes]
                feature_estimate = feature_estimate.transpose(1, 2)
                feature_output = feature_output.transpose(1, 2)
                feature_mask = feature_mask.transpose(1, 2)

                loss += F.cross_entropy(
                    feature_estimate[feature_mask].reshape(
                        -1, feature_slice_model_output.stop - feature_slice_model_output.start),
                    feature_output.long()[mask.transpose(1, 2)],
                    weights
                )
            else:
                # Regression loss
                loss += F.mse_loss(
                    feature_estimate[feature_mask], feature_output[feature_mask])

        return loss


class AADNLoss(torch.nn.Module):
    """
    Aligned Audio-MEG Decoding Network (AADN) loss.

    This wraps the standard CLIP-style contrastive loss used in the original
    BrainMagick codebase and adds:
      - IMM: InfoNCE-based mutual information maximization between cross-modal
        *consistent* representations.
      - CMM: CLUB-based mutual information minimization between modality-specific
        representations.
      - CRD: Cross-feature Reconstruction Decoder to mitigate information loss
        during disentanglement.

    The implementation follows the high-level description from
    \"Zero-Shot Speech Perception Decoding via Advancing Representation Consistency\"
    and is designed to be a drop-in replacement for the original ClipLoss, sharing
    the same inputs and interface.
    """

    def __init__(
        self,
        clip_args: dict,
        aadn_args: dict,
        dset_args=None,
    ):
        super().__init__()
        # Base CLIP loss (coarse-grained alignment, MACF baseline).
        clip_kwargs = dict(clip_args)
        clip_kwargs.pop(\"save_best\", None)
        clip_kwargs.pop(\"sync_grad\", None)
        self.clip = ClipLoss(**clip_kwargs, dset_args=dset_args)

        # Hyper-parameters for AADN components.
        self.mi_weight: float = float(aadn_args.get(\"mi_weight\", 1.0))
        self.club_weight: float = float(aadn_args.get(\"club_weight\", 0.1))
        self.recon_weight: float = float(aadn_args.get(\"recon_weight\", 0.1))
        self.proj_dim: int = int(aadn_args.get(\"proj_dim\", 256))
        self.tau: float = float(aadn_args.get(\"tau\", 0.07))

        # Projections for consistent and specific features (built lazily).
        # We rely on Lazy modules so that parameters are registered before
        # optimizer creation while inferring in_channels on first use.
        self.consistent_est = torch.nn.LazyConv1d(self.proj_dim, kernel_size=1)
        self.consistent_cand = torch.nn.LazyConv1d(self.proj_dim, kernel_size=1)
        self.specific_est = torch.nn.LazyConv1d(self.proj_dim, kernel_size=1)
        self.specific_cand = torch.nn.LazyConv1d(self.proj_dim, kernel_size=1)

        # CLUB network: maps modality-specific features to parameters of q(y|x)
        # (Gaussian with diagonal covariance).
        # Input features are pooled over time, so the LazyLinear infers the
        # feature dimension automatically.
        hidden = self.proj_dim
        self.club_net = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2 * self.proj_dim),
        )

        # Reconstruction decoders for CRD.
        # We use LazyConv1d; out_channels is inferred from weight shape when
        # first used.
        self.reconstruct_est = torch.nn.LazyConv1d(kernel_size=1, out_channels=None)  # type: ignore[arg-type]
        self.reconstruct_cand = torch.nn.LazyConv1d(kernel_size=1, out_channels=None)  # type: ignore[arg-type]

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"Global average pool over time dimension [B, C, T] -> [B, C].\"\"\"
        return x.mean(dim=-1)

    def _info_nce(self, z_est: torch.Tensor, z_cand: torch.Tensor) -> torch.Tensor:
        \"\"\"
        InfoNCE over batch for cross-modal consistent features.

        Args:
            z_est: [B, D]
            z_cand: [B', D] (we only use first B entries as positives)
        \"\"\"
        b = z_est.shape[0]
        z_cand = z_cand[:b]
        # Normalize for cosine similarity
        z_est = torch.nn.functional.normalize(z_est, dim=-1)
        z_cand = torch.nn.functional.normalize(z_cand, dim=-1)
        logits = torch.matmul(z_est, z_cand.t()) / max(self.tau, 1e-6)
        target = torch.arange(b, device=z_est.device)
        return torch.nn.functional.cross_entropy(logits, target)

    def _club_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        \"\"\"
        CLUB mutual information upper bound between modality-specific features.

        Args:
            x: [B, D]
            y: [B, D]
        Returns:
            Estimated MI (scalar). We *minimize* this term.
        \"\"\"
        b = x.shape[0]
        y = y[:b]

        out = self.club_net(x)
        mu, log_var = out.chunk(2, dim=-1)

        # Positive pair log-likelihood E_{p(x,y)}[log q(y|x)]
        positive = -0.5 * (
            ((y - mu) ** 2) / log_var.exp() + log_var + math.log(2 * math.pi)
        )
        positive = positive.sum(dim=-1).mean()

        # Negative pair log-likelihood E_{p(x)p(y)}[log q(y|x)]
        idx = torch.randperm(b, device=x.device)
        y_shuffled = y[idx]
        negative = -0.5 * (
            ((y_shuffled - mu) ** 2) / log_var.exp() + log_var + math.log(2 * math.pi)
        )
        negative = negative.sum(dim=-1).mean()

        mi_est = positive - negative
        return mi_est

    def forward(self, estimate, candidate, mask=None):
        \"\"\"
        Args:
            estimate: [B, C, T] predicted features from MEG.
            candidate: [B', C, T] audio-side features (first B are positives).
            mask: [B, 1, T] or broadcastable boolean mask for valid positions.
        \"\"\"
        # We follow the same assumptions as ClipLoss regarding the mask: for the
        # Gwilliams zero-shot setup with Wav2Vec features, the mask is typically
        # all-True or trimmed at the dataset level.
        if mask is not None:
            assert mask.all(), \"AADNLoss does not support masked loss yet.\"
        assert estimate.size(0) <= candidate.size(0), \"need at least as many targets as estimates\"

        # 1) Base CLIP loss on original features (MACF baseline).
        base_mask = torch.ones_like(estimate[:, :1, :], dtype=torch.bool, device=estimate.device)
        clip_loss = self.clip(estimate, candidate, mask=base_mask)

        # 2) Cross-modal consistent information (IMM via InfoNCE).
        est_trim, cand_trim = self.clip.trim_samples(estimate, candidate)

        z_est_c = self.consistent_est(est_trim)
        z_cand_c = self.consistent_cand(cand_trim)
        z_est_c_vec = self._pool_time(z_est_c)
        z_cand_c_vec = self._pool_time(z_cand_c)
        imm_loss = self._info_nce(z_est_c_vec, z_cand_c_vec)

        # 3) Modality-specific features (CMM via CLUB).
        z_est_s = self.specific_est(est_trim)
        z_cand_s = self.specific_cand(cand_trim)
        z_est_s_vec = self._pool_time(z_est_s)
        z_cand_s_vec = self._pool_time(z_cand_s)
        club_mi = self._club_mi(z_est_s_vec, z_cand_s_vec)

        # 4) Cross-feature Reconstruction Decoder (CRD).
        # We reconstruct the original (trimmed) features from the concatenation
        # of consistent and specific parts for each modality.
        est_concat = torch.cat([z_est_c, z_est_s], dim=1)
        cand_concat = torch.cat([z_cand_c, z_cand_s], dim=1)

        est_recon = self.reconstruct_est(est_concat)
        cand_recon = self.reconstruct_cand(cand_concat)

        est_target = est_trim
        cand_target = cand_trim
        recon_loss = torch.nn.functional.mse_loss(est_recon, est_target) + torch.nn.functional.mse_loss(
            cand_recon, cand_target
        )

        total = clip_loss
        total = total + self.mi_weight * imm_loss
        total = total + self.club_weight * club_mi
        total = total + self.recon_weight * recon_loss
        return total
