"""
Viral Optimizer (Block C) — MLX port.

Uses Projected Gradient Ascent on video embeddings to find the direction
of change that maximises the predicted viral score, while staying
semantically close to the original embeddings.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class OptimizationResult:
    """Result of the optimisation loop."""
    delta_z: mx.array               # [Batch, Time, Dim]
    final_score: mx.array           # [Batch, 1]
    score_improvement: float        # final - initial mean score
    per_frame_delta_norm: mx.array  # [Batch, Time]


class ViralOptimizer:
    """
    Projected Gradient Ascent optimiser for video embeddings.

    Supports 3-D optimisation:
        Input:  [Batch, Time, Dim]  — sequence of video-token embeddings
        Output: delta_z with per-frame norms for timecode analysis.
    """

    def __init__(
        self,
        predictor_model: nn.Module,
        steps: int = 50,
        lr: float = 0.05,
        lambda_cos: float = 10.0,
        lambda_smooth: float = 2.0,
        pooling_mode: Literal["mean", "last"] = "mean",
    ):
        self.predictor = predictor_model
        self.steps = steps
        self.lr = lr
        self.lambda_cos = lambda_cos
        self.lambda_smooth = lambda_smooth
        self.pooling_mode = pooling_mode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pool_sequence(
        self,
        z: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """[B, T, D] -> [B, D]."""
        if z.ndim == 2:
            return z

        if self.pooling_mode == "mean":
            if mask is not None:
                m = mx.expand_dims(mask, axis=-1)  # [B, T, 1]
                return mx.sum(z * m, axis=1) / mx.maximum(
                    mx.sum(m, axis=1), mx.array(1.0)
                )
            return mx.mean(z, axis=1)

        if self.pooling_mode == "last":
            if mask is not None:
                lengths = mx.sum(mask, axis=1).astype(mx.int32) - 1
                batch_size = z.shape[0]
                return z[mx.arange(batch_size), lengths]
            return z[:, -1, :]

        raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    @staticmethod
    def _cosine_similarity(a: mx.array, b: mx.array, axis: int = -1) -> mx.array:
        """Element-wise cosine similarity along *axis*."""
        dot = mx.sum(a * b, axis=axis)
        norm_a = mx.sqrt(mx.sum(a * a, axis=axis) + 1e-8)
        norm_b = mx.sqrt(mx.sum(b * b, axis=axis) + 1e-8)
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        z_orig: mx.array,
        mask: Optional[mx.array] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run Projected Gradient Ascent on *z_orig*.

        Parameters
        ----------
        z_orig : [B, T, D] or [B, D]
        mask   : [B, T] optional padding mask
        verbose: print progress every 10 steps

        Returns
        -------
        OptimizationResult
        """
        # Ensure 3-D
        if z_orig.ndim == 2:
            z_orig = mx.expand_dims(z_orig, axis=1)

        batch_size, time_steps, dim = z_orig.shape

        # Original norms for PGD projection
        orig_norm = mx.sqrt(mx.sum(z_orig * z_orig, axis=-1, keepdims=True) + 1e-8)

        # Initial score
        z_pooled_init = self._pool_sequence(z_orig, mask)
        initial_score = self.predictor(z_pooled_init)

        if verbose:
            print(f"[Optimizer] Shape: {z_orig.shape}, "
                  f"Avg norm: {mx.mean(orig_norm).item():.2f}")
            print(f"[Optimizer] Initial score: {mx.mean(initial_score).item():.3f}")

        # Working copy
        z_opt = mx.array(z_orig)

        for i in range(self.steps):
            # Build the total-loss function with respect to z
            def total_loss_fn(z):
                z_pooled = self._pool_sequence(z, mask)
                score = self.predictor(z_pooled)
                loss_viral = -mx.mean(score)

                # Cosine penalty
                cos_sim = self._cosine_similarity(z, z_orig, axis=-1)
                if mask is not None:
                    cos_dist = (1.0 - cos_sim) * mask
                    loss_cos = mx.sum(cos_dist) / mx.maximum(
                        mx.sum(mask), mx.array(1.0)
                    )
                else:
                    loss_cos = mx.mean(1.0 - cos_sim)

                # Smoothness penalty
                if time_steps > 1:
                    diff = z[:, 1:, :] - z[:, :-1, :]
                    diff_norm = mx.sqrt(mx.sum(diff * diff, axis=-1) + 1e-8)
                    if mask is not None:
                        t_mask = mask[:, 1:] * mask[:, :-1]
                        loss_smooth = mx.sum(diff_norm * t_mask) / mx.maximum(
                            mx.sum(t_mask), mx.array(1.0)
                        )
                    else:
                        loss_smooth = mx.mean(diff_norm)
                else:
                    loss_smooth = mx.array(0.0)

                return (loss_viral
                        + self.lambda_cos * loss_cos
                        + self.lambda_smooth * loss_smooth)

            grad_fn = mx.grad(total_loss_fn)
            grad = grad_fn(z_opt)

            # Adam-like step (simple SGD for clarity; could add momentum)
            z_opt = z_opt - self.lr * grad

            # PGD: project back to original norm sphere
            new_norm = mx.sqrt(
                mx.sum(z_opt * z_opt, axis=-1, keepdims=True) + 1e-8
            )
            z_opt = z_opt * (orig_norm / new_norm)
            mx.eval(z_opt)

            if verbose and i % 10 == 0:
                z_pooled = self._pool_sequence(z_opt, mask)
                cur_score = self.predictor(z_pooled)
                cos_sim = self._cosine_similarity(z_opt, z_orig, axis=-1)
                print(f"  Step {i}: score={mx.mean(cur_score).item():.3f}  "
                      f"cos_dist={mx.mean(1.0 - cos_sim).item():.4f}")

        # Final results
        z_pooled_final = self._pool_sequence(z_opt, mask)
        final_score = self.predictor(z_pooled_final)
        delta_z = z_opt - z_orig
        per_frame_delta_norm = mx.sqrt(
            mx.sum(delta_z * delta_z, axis=-1) + 1e-8
        )
        score_improvement = (
            mx.mean(final_score).item() - mx.mean(initial_score).item()
        )

        if verbose:
            print(f"[Optimizer] Final score: {mx.mean(final_score).item():.3f}, "
                  f"improvement: {score_improvement:+.3f}")

        return OptimizationResult(
            delta_z=delta_z,
            final_score=final_score,
            score_improvement=score_improvement,
            per_frame_delta_norm=per_frame_delta_norm,
        )
