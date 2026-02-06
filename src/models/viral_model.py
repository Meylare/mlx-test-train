"""
Viral Predictor Model for MLX.

Loads Qwen2.5-VL via mlx-vlm, applies LoRA to target linear layers,
and adds an MLP regression head to predict viral scores from hidden states.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.utils import load_config
from typing import Optional, Tuple, Union, Any


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

LORA_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def _apply_lora_to_module(
    module: nn.Module,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    lora_scale: Optional[float] = None,
    prefix: str = "",
) -> int:
    """
    Recursively walk *module* and replace every ``nn.Linear`` whose **leaf
    name** is in ``LORA_TARGETS`` with ``nn.LoRALinear``.

    Returns the number of layers replaced.
    """
    if lora_scale is None:
        lora_scale = lora_alpha / lora_r

    count = 0
    for name, child in module.children().items():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and name in LORA_TARGETS:
            lora_layer = nn.LoRALinear.from_linear(
                child,
                r=lora_r,
                scale=lora_scale,
                dropout=lora_dropout,
            )
            # Replace the child in the parent module
            setattr(module, name, lora_layer)
            count += 1
        elif isinstance(child, nn.Module):
            count += _apply_lora_to_module(
                child,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_scale=lora_scale,
                prefix=full_name,
            )
    return count


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter that does **not** belong to a LoRA adapter or
    the ``viral_head`` MLP."""
    model.freeze()
    # Un-freeze LoRA weights
    for name, param in model.named_modules():
        if isinstance(param, nn.LoRALinear):
            param.unfreeze()
    # Un-freeze MLP head
    if hasattr(model, "viral_head"):
        model.viral_head.unfreeze()


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights to *path* (``*.safetensors``)."""
    lora_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.LoRALinear):
            params = module.parameters()
            for pname, pval in _flatten(params, prefix=name):
                if "lora" in pname.lower():
                    lora_weights[pname] = pval
    mx.savez(path, **lora_weights)


def _flatten(tree, prefix=""):
    """Flatten a nested dict coming from ``module.parameters()``."""
    results = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            results.extend(_flatten(v, new_prefix))
    elif isinstance(tree, mx.array):
        results.append((prefix, tree))
    return results


# ---------------------------------------------------------------------------
# MLP Head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Regression head: hidden_size -> 1024 -> 256 -> 1."""

    def __init__(self, hidden_size: int = 3584):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dropout(self.gelu(self.l1(x)))
        x = self.gelu(self.l2(x))
        return self.l3(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ViralPredictorModel(nn.Module):
    """
    Wraps a Qwen2.5-VL backbone loaded via *mlx-vlm*, injects LoRA adapters
    into the language-model projection layers, and attaches an MLP regression
    head that maps the last hidden state to a scalar viral score.

    Usage::

        model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        model = ViralPredictorModel(model_path)
        score = model(input_ids, pixel_values_videos, attention_mask, ...)
    """

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 3584,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # -- Load backbone via mlx-vlm -----------------------------------------
        self.backbone, self.processor = load(model_path)
        self.config = load_config(model_path)

        # Detect hidden_size from config if possible
        cfg = self.config if isinstance(self.config, dict) else {}
        if "hidden_size" in cfg:
            hidden_size = cfg["hidden_size"]
        elif "text_config" in cfg and "hidden_size" in cfg["text_config"]:
            hidden_size = cfg["text_config"]["hidden_size"]

        # -- Apply LoRA ---------------------------------------------------------
        n_replaced = _apply_lora_to_module(
            self.backbone,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        print(f"[ViralPredictorModel] LoRA applied to {n_replaced} layers "
              f"(r={lora_r}, alpha={lora_alpha})")

        # -- MLP Head -----------------------------------------------------------
        self.viral_head = MLPHead(hidden_size)

        # -- Freeze everything except LoRA + MLP Head ---------------------------
        freeze_non_lora(self)
        self._print_trainable_params()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values_videos: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        return_full_sequence: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Returns
        -------
        score : mx.array, shape ``[B, 1]``
            Predicted viral score.
        video_hidden_states : mx.array (only when ``return_full_sequence=True``)
            Hidden states of visual tokens ``[B, N_vis, H]``.
        """
        # --- Get hidden states from backbone ---
        hidden_states = self._get_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # --- Pool: last non-padding token ---
        batch_size = input_ids.shape[0]
        seq_lengths = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1
        h_final = hidden_states[mx.arange(batch_size), seq_lengths]  # [B, H]
        h_final = h_final.astype(mx.float32)

        # --- Regression head ---
        score = self.viral_head(h_final)  # [B, 1]

        if return_full_sequence:
            video_hidden_states = self._extract_video_tokens(
                hidden_states, input_ids, video_grid_thw, image_grid_thw,
            )
            return score, video_hidden_states

        return score

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values_videos: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Run the backbone and return the **last hidden state** before the
        ``lm_head`` projection.

        mlx-vlm Qwen2.5-VL models expose an inner ``model`` (the transformer
        body) and a ``lm_head``.  We call ``model(...)`` directly to get the
        hidden states.
        """
        backbone = self.backbone

        # -- Build inputs_embeds by merging text and vision embeddings --
        # The Qwen2.5-VL model in mlx-vlm has:
        #   backbone.model.embed_tokens   -> text embedding
        #   backbone.vision_tower / visual -> vision encoder
        #   backbone.model(...)           -> transformer body returning hidden states
        #   backbone.lm_head             -> final projection (we skip this)

        # Step 1: get text embeddings
        inputs_embeds = backbone.model.embed_tokens(input_ids)

        # Step 2: get vision features and merge into embeddings
        if pixel_values_videos is not None:
            vision_features = backbone.vision_tower(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            inputs_embeds = self._merge_vision_embeddings(
                input_ids, inputs_embeds, vision_features,
                token_id=151656,  # <|video_pad|>
            )
        elif pixel_values is not None:
            vision_features = backbone.vision_tower(
                pixel_values, grid_thw=image_grid_thw
            )
            inputs_embeds = self._merge_vision_embeddings(
                input_ids, inputs_embeds, vision_features,
                token_id=151655,  # <|image_pad|>
            )

        # Step 3: run through transformer layers
        # mlx-vlm Qwen2.5-VL model.model is the transformer body
        # It expects inputs_embeds and returns hidden states
        hidden_states = inputs_embeds
        mask = None  # Will be created internally or we pass attention_mask

        for layer in backbone.model.layers:
            hidden_states = layer(hidden_states, mask=mask)

        # Final norm
        if hasattr(backbone.model, "norm"):
            hidden_states = backbone.model.norm(hidden_states)

        return hidden_states

    def _merge_vision_embeddings(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        vision_features: mx.array,
        token_id: int,
    ) -> mx.array:
        """Replace placeholder token embeddings with vision encoder outputs."""
        # Find positions of the placeholder tokens
        mask = (input_ids == token_id)

        # vision_features: [N_total_patches, H]
        # We need to scatter them into the right positions
        batch_size, seq_len, hidden = inputs_embeds.shape

        # For each batch element, replace placeholder positions with vision features
        # Simplified for batch_size=1 (training scenario)
        for b in range(batch_size):
            positions = mx.where(mask[b])[0]
            n_vis = positions.shape[0]
            n_feat = vision_features.shape[0] if vision_features.ndim == 2 else vision_features.shape[1]
            n = min(n_vis, n_feat)
            if n > 0:
                feats = vision_features[:n] if vision_features.ndim == 2 else vision_features[b, :n]
                # Scatter vision features into input embeddings
                inputs_embeds = inputs_embeds.at[b, positions[:n]].set(
                    feats.astype(inputs_embeds.dtype)
                )

        return inputs_embeds

    # ------------------------------------------------------------------
    # Video token extraction (for inference optimizer)
    # ------------------------------------------------------------------

    def _extract_video_tokens(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        video_grid_thw: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract hidden states that correspond to visual tokens."""
        VISION_START = 151652
        VISION_END = 151653

        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        tokens_list = []
        max_len = 0

        for b in range(batch_size):
            ids = input_ids[b]
            start_mask = (ids == VISION_START)
            end_mask = (ids == VISION_END)

            start_positions = mx.where(start_mask)[0]
            end_positions = mx.where(end_mask)[0]

            if start_positions.size > 0 and end_positions.size > 0:
                start_idx = int(start_positions[0].item()) + 1
                end_idx = int(end_positions[0].item())
                tokens = hidden_states[b, start_idx:end_idx]
            else:
                # Fallback: take middle tokens
                tokens = hidden_states[b, 1:-1]

            tokens_list.append(tokens.astype(mx.float32))
            max_len = max(max_len, tokens.shape[0])

        # Pad to same length
        result = mx.zeros((batch_size, max_len, hidden_size))
        for b, tokens in enumerate(tokens_list):
            n = tokens.shape[0]
            result = result.at[b, :n].set(tokens)

        return result

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _print_trainable_params(self) -> None:
        """Print the number of trainable vs total parameters."""
        total = 0
        trainable = 0
        for name, param in self._all_params():
            n = param.size
            total += n
            # LoRA params or viral_head params are trainable
            if "lora" in name.lower() or "viral_head" in name:
                trainable += n
        print(f"[ViralPredictorModel] Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / max(total, 1):.2f}%)")

    def _all_params(self, prefix=""):
        """Yield ``(name, mx.array)`` for every parameter in the model."""
        for name, value in self.parameters().items():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(value, mx.array):
                yield full, value
            elif isinstance(value, dict):
                yield from _flatten_params(value, full)


def _flatten_params(tree, prefix=""):
    """Recursively flatten a nested parameter dict."""
    for k, v in tree.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, mx.array):
            yield name, v
        elif isinstance(v, dict):
            yield from _flatten_params(v, name)
