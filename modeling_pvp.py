from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

try:
    import transformers as _transformers_module
    from transformers import AutoConfig, AutoModel
    try:
        from transformers import AutoModelForImageTextToText
    except Exception:  # pragma: no cover - optional API across transformers versions
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except Exception:  # pragma: no cover - optional API across transformers versions
        AutoModelForVision2Seq = None
    try:
        from transformers import AutoModelForCausalLM
    except Exception:  # pragma: no cover - optional API across transformers versions
        AutoModelForCausalLM = None
except Exception as exc:  # pragma: no cover - optional dependency guard
    _transformers_module = None
    AutoConfig = None
    AutoModel = None
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover - optional dependency guard
    CausalLMOutputWithPast = None

FastLanguageModel = None
_UNSLOTH_IMPORT_ERROR: Optional[Exception] = None


def _require_transformers() -> None:
    if _transformers_module is None or AutoModel is None or AutoConfig is None:
        raise ImportError(
            "transformers is required but not installed."
        ) from _TRANSFORMERS_IMPORT_ERROR


def _require_unsloth() -> None:
    global FastLanguageModel, _UNSLOTH_IMPORT_ERROR
    if FastLanguageModel is None:
        try:
            from unsloth import FastLanguageModel as _FastLanguageModel
            FastLanguageModel = _FastLanguageModel
        except Exception as exc:  # pragma: no cover - optional dependency guard
            _UNSLOTH_IMPORT_ERROR = exc
    if FastLanguageModel is None:
        raise ImportError(
            "unsloth is required but not installed."
        ) from _UNSLOTH_IMPORT_ERROR


def _get_hidden_size(config: Any, fallback: Optional[int] = None) -> int:
    candidate_names = (
        "hidden_size",
        "vision_hidden_size",
        "mm_hidden_size",
        "embed_dim",
        "d_model",
    )

    for name in candidate_names:
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, int) and value > 0:
                return value

    nested_attrs = (
        "vision_config",
        "text_config",
        "llm_config",
        "language_config",
        "model_config",
        "thinker_config",
        "visual_config",
    )
    for attr in nested_attrs:
        nested = getattr(config, attr, None)
        if nested is None:
            continue
        try:
            return _get_hidden_size(nested, fallback=None)
        except ValueError:
            continue

    if hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
        if isinstance(cfg_dict, dict):
            stack: List[dict] = [cfg_dict]
            while stack:
                node = stack.pop()
                for k, v in node.items():
                    if isinstance(v, dict):
                        stack.append(v)
                        continue
                    if k in candidate_names and isinstance(v, int) and v > 0:
                        return v

    if fallback is None:
        raise ValueError("Unable to infer hidden size from config; provide fallback.")
    return fallback


class VisionTower(nn.Module):
    """Frozen Qwen2.5-Omni vision encoder."""

    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        _require_transformers()
        self.model_name = model_name
        self.model = self._load_model(
            model_name=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._stabilize_omni_visual()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        try:
            self.hidden_size = _get_hidden_size(getattr(self.model, "config", object()), fallback=None)
        except ValueError:
            self.hidden_size = None

    def _stabilize_omni_visual(self) -> None:
        """Work around inconsistent Omni vision configs where fullatt indexes are missing."""
        for module in self.model.modules():
            if hasattr(module, "fullatt_block_indexes"):
                indexes = getattr(module, "fullatt_block_indexes", None)
                if indexes is None:
                    setattr(module, "fullatt_block_indexes", [])

    def _load_model(
        self,
        model_name: str,
        torch_dtype: torch.dtype,
        device_map: Optional[str],
        trust_remote_code: bool,
    ) -> nn.Module:
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model_type = str(getattr(config, "model_type", "")).lower()

        candidate_loaders: List[Tuple[str, Any, Dict[str, Any]]] = []

        if model_type == "qwen2_5_omni":
            thinker_cls = getattr(_transformers_module, "Qwen2_5OmniThinkerForConditionalGeneration", None)
            omni_cls = getattr(_transformers_module, "Qwen2_5OmniForConditionalGeneration", None)
            if thinker_cls is not None:
                candidate_loaders.append(("Qwen2_5OmniThinkerForConditionalGeneration", thinker_cls, {}))
            if omni_cls is not None:
                candidate_loaders.append(
                    (
                        "Qwen2_5OmniForConditionalGeneration",
                        omni_cls,
                        {"enable_audio_output": False},
                    )
                )

        if AutoModelForImageTextToText is not None:
            candidate_loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText, {}))
        if AutoModelForVision2Seq is not None:
            candidate_loaders.append(("AutoModelForVision2Seq", AutoModelForVision2Seq, {}))
        if AutoModelForCausalLM is not None:
            candidate_loaders.append(("AutoModelForCausalLM", AutoModelForCausalLM, {}))
        candidate_loaders.append(("AutoModel", AutoModel, {}))

        errors: List[str] = []
        for loader_name, loader_cls, extra_kwargs in candidate_loaders:
            try:
                model = loader_cls.from_pretrained(
                    model_name,
                    **load_kwargs,
                    **extra_kwargs,
                )
                self.loader_name = loader_name
                return model
            except Exception as exc:
                errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")

        joined = "\n".join(errors)
        raise ValueError(
            f"Failed to load vision model '{model_name}'. Tried loaders:\n{joined}"
        )

    @staticmethod
    def _as_tensor(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "last_hidden_state"):
            value = getattr(features, "last_hidden_state")
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(features, (tuple, list)) and len(features) > 0:
            head = features[0]
            if isinstance(head, torch.Tensor):
                return head
            if hasattr(head, "last_hidden_state"):
                value = getattr(head, "last_hidden_state")
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError("Vision tower output is not a tensor-like structure.")

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode images or video frames.

        Args:
            pixel_values: Tensor with shape [B, C, H, W] or [B, T, C, H, W].
        Returns:
            Vision features with shape [B, N, D] or [B, T, N, D].
        """
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if hasattr(self.model, "encode_images"):
            features = self.model.encode_images(pixel_values)
        elif hasattr(self.model, "get_image_features"):
            call_kwargs: Dict[str, Any] = {}
            if image_grid_thw is not None:
                call_kwargs["image_grid_thw"] = image_grid_thw
            features = self.model.get_image_features(pixel_values, **call_kwargs)
        elif hasattr(self.model, "thinker") and hasattr(self.model.thinker, "get_image_features"):
            call_kwargs = {}
            if image_grid_thw is not None:
                call_kwargs["image_grid_thw"] = image_grid_thw
            features = self.model.thinker.get_image_features(pixel_values, **call_kwargs)
        else:
            call_kwargs = {}
            if image_grid_thw is not None:
                call_kwargs["image_grid_thw"] = image_grid_thw
            if video_grid_thw is not None:
                call_kwargs["video_grid_thw"] = video_grid_thw
            outputs = self.model(pixel_values=pixel_values, return_dict=True, **call_kwargs)
            features = outputs
        return self._as_tensor(features)


class PrecomputedVisionTower(nn.Module):
    """No-op vision tower for precomputed-vision training.

    This module avoids loading a heavy vision backbone when batches already
    contain `*_vision_features`.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = int(hidden_size)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=self.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # pragma: no cover - guard branch
        raise RuntimeError(
            "VisionTower is disabled (precomputed mode). "
            "Provide style_vision_features/target_vision_features instead of pixel_values."
        )


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(latents)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        latents = latents + attn_out
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class PerceiverResampler(nn.Module):
    """Perceiver Resampler for compressing vision tokens.

    Args:
        dim: Hidden size of vision features.
        depth: Number of Perceiver blocks.
        num_queries: Number of latent queries (style tokens).
    """

    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_queries: int = 16,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_queries, dim) * (dim**-0.5))
        self.layers = nn.ModuleList(
            [
                PerceiverBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resample vision tokens into fixed latent queries.

        Args:
            x: Vision tokens with shape [B, N, D] or [B, T, N, D].
        Returns:
            Latent tokens with shape [B, num_queries, D].
        """
        if x.ndim == 4:
            x = rearrange(x, "b t n d -> b (t n) d")
        elif x.ndim != 3:
            raise ValueError(f"Expected x with 3 or 4 dims, got {x.shape}.")
        latents = rearrange(self.latents, "q d -> 1 q d").expand(x.size(0), -1, -1)
        for layer in self.layers:
            latents = layer(latents, x, context_mask=context_mask)
        return self.norm(latents)


class Projector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_mult <= 1:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim * hidden_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim * hidden_mult, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_llm(
    model_name: str,
    max_seq_length: int,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
    bnb_4bit_use_double_quant: bool = True,
    device_map: Optional[str] = "auto",
) -> Tuple[nn.Module, Any]:
    """Load LLM via Unsloth with NF4 4-bit quantization."""
    _require_unsloth()
    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = torch_dtype
    kwargs: Dict[str, Any] = dict(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
        device_map=device_map,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    except Exception as exc:
        if attn_implementation == "flash_attention_3":
            kwargs["attn_implementation"] = "flash_attention_2"
            model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        else:
            raise exc
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer


def apply_lora(
    model: nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
    max_seq_length: int,
    use_gradient_checkpointing: str = "unsloth",
) -> nn.Module:
    _require_unsloth()
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=3407,
        max_seq_length=max_seq_length,
    )


class PVPModel(nn.Module):
    """Personalized Virality Predictor (PVP) model."""

    def __init__(
        self,
        vision_tower: nn.Module,
        llm: nn.Module,
        tokenizer: Any,
        resampler: PerceiverResampler,
        projector: Projector,
        num_style_tokens: int = 16,
        vision_accepts_video: bool = False,
        fixed_image_size: Optional[Tuple[int, int]] = (448, 448),
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.llm = llm
        self.tokenizer = tokenizer
        self.resampler = resampler
        self.projector = projector
        self.num_style_tokens = num_style_tokens
        self.vision_accepts_video = vision_accepts_video
        self.fixed_image_size = fixed_image_size
        self.config = getattr(llm, "config", None)
        self._align_multimodal_modules_to_llm()

    @classmethod
    def from_pretrained(
        cls,
        vision_model_name: str,
        llm_model_name: str,
        max_seq_length: int = 4096,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
        bnb_4bit_use_double_quant: bool = True,
        device_map: Optional[str] = "auto",
        num_style_tokens: int = 16,
        perceiver_depth: int = 6,
        perceiver_heads: int = 8,
        projector_hidden_mult: int = 4,
        lora_r: Optional[int] = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_use_gradient_checkpointing: str = "unsloth",
        vision_accepts_video: bool = False,
        vision_hidden_size: Optional[int] = None,
        skip_vision_tower: bool = False,
        fixed_image_size: Optional[Tuple[int, int]] = (448, 448),
    ) -> "PVPModel":
        if skip_vision_tower:
            if vision_hidden_size is None:
                raise ValueError(
                    "vision_hidden_size must be set when skip_vision_tower=True. "
                    "It must match precomputed vision feature dim D."
                )
            vision_tower = PrecomputedVisionTower(hidden_size=int(vision_hidden_size))
            vision_hidden = int(vision_hidden_size)
        else:
            vision_tower = VisionTower(
                model_name=vision_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            vision_hidden = _get_hidden_size(
                vision_tower.model.config,
                fallback=vision_hidden_size,
            )
        llm, tokenizer = load_llm(
            model_name=llm_model_name,
            max_seq_length=max_seq_length,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            device_map=device_map,
        )
        if lora_r is not None and lora_r > 0:
            llm = apply_lora(
                model=llm,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                max_seq_length=max_seq_length,
                use_gradient_checkpointing=lora_use_gradient_checkpointing,
            )
        llm_hidden = _get_hidden_size(llm.config)
        resampler = PerceiverResampler(
            dim=vision_hidden,
            depth=perceiver_depth,
            num_queries=num_style_tokens,
            num_heads=perceiver_heads,
        )
        projector = Projector(
            in_dim=vision_hidden,
            out_dim=llm_hidden,
            hidden_mult=projector_hidden_mult,
        )
        return cls(
            vision_tower=vision_tower,
            llm=llm,
            tokenizer=tokenizer,
            resampler=resampler,
            projector=projector,
            num_style_tokens=num_style_tokens,
            vision_accepts_video=vision_accepts_video,
            fixed_image_size=fixed_image_size,
        )

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.llm, "get_input_embeddings"):
            return self.llm.get_input_embeddings()
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "get_input_embeddings"):
            return self.llm.model.get_input_embeddings()
        raise AttributeError("LLM does not expose input embeddings.")

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings()(input_ids)

    def _align_multimodal_modules_to_llm(self) -> None:
        embed_module = self.get_input_embeddings()
        embed_params = list(embed_module.parameters())
        if not embed_params:
            return
        target_param = embed_params[0]
        self.resampler.to(device=target_param.device, dtype=target_param.dtype)
        self.projector.to(device=target_param.device, dtype=target_param.dtype)

    def _validate_fixed_image_size(self, pixel_values: torch.Tensor, name: str) -> None:
        if self.fixed_image_size is None:
            return
        if pixel_values.ndim < 4:
            raise ValueError(f"{name} must have at least 4 dims, got {pixel_values.shape}.")
        h, w = pixel_values.shape[-2], pixel_values.shape[-1]
        expected_h, expected_w = self.fixed_image_size
        if (h, w) != (expected_h, expected_w):
            raise ValueError(
                f"{name} has resolution {(h, w)}. "
                f"Expected fixed resolution {(expected_h, expected_w)}. "
                "Use deterministic preprocessing resize to avoid variable vision token lengths."
            )

    def _to_causallm_output(
        self,
        outputs: Any,
        labels_provided: bool,
    ) -> Any:
        if hasattr(outputs, "logits"):
            return outputs
        if not isinstance(outputs, (tuple, list)):
            raise TypeError("LLM output must provide `logits` for DPO compatibility.")
        if CausalLMOutputWithPast is None:
            raise TypeError(
                "transformers.modeling_outputs.CausalLMOutputWithPast is unavailable. "
                "Enable return_dict=True in LLM or install compatible transformers."
            )
        if labels_provided:
            if len(outputs) < 2:
                raise TypeError("Expected tuple output with at least (loss, logits).")
            loss = outputs[0]
            logits = outputs[1]
            past_key_values = outputs[2] if len(outputs) > 2 else None
            hidden_states = outputs[3] if len(outputs) > 3 else None
            attentions = outputs[4] if len(outputs) > 4 else None
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        logits = outputs[0] if len(outputs) > 0 else None
        if logits is None:
            raise TypeError("Expected tuple output with logits at index 0.")
        past_key_values = outputs[1] if len(outputs) > 1 else None
        hidden_states = outputs[2] if len(outputs) > 2 else None
        attentions = outputs[3] if len(outputs) > 3 else None
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def _encode_frames(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._validate_fixed_image_size(pixel_values, "pixel_values")
        with torch.no_grad():
            if pixel_values.ndim == 5 and not self.vision_accepts_video:
                b, t, c, h, w = pixel_values.shape
                flat = rearrange(pixel_values, "b t c h w -> (b t) c h w")
                feats = self.vision_tower(flat)
                if feats.ndim == 3:
                    feats = rearrange(feats, "(b t) n d -> b (t n) d", b=b, t=t)
                elif feats.ndim == 4:
                    feats = rearrange(feats, "(b t) s n d -> b (t s n) d", b=b, t=t)
                else:
                    raise ValueError(f"Unexpected vision features shape: {feats.shape}")
            else:
                feats = self.vision_tower(pixel_values)
                if feats.ndim == 4:
                    feats = rearrange(feats, "b t n d -> b (t n) d")
                elif feats.ndim != 3:
                    raise ValueError(f"Unexpected vision features shape: {feats.shape}")
        return feats.detach()

    def _encode_style(self, style_pixel_values: torch.Tensor) -> torch.Tensor:
        if style_pixel_values.ndim == 6:
            self._validate_fixed_image_size(style_pixel_values, "style_pixel_values")
            b, s, t, c, h, w = style_pixel_values.shape
            flat = rearrange(style_pixel_values, "b s t c h w -> (b s) t c h w")
            feats = self._encode_frames(flat)
            if feats.ndim == 3:
                feats = rearrange(feats, "(b s) n d -> b s n d", b=b, s=s)
            else:
                raise ValueError(f"Unexpected style features shape: {feats.shape}")
            return feats
        raise ValueError(
            "style_pixel_values must have shape [B, S, T, C, H, W]."
        )

    def _encode_target(self, target_pixel_values: torch.Tensor) -> torch.Tensor:
        if target_pixel_values.ndim == 6 and target_pixel_values.size(1) == 1:
            target_pixel_values = target_pixel_values[:, 0]
        self._validate_fixed_image_size(target_pixel_values, "target_pixel_values")
        if target_pixel_values.ndim in (4, 5):
            return self._encode_frames(target_pixel_values)
        raise ValueError(
            "target_pixel_values must have shape [B, T, C, H, W] or [B, C, H, W]."
        )

    def _prepare_style_features(self, style_vision_features: torch.Tensor) -> torch.Tensor:
        if style_vision_features.ndim == 4:
            return rearrange(style_vision_features, "b s n d -> b (s n) d")
        if style_vision_features.ndim == 3:
            return style_vision_features
        raise ValueError(
            "style_vision_features must have shape [B, S, N, D] or [B, N, D]."
        )

    def _prepare_target_features(self, target_vision_features: torch.Tensor) -> torch.Tensor:
        if target_vision_features.ndim == 4:
            return rearrange(target_vision_features, "b t n d -> b (t n) d")
        if target_vision_features.ndim == 3:
            return target_vision_features
        raise ValueError(
            "target_vision_features must have shape [B, N, D] or [B, T, N, D]."
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        style_pixel_values: Optional[torch.Tensor] = None,
        target_pixel_values: Optional[torch.Tensor] = None,
        style_vision_features: Optional[torch.Tensor] = None,
        target_vision_features: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass compatible with TRL DPOTrainer.

        Args:
            style_pixel_values: Style videos, shape [B, S, T, C, H, W], S=20.
            target_pixel_values: Target video, shape [B, T, C, H, W] or [B, 1, T, C, H, W].
            style_vision_features: Precomputed style features, shape [B, S, N, D] or [B, N, D].
            target_vision_features: Precomputed target features, shape [B, N, D] or [B, T, N, D].
            input_ids: Text tokens, shape [B, L].
            attention_mask: Mask for tokens, shape [B, L] or [B, L+M].
            labels: Labels for LM loss, shape [B, L] or [B, L+M].
        Returns:
            LLM outputs (CausalLMOutputWithPast compatible).
        """
        style_alias = kwargs.pop("concatenated_style_pixel_values", None)
        target_alias = kwargs.pop("concatenated_target_pixel_values", None)
        style_feat_alias = kwargs.pop("concatenated_style_vision_features", None)
        target_feat_alias = kwargs.pop("concatenated_target_vision_features", None)
        chosen_target = kwargs.pop("chosen_target_pixel_values", None)
        rejected_target = kwargs.pop("rejected_target_pixel_values", None)
        chosen_target_feats = kwargs.pop("chosen_target_vision_features", None)
        rejected_target_feats = kwargs.pop("rejected_target_vision_features", None)
        pixel_values_alias = kwargs.pop("pixel_values", None)

        if style_pixel_values is None and style_alias is not None:
            style_pixel_values = style_alias
        if style_vision_features is None and style_feat_alias is not None:
            style_vision_features = style_feat_alias

        if target_vision_features is None:
            if target_feat_alias is not None:
                target_vision_features = target_feat_alias
            elif chosen_target_feats is not None and rejected_target_feats is not None:
                target_vision_features = torch.cat([chosen_target_feats, rejected_target_feats], dim=0)
            elif chosen_target_feats is not None:
                target_vision_features = chosen_target_feats
            elif rejected_target_feats is not None:
                target_vision_features = rejected_target_feats

        if target_pixel_values is None and target_vision_features is None:
            if target_alias is not None:
                target_pixel_values = target_alias
            elif pixel_values_alias is not None:
                target_pixel_values = pixel_values_alias
            elif chosen_target is not None and rejected_target is not None:
                target_pixel_values = torch.cat([chosen_target, rejected_target], dim=0)
            elif chosen_target is not None:
                target_pixel_values = chosen_target
            elif rejected_target is not None:
                target_pixel_values = rejected_target

        batch_size_hint: Optional[int] = None
        if input_ids is not None:
            batch_size_hint = input_ids.size(0)
        elif inputs_embeds is not None:
            batch_size_hint = inputs_embeds.size(0)
        elif target_vision_features is not None:
            batch_size_hint = target_vision_features.size(0)
        elif target_pixel_values is not None:
            batch_size_hint = target_pixel_values.size(0)
        elif style_vision_features is not None:
            batch_size_hint = style_vision_features.size(0)

        if style_pixel_values is not None and batch_size_hint is not None:
            style_batch = style_pixel_values.size(0)
            if style_batch != batch_size_hint:
                if batch_size_hint % style_batch == 0:
                    repeats = batch_size_hint // style_batch
                    style_pixel_values = torch.cat([style_pixel_values] * repeats, dim=0)
                else:
                    raise ValueError(
                        f"style_pixel_values batch {style_batch} is incompatible with "
                        f"text/target batch {batch_size_hint}."
                    )

        if style_vision_features is not None and batch_size_hint is not None:
            style_feat_batch = style_vision_features.size(0)
            if style_feat_batch != batch_size_hint:
                if batch_size_hint % style_feat_batch == 0:
                    repeats = batch_size_hint // style_feat_batch
                    style_vision_features = torch.cat([style_vision_features] * repeats, dim=0)
                else:
                    raise ValueError(
                        f"style_vision_features batch {style_feat_batch} is incompatible with "
                        f"text/target batch {batch_size_hint}."
                    )

        if target_pixel_values is not None and batch_size_hint is not None:
            target_batch = target_pixel_values.size(0)
            if target_batch != batch_size_hint:
                raise ValueError(
                    f"target_pixel_values batch {target_batch} is incompatible with "
                    f"text batch {batch_size_hint}."
                )

        if target_vision_features is not None and batch_size_hint is not None:
            target_feat_batch = target_vision_features.size(0)
            if target_feat_batch != batch_size_hint:
                raise ValueError(
                    f"target_vision_features batch {target_feat_batch} is incompatible with "
                    f"text batch {batch_size_hint}."
                )

        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("return_dict", True)

        inputs_embeds_provided = inputs_embeds is not None
        labels_provided = labels is not None
        mm_embeds: Optional[torch.Tensor] = None
        if style_vision_features is not None:
            style_context = self._prepare_style_features(style_vision_features)
            style_context = style_context.to(
                device=self.resampler.latents.device,
                dtype=self.resampler.latents.dtype,
            )
            style_tokens = self.resampler(style_context)
            style_embeds = self.projector(style_tokens)
            mm_embeds = style_embeds
        elif style_pixel_values is not None:
            style_feats = self._encode_style(style_pixel_values)
            style_context = rearrange(style_feats, "b s n d -> b (s n) d")
            style_context = style_context.to(
                device=self.resampler.latents.device,
                dtype=self.resampler.latents.dtype,
            )
            style_tokens = self.resampler(style_context)
            style_embeds = self.projector(style_tokens)
            mm_embeds = style_embeds

        if target_vision_features is not None:
            target_feats = self._prepare_target_features(target_vision_features)
            projector_param = next(self.projector.parameters())
            target_feats = target_feats.to(
                device=projector_param.device,
                dtype=projector_param.dtype,
            )
            target_embeds = self.projector(target_feats)
            if mm_embeds is None:
                mm_embeds = target_embeds
            else:
                mm_embeds = torch.cat([mm_embeds, target_embeds], dim=1)
        elif target_pixel_values is not None:
            target_feats = self._encode_target(target_pixel_values)
            projector_param = next(self.projector.parameters())
            target_feats = target_feats.to(
                device=projector_param.device,
                dtype=projector_param.dtype,
            )
            target_embeds = self.projector(target_feats)
            if mm_embeds is None:
                mm_embeds = target_embeds
            else:
                mm_embeds = torch.cat([mm_embeds, target_embeds], dim=1)

        if mm_embeds is None and inputs_embeds is None and input_ids is None:
            raise ValueError("Provide input_ids/inputs_embeds or vision inputs.")

        if mm_embeds is None and not inputs_embeds_provided:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
            return self._to_causallm_output(outputs, labels_provided=labels_provided)

        text_embeds = inputs_embeds
        if text_embeds is None and input_ids is not None:
            text_embeds = self._embed_tokens(input_ids)

        if mm_embeds is not None:
            if text_embeds is not None:
                inputs_embeds = torch.cat([mm_embeds, text_embeds], dim=1)
                text_len = text_embeds.size(1)
            else:
                inputs_embeds = mm_embeds
                text_len = 0
            mm_len = mm_embeds.size(1)
        else:
            inputs_embeds = text_embeds
            text_len = text_embeds.size(1) if text_embeds is not None else 0
            mm_len = 0

        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.size(0), text_len if mm_len > 0 else inputs_embeds.size(1)),
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        if mm_len > 0 and attention_mask.size(1) == text_len:
            mm_mask = torch.ones(
                (attention_mask.size(0), mm_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mm_mask, attention_mask], dim=1)
        elif mm_len > 0 and attention_mask.size(1) != (text_len + mm_len):
            raise ValueError(
                f"attention_mask length {attention_mask.size(1)} is invalid for "
                f"text_len={text_len}, mm_len={mm_len}. Expected {text_len} or {text_len + mm_len}."
            )

        if labels is not None and mm_len > 0 and labels.size(1) == text_len:
            mm_labels = torch.full(
                (labels.size(0), mm_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([mm_labels, labels], dim=1)
        elif labels is not None and mm_len > 0 and labels.size(1) != (text_len + mm_len):
            raise ValueError(
                f"labels length {labels.size(1)} is invalid for text_len={text_len}, mm_len={mm_len}. "
                f"Expected {text_len} (text-only labels for DPO) or {text_len + mm_len}."
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return self._to_causallm_output(outputs, labels_provided=labels_provided)
