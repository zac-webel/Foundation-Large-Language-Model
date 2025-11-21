"""Model definition for the Pathogen AI Encoder Base backbone."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from keras_hub.models import DebertaV3Backbone

# Hard-coded configuration derived from the Pathogen AI Encoder Base design.
BASE_MODEL_CONFIG: Dict[str, Any] = {
    "vocabulary_size": 65,  # PAD token + 64 codons
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    "intermediate_dim": 3072,
    "dropout": 0.1,
    "max_sequence_length": 567,
    "bucket_size": 284,
    "dtype": "float32",
}


def build_backbone(*, weights_path: str | Path | None = None, **overrides: Any) -> DebertaV3Backbone:
    """Instantiate the DeBERTa v3 backbone with optional configuration overrides.

    Args:
        weights_path: Optional filesystem path to a set of model weights. When
            provided, the weights are loaded into the freshly constructed
            backbone before it is returned.
        **overrides: Keyword arguments that override the base model config.
    """
    config = {**BASE_MODEL_CONFIG, **overrides}
    backbone = DebertaV3Backbone(**config)

    if weights_path is not None:
        weight_file = Path(weights_path)
        if not weight_file.exists():
            raise FileNotFoundError(f"Weights file not found: {weight_file}")
        try:
            backbone.load_weights(str(weight_file))
        except Exception as exc:  # pragma: no cover - passthrough for Keras load errors
            raise RuntimeError(f"Failed to load weights from {weight_file}") from exc

    return backbone


def model(*, weights_path: str | Path | None = None, **overrides: Any) -> DebertaV3Backbone:
    """Compatibility wrapper so callers can do `model = model.model()`.

    Args:
        weights_path: Optional filesystem path to a set of model weights.
        **overrides: Keyword arguments that override the base model config.
    """
    return build_backbone(weights_path=weights_path, **overrides)
