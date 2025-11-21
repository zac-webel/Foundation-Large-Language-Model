"""Data streaming utilities for pretraining the Pathogen AI encoder."""
from __future__ import annotations

import random
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from datasets import IterableDataset, load_dataset

from tokenizer import CodonTokenizer

_ALLOWED_BASES = set(CodonTokenizer.BASES)
_BASE_CHOICES: Tuple[str, ...] = CodonTokenizer.BASES


def _strip_whitespace(sequence: str) -> str:
    """Remove whitespace and normalise to uppercase."""
    return "".join(sequence.upper().split())


def _prepare_sequence(sequence: str, max_base_length: int) -> Optional[str]:
    """Clean and truncate a sequence so it can be tokenised safely."""
    cleaned = _strip_whitespace(sequence)
    if not cleaned or set(cleaned) - _ALLOWED_BASES:
        return None

    usable_length = min(len(cleaned), max_base_length)
    usable_length -= usable_length % 3
    if usable_length <= 0:
        return None

    return cleaned[:usable_length]


def _apply_random_noise(sequence: str, rng: random.Random) -> str:
    """Randomly substitute 0-30% of nucleotides with uniform bases."""
    substitution_rate = rng.uniform(0.0, 0.3)
    noisy = []
    for base in sequence:
        if rng.random() < substitution_rate:
            noisy.append(rng.choice(_BASE_CHOICES))
        else:
            noisy.append(base)
    return "".join(noisy)


def _attention_mask(token_ids: Sequence[int], pad_token_id: int) -> Sequence[int]:
    """Return a binary mask indicating which positions are not padding."""
    return [0 if token_id == pad_token_id else 1 for token_id in token_ids]


class SequenceDenoisingGenerator:
    """Python generator that yields noisy/original codon batches for Keras."""

    def __init__(
        self,
        *,
        batch_size: int = 16,
        dataset_name: str = "InstaDeepAI/multi_species_genomes",
        split: str = "train",
        shuffle_buffer: int = 10_000,
        seed: Optional[int] = None,
        tokenizer: Optional[CodonTokenizer] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.tokenizer = tokenizer or CodonTokenizer()
        self._max_base_length = self.tokenizer.CONTEXT_LENGTH * 3
        self._rng = random.Random(seed)

        # Hugging Face datasets >=3.0 requires explicit opt-in for Python dataset scripts.
        try:
            dataset: IterableDataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code,
            )
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" in str(exc):
                raise RuntimeError(
                    "The Hugging Face dataset requires the `datasets` package < 3.0.0. "
                    "Please install a compatible version, e.g. `pip install \"datasets<3.0.0\"`."
                ) from exc
            raise
        if shuffle_buffer:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        self._dataset = dataset
        self._iterator: Iterator[Dict[str, str]] = iter(self._dataset)

    def __iter__(self) -> "SequenceDenoisingGenerator":
        return self

    def __next__(self):  # type: ignore[override]
        while True:
            inputs = []
            labels = []
            masks = []

            while len(inputs) < self.batch_size:
                example = self._next_valid_example()
                if example is None:
                    continue

                original = example
                noisy = _apply_random_noise(original, self._rng)

                input_ids = self.tokenizer.encode(noisy)
                label_ids = self.tokenizer.encode(original)
                mask = _attention_mask(label_ids, self.tokenizer.PAD_TOKEN_ID)

                inputs.append(input_ids)
                labels.append(label_ids)
                masks.append(mask)

            batch_inputs = {
                "token_ids": np.asarray(inputs, dtype=np.int32),
                # Use int32 to keep compatibility with TensorFlow expectations.
                "padding_mask": np.asarray(masks, dtype=np.int32),
            }
            batch_labels = np.asarray(labels, dtype=np.int32)

            if self._contains_nan(batch_inputs, batch_labels):
                print("Skipping batch containing NaN values in tokenized inputs.")
                continue

            return batch_inputs, batch_labels

    def _contains_nan(self, batch_inputs: Dict[str, np.ndarray], batch_labels: np.ndarray) -> bool:
        """Return True when any tensor in the batch contains NaN values."""
        token_ids = batch_inputs["token_ids"].astype(np.float32)
        padding_mask = batch_inputs["padding_mask"].astype(np.float32)
        labels = batch_labels.astype(np.float32)
        return bool(
            np.isnan(token_ids).any()
            or np.isnan(padding_mask).any()
            or np.isnan(labels).any()
        )

    def _next_valid_example(self) -> Optional[str]:
        """Fetch the next sequence that meets base and length constraints."""
        while True:
            try:
                record = next(self._iterator)
            except StopIteration:
                self._iterator = iter(self._dataset)
                continue

            sequence = record.get("sequence") if isinstance(record, dict) else None
            if not sequence:
                continue

            prepared = _prepare_sequence(sequence, self._max_base_length)
            if prepared is None:
                continue
            return prepared


def create_denoising_generator(**kwargs) -> SequenceDenoisingGenerator:
    """Factory for `SequenceDenoisingGenerator` with convenience keyword args."""
    return SequenceDenoisingGenerator(**kwargs)


def data_generator(**kwargs) -> Iterator:
    """Backward-compatible helper that exposes the generator protocol."""
    return iter(SequenceDenoisingGenerator(**kwargs))


__all__ = ["SequenceDenoisingGenerator", "create_denoising_generator", "data_generator"]
