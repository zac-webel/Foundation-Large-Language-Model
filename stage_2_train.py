"""Stage 2 training script fine-tuning the Pathogen AI encoder on influenza data."""
from __future__ import annotations

import math
import os
import random
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import keras
from keras import callbacks, losses, mixed_precision, optimizers
from keras.optimizers.schedules import CosineDecay
import numpy as np
import tensorflow as tf
from datasets import Dataset, load_dataset

from model import build_backbone
from processing import _apply_random_noise, _attention_mask, _prepare_sequence
from tokenizer import CodonTokenizer

# # Enable mixed precision across the training script.
mixed_precision.set_global_policy("mixed_float16")

# Training hyperparameters.
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 128//BATCH_SIZE
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GLOBAL_CLIPNORM = 1.0
COSINE_DECAY_ALPHA = 0.1
WARMUP_RATIO = 0.1
NUM_EPOCHS = 50
DATASET_NAME = ""
DATASET_SPLIT = "train"
DATE_CUTOFF = "2023-01-01"
MAX_VALIDATION_SEQUENCES = 2048
RANDOM_SEED = 1234
STAGE_1_WEIGHTS_PATH = "stage_1_base.weights.h5"
STAGE_2_WEIGHTS_PATH = "stage_2_base.weights.h5"


def build_stage_2_model(tokenizer: CodonTokenizer) -> keras.Model:
    """Attach a softmax classification head to the DeBERTa backbone."""
    backbone = build_backbone()
    vocab_size = len(tokenizer.vocabulary(print_output=False))

    sequence_output = backbone.output
    logits = keras.layers.Dense(
        vocab_size,
        dtype="float32",
        name="stage_2_classification_head",
    )(sequence_output)

    return keras.Model(inputs=backbone.input, outputs=logits, name="pathogen_encoder_stage_2")


def _is_train_example(example: Dict[str, str], *, cutoff: str) -> bool:
    """Return True when the record date is on/before the configured cutoff."""
    date_value = example.get("date")
    if not date_value:
        return False
    return str(date_value) <= cutoff


def _is_validation_example(example: Dict[str, str], *, cutoff: str) -> bool:
    """Return True when the record date is after the configured cutoff."""
    date_value = example.get("date")
    if not date_value:
        return False
    return str(date_value) > cutoff


def _normalize_date_value(raw_value: Any) -> str | None:
    """Normalise dates to ISO8601 strings where possible."""
    if raw_value is None:
        return None
    if isinstance(raw_value, (datetime, date)):
        return raw_value.isoformat()

    value = str(raw_value).strip()
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value


def _extract_sequence_date_pairs(record: Dict[str, Any]) -> List[Tuple[str, str | None]]:
    """Extract (sequence, date) pairs from a dataset record supporting multiple schemas."""
    candidate_fields: tuple[tuple[str, str | None], ...] = (
        ("seq", "date"),
        ("seq", "collection_date"),
        ("sequence", "date"),
        ("sequence", "collection_date"),
        ("seq_1", "date_1"),
        ("seq_2", "date_2"),
        ("seq_1", "collection_date_1"),
        ("seq_2", "collection_date_2"),
    )

    pairs: List[Tuple[str, str | None]] = []
    seen: set[Tuple[str, str | None]] = set()

    for seq_field, date_field in candidate_fields:
        sequence_value = record.get(seq_field)
        if not sequence_value:
            continue

        sequence = str(sequence_value)
        date_value = _normalize_date_value(record.get(date_field)) if date_field else None
        key = (sequence, date_value)
        if key in seen:
            continue

        seen.add(key)
        pairs.append(key)

    # Fallback for records that still expose a single sequence without a recognised date.
    if not pairs:
        sequence_value = record.get("seq") or record.get("sequence")
        if sequence_value:
            sequence = str(sequence_value)
            key = (sequence, None)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    return pairs


def _load_split_sequences(
    max_base_length: int,
) -> tuple[List[str], List[str], int, int]:
    """Load influenza sequences split by cutoff for training and validation."""
    dataset: Dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    train_sequences: List[str] = []
    validation_sequences: List[str] = []
    train_filtered_rows = 0
    validation_filtered_rows = 0

    for record in dataset:
        for sequence_value, date_value in _extract_sequence_date_pairs(record):
            if date_value is None:
                continue

            prepared = _prepare_sequence(sequence_value, max_base_length)
            if date_value <= DATE_CUTOFF:
                train_filtered_rows += 1
                if prepared is None:
                    continue
                train_sequences.append(prepared)
            else:
                validation_filtered_rows += 1
                if prepared is None:
                    continue
                validation_sequences.append(prepared)

    if (
        MAX_VALIDATION_SEQUENCES
        and MAX_VALIDATION_SEQUENCES > 0
        and len(validation_sequences) > MAX_VALIDATION_SEQUENCES
    ):
        rng = random.Random(RANDOM_SEED + 99)
        validation_sequences = rng.sample(validation_sequences, MAX_VALIDATION_SEQUENCES)

    if not train_sequences:
        raise RuntimeError(
            "No usable influenza sequences found on or before the configured cutoff."
        )
    if not validation_sequences:
        raise RuntimeError(
            "No usable influenza sequences found after the configured cutoff for validation."
        )

    return train_sequences, validation_sequences, train_filtered_rows, validation_filtered_rows


class InfluenzaSequenceDenoisingGenerator:
    """Finite dataset variant of the denoising generator for influenza sequences."""

    def __init__(
        self,
        *,
        sequences: Sequence[str],
        batch_size: int,
        tokenizer: CodonTokenizer,
        seed: int | None = None,
    ) -> None:
        if not sequences:
            raise ValueError("The generator requires at least one prepared sequence.")

        self._sequences: Sequence[str] = sequences
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self._rng = random.Random(seed)
        self._order: List[int] = []
        self._order_iter: Iterator[int] | None = None
        self._reset_cycle()

    def _reset_cycle(self) -> None:
        self._order = list(range(len(self._sequences)))
        self._rng.shuffle(self._order)
        self._order_iter = iter(self._order)

    def _next_sequence(self) -> str:
        if self._order_iter is None:
            self._reset_cycle()
        try:
            idx = next(self._order_iter)  # type: ignore[assignment]
        except StopIteration:
            self._reset_cycle()
            idx = next(self._order_iter)  # type: ignore[assignment]
        return self._sequences[idx]

    def _contains_nan(self, batch_inputs: Dict[str, np.ndarray], batch_labels: np.ndarray) -> bool:
        """Return True if any tensor in the batch contains NaN values."""
        token_ids = batch_inputs["token_ids"].astype(np.float32)
        padding_mask = batch_inputs["padding_mask"].astype(np.float32)
        labels = batch_labels.astype(np.float32)
        return bool(
            np.isnan(token_ids).any()
            or np.isnan(padding_mask).any()
            or np.isnan(labels).any()
        )

    def __iter__(self) -> "InfluenzaSequenceDenoisingGenerator":
        return self

    def __next__(self):  # type: ignore[override]
        inputs: List[List[int]] = []
        labels: List[List[int]] = []
        masks: List[List[int]] = []

        while len(inputs) < self.batch_size:
            original = self._next_sequence()
            noisy = _apply_random_noise(original, self._rng)

            input_ids = self.tokenizer.encode(noisy)
            label_ids = self.tokenizer.encode(original)
            mask = _attention_mask(label_ids, self.tokenizer.PAD_TOKEN_ID)

            inputs.append(input_ids)
            labels.append(label_ids)
            masks.append(mask)

        batch_inputs = {
            "token_ids": np.asarray(inputs, dtype=np.int32),
            "padding_mask": np.asarray(masks, dtype=np.int32),
        }
        batch_labels = np.asarray(labels, dtype=np.int32)

        if self._contains_nan(batch_inputs, batch_labels):
            print("Skipping batch containing NaN values in tokenized inputs.")
            return self.__next__()

        return batch_inputs, batch_labels


def build_datasets(
    tokenizer: CodonTokenizer,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, int, int, int, int, int]:
    """Create train/validation tf.data datasets and report dataset statistics."""
    max_base_length = tokenizer.CONTEXT_LENGTH * 3
    train_sequences, validation_sequences, train_filtered_rows, validation_filtered_rows = (
        _load_split_sequences(max_base_length)
    )
    usable_train_sequences = len(train_sequences)
    usable_validation_sequences = len(validation_sequences)
    steps_per_epoch = max(1, math.ceil(usable_train_sequences / BATCH_SIZE))
    validation_steps = max(1, math.ceil(usable_validation_sequences / BATCH_SIZE))

    train_generator = InfluenzaSequenceDenoisingGenerator(
        sequences=train_sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        seed=RANDOM_SEED,
    )

    validation_generator = InfluenzaSequenceDenoisingGenerator(
        sequences=validation_sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        seed=RANDOM_SEED + 1,
    )

    def train_generator_fn():
        while True:
            yield next(train_generator)

    def validation_generator_fn():
        while True:
            yield next(validation_generator)

    output_signature = (
        {
            "token_ids": tf.TensorSpec(
                shape=(BATCH_SIZE, tokenizer.CONTEXT_LENGTH),
                dtype=tf.int32,
            ),
            "padding_mask": tf.TensorSpec(
                shape=(BATCH_SIZE, tokenizer.CONTEXT_LENGTH),
                dtype=tf.int32,
            ),
        },
        tf.TensorSpec(
            shape=(BATCH_SIZE, tokenizer.CONTEXT_LENGTH),
            dtype=tf.int32,
        ),
    )

    train_dataset = tf.data.Dataset.from_generator(
        train_generator_fn,
        output_signature=output_signature,
    ).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_generator(
        validation_generator_fn,
        output_signature=output_signature,
    ).prefetch(tf.data.AUTOTUNE)

    return (
        train_dataset,
        validation_dataset,
        steps_per_epoch,
        validation_steps,
        train_filtered_rows,
        validation_filtered_rows,
        usable_train_sequences,
        usable_validation_sequences,
    )


def compute_training_schedule(
    model: keras.Model,
    tokenizer: CodonTokenizer,
    *,
    usable_sequences: int,
    steps_per_epoch: int,
) -> dict[str, int]:
    """Compute training steps using the finite influenza dataset."""
    total_params = model.count_params()
    tokens_per_micro_step = BATCH_SIZE * tokenizer.CONTEXT_LENGTH
    total_micro_steps = steps_per_epoch * NUM_EPOCHS
    optimizer_updates = max(1, math.ceil(total_micro_steps / GRADIENT_ACCUMULATION_STEPS))
    token_budget = usable_sequences * tokenizer.CONTEXT_LENGTH * NUM_EPOCHS

    return {
        "total_params": total_params,
        "token_budget": token_budget,
        "tokens_per_micro_step": tokens_per_micro_step,
        "tokens_per_update": tokens_per_micro_step * GRADIENT_ACCUMULATION_STEPS,
        "total_micro_steps": total_micro_steps,
        "optimizer_updates": optimizer_updates,
        "steps_per_epoch": steps_per_epoch,
    }


def create_learning_rate_schedule(total_updates: int) -> tuple[CosineDecay, int]:
    """Build cosine decay with linear warmup for the optimizer."""
    warmup_steps = max(1, min(total_updates, int(total_updates * WARMUP_RATIO)))
    schedule = CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_updates,
        alpha=COSINE_DECAY_ALPHA,
        warmup_target=LEARNING_RATE,
        warmup_steps=warmup_steps,
        name="stage_2_cosine_decay",
    )
    return schedule, warmup_steps


def main() -> None:
    tokenizer = CodonTokenizer()
    model = build_stage_2_model(tokenizer)

    if os.path.exists(STAGE_1_WEIGHTS_PATH):
        model.load_weights(STAGE_1_WEIGHTS_PATH)
        print(f"Loaded stage 1 weights from {STAGE_1_WEIGHTS_PATH}.")

    (
        train_dataset,
        validation_dataset,
        steps_per_epoch,
        validation_steps,
        train_filtered_rows,
        validation_filtered_rows,
        usable_train_sequences,
        usable_validation_sequences,
    ) = build_datasets(tokenizer)
    schedule = compute_training_schedule(
        model,
        tokenizer,
        usable_sequences=usable_train_sequences,
        steps_per_epoch=steps_per_epoch,
    )
    learning_rate_schedule, warmup_steps = create_learning_rate_schedule(
        schedule["optimizer_updates"]
    )

    print("Stage 2 training configuration:")
    print(f"  Training records (date <= {DATE_CUTOFF}): {train_filtered_rows:,}")
    print(f"  Validation records (date > {DATE_CUTOFF}): {validation_filtered_rows:,}")
    print(
        f"  Usable training sequences after preprocessing: {usable_train_sequences:,}"
    )
    print(
        f"  Usable validation sequences after preprocessing: {usable_validation_sequences:,}"
    )
    if usable_validation_sequences < validation_filtered_rows:
        print(
            f"    Sampled from {validation_filtered_rows:,} total validation sequences "
            f"(cap {MAX_VALIDATION_SEQUENCES:,})."
        )
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Steps per epoch: {schedule['steps_per_epoch']:,}")
    print(f"  Validation steps per epoch: {validation_steps:,}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Total model parameters: {schedule['total_params']:,}")
    print(f"  Estimated token budget: {schedule['token_budget']:,}")
    print(f"  Tokens per micro step: {schedule['tokens_per_micro_step']:,}")
    print(f"  Tokens per optimizer update: {schedule['tokens_per_update']:,}")
    print(f"  Optimizer updates: {schedule['optimizer_updates']:,}")
    print(f"  Learning rate warmup steps: {warmup_steps:,}")
    print(f"  Initial learning rate: {LEARNING_RATE:.2e}")
    print(f"  Final learning rate after decay: {LEARNING_RATE * COSINE_DECAY_ALPHA:.2e}")

    optimizer = optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        global_clipnorm=GLOBAL_CLIPNORM,
    )

    loss = losses.SparseCategoricalCrossentropy(
        from_logits=True,
        ignore_class=0,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
    )

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=STAGE_2_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=False,
        save_freq="epoch",
        verbose=1,
    )

    callbacks_list = [checkpoint_cb, callbacks.TerminateOnNaN()]

    model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1,
    )

    print("Stage 2 training complete. Final weights saved to", STAGE_2_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
