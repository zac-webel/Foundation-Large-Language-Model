"""Stage 1 training script for the Pathogen AI Encoder Base model."""
from __future__ import annotations

import math

import keras
from keras import callbacks, losses, mixed_precision, optimizers
from keras.optimizers.schedules import CosineDecay
import tensorflow as tf

from model import build_backbone
from processing import SequenceDenoisingGenerator
from tokenizer import CodonTokenizer

# # Enable mixed precision across the training script.
mixed_precision.set_global_policy("mixed_float16")

# Training hyperparameters.
MAX_SEQUENCE_LENGTH =  567
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = int(100_000 / (BATCH_SIZE * MAX_SEQUENCE_LENGTH))
TOKENS_PER_PARAMETER = 2 #20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5
GLOBAL_CLIPNORM = 1.0
COSINE_DECAY_ALPHA = 0.1
WARMUP_RATIO = 0.1
NUM_EPOCHS = 100
DATASET_NAME = "InstaDeepAI/multi_species_genomes"
DATASET_SPLIT = "train"
SHUFFLE_BUFFER = 10_000
RANDOM_SEED = 1234
STAGE_1_WEIGHTS_PATH = "stage_1_base.weights.h5"


def build_stage_1_model(tokenizer: CodonTokenizer) -> keras.Model:
    """Attach a softmax classification head to the DeBERTa backbone."""
    backbone = build_backbone()
    vocab_size = len(tokenizer.vocabulary(print_output=False))

    sequence_output = backbone.output
    logits = keras.layers.Dense(
        vocab_size,
        dtype="float32",  
        name="stage_1_classification_head",
    )(sequence_output)

    return keras.Model(inputs=backbone.input, outputs=logits, name="pathogen_encoder_stage_1")


def compute_training_schedule(model: keras.Model, tokenizer: CodonTokenizer) -> dict[str, int]:
    """Determine token budget aligned training and update counts for stage 1."""
    total_params = model.count_params()
    token_budget = total_params * TOKENS_PER_PARAMETER

    tokens_per_micro_step = BATCH_SIZE * tokenizer.CONTEXT_LENGTH
    total_micro_steps = max(1, math.ceil(token_budget / tokens_per_micro_step))
    optimizer_updates = max(1, math.ceil(total_micro_steps / GRADIENT_ACCUMULATION_STEPS))

    return {
        "total_params": total_params,
        "token_budget": token_budget,
        "tokens_per_micro_step": tokens_per_micro_step,
        "tokens_per_update": tokens_per_micro_step * GRADIENT_ACCUMULATION_STEPS,
        "total_micro_steps": total_micro_steps,
        "epochs": NUM_EPOCHS,
        "optimizer_updates": optimizer_updates,
    }


def compute_epoch_step_allocation(total_micro_steps: int) -> list[int]:
    """Split the total micro steps across the configured epochs."""
    base_steps = total_micro_steps // NUM_EPOCHS
    remainder = total_micro_steps % NUM_EPOCHS
    steps = []
    for epoch_idx in range(NUM_EPOCHS):
        increment = 1 if epoch_idx < remainder else 0
        steps.append(base_steps + increment)

    # Guard against zero-step epochs by collapsing trailing zeros (should be rare).
    if steps and steps[-1] == 0:
        non_zero_steps = [s for s in steps if s > 0]
        if not non_zero_steps:
            non_zero_steps = [total_micro_steps]
        steps = non_zero_steps
    return steps


def create_learning_rate_schedule(total_updates: int) -> tuple[CosineDecay, int]:
    """Build cosine decay with linear warmup for the optimizer."""
    warmup_steps = max(1, min(total_updates, int(total_updates * WARMUP_RATIO)))
    schedule = CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_updates,
        alpha=COSINE_DECAY_ALPHA,
        warmup_target=LEARNING_RATE,
        warmup_steps=warmup_steps,
        name="stage_1_cosine_decay",
    )
    return schedule, warmup_steps


def build_training_dataset(tokenizer: CodonTokenizer) -> tf.data.Dataset:
    """Wrap the Python generator in a tf.data.Dataset for Keras 3.x."""

    def generator_fn():
        generator = SequenceDenoisingGenerator(
            batch_size=BATCH_SIZE,
            dataset_name=DATASET_NAME,
            split=DATASET_SPLIT,
            shuffle_buffer=SHUFFLE_BUFFER,
            seed=RANDOM_SEED,
            tokenizer=tokenizer,
        )
        for batch_inputs, batch_labels in generator:
            yield batch_inputs, batch_labels

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

    return tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=output_signature,
    ).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    tokenizer = CodonTokenizer()
    model = build_stage_1_model(tokenizer)

    schedule = compute_training_schedule(model, tokenizer)
    learning_rate_schedule, warmup_steps = create_learning_rate_schedule(
        schedule["optimizer_updates"]
    )

    print("Stage 1 training configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Total model parameters: {schedule['total_params']:,}")
    print(f"  Token budget: {schedule['token_budget']:,}")
    print(f"  Tokens per micro step: {schedule['tokens_per_micro_step']:,}")
    print(f"  Tokens per optimizer update: {schedule['tokens_per_update']:,}")
    epoch_steps = compute_epoch_step_allocation(schedule["total_micro_steps"])
    total_steps_from_allocation = sum(epoch_steps)
    if total_steps_from_allocation != schedule["total_micro_steps"]:
        raise RuntimeError(
            "Epoch step allocation does not match total micro steps."
        )
    actual_epochs = len(epoch_steps)
    schedule["epochs"] = actual_epochs
    print(f"  Epochs: {actual_epochs}")
    print(
        f"  Steps per epoch (distribution): base={schedule['total_micro_steps'] // NUM_EPOCHS:,}"
        f" remainder={schedule['total_micro_steps'] % NUM_EPOCHS}"
    )
    print(
        f"  Optimizer updates per epoch (avg): "
        f"{schedule['optimizer_updates'] / max(1, actual_epochs):.0f}"
    )
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


    model.compile(optimizer=optimizer, loss=loss)

    train_dataset = build_training_dataset(tokenizer)

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=STAGE_1_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=False,
        save_freq="epoch",
        verbose=1,
    )

    callbacks_list = [checkpoint_cb, callbacks.TerminateOnNaN()]
    initial_epoch = 0
    for epoch_idx, steps_per_epoch in enumerate(epoch_steps, start=1):
        if steps_per_epoch == 0:
            continue
        model.fit(
            train_dataset,
            epochs=epoch_idx,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks_list,
            verbose=1,
        )
        initial_epoch = epoch_idx

    print("Training complete. Final weights saved to", STAGE_1_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
