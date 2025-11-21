# Foundation LLM

Pathogen AI Encoder Base is a foundation model for pathogen genomics implemented as a encoder-only transformer. It builds on the DeBERTa v3 backbone to learn codon-level representations of nucleotide sequences for downstream tasks in pathogen evolution modeling, vaccine design, and biosecurity analytics.

## Key Capabilities
- Learns contextual codon embeddings that capture genome-wide structure and long-range dependencies.
- Detects biologically implausible substitutions through a replaced token detection (RTD) objective.
- Transfers knowledge from diverse species before specializing on virus-only corpora.
- Operates entirely within an encoder-only transformer stack, avoiding autoregressive decoding overhead.

## Architecture Overview
The model follows a DeBERTa v3 style encoder configuration with disentangled attention. No decoder component is included. Padding is the only special token (index `0`), keeping the embedding table aligned with codon vocabulary indices. Sequences are limited to a fixed context window of 567 codon tokens.

## Tokenization and Vocabulary
- Codon-level tokenization segments nucleotide strings into contiguous 3-mer units over the alphabet `{A, T, C, G}`.
- The discrete vocabulary covers all possible codons, enabling direct modeling of synonymous and nonsynonymous variation.
- Inputs are truncated or right-padded with the padding token (`0`) to reach the 567-token context length.

## Pretraining Curriculum
Pretraining proceeds in two sequential stages while keeping the same self-supervised objective. Both stages share the identical corruption process and loss, ensuring a smooth transfer between datasets.

1. **Broad Genomic Exposure** – Stage one trains on the `InstaDeepAI/multi_species_genomes` dataset to capture cross-species codon usage patterns and general genomic structure.
2. **Virus Specialization** – Stage two fine-tunes the encoder on a private virus-only dataset, adapting representations to viral sequence dynamics without altering the architecture.

## Replaced Token Detection Objective
Replaced Token Detection (RTD) is the sole pretraining objective across both stages. The encoder learns to distinguish between original codons and positions that have been replaced by a corruptor. This objective encourages the model to:
- Build awareness of codon context and biological plausibility.
- Retain sensitivity to long-range dependencies in coding regions.
- Produce robust embeddings that generalize to downstream tasks such as variant impact assessment and sequence quality control.

## Data Pipeline
For each nucleotide sequence in the training corpus:
1. Two copies of the sequence are created.
2. In the first copy, 0–30% of positions are selected uniformly at random and the chosen nucleotides are replaced with a random base from `{"A", "T", "C", "G"}`.
3. The corrupted sequence is tokenized into codons and used as the model input.
4. The unaltered copy is tokenized into codons and serves as the ground-truth target for RTD supervision.
5. A padding mask is constructed so that padding tokens (`0`) do not affect loss computation or attention weights.

## Downstream Applications
The resulting encoder is intended to support:
- Variant impact assessment and evolutionary trajectory modeling.
- Vaccine design workflows that rely on codon-level representations.
- Biosecurity analytics, including sequence quality control and anomaly detection.

## Running Stage 1 Training

Create a fresh Python environment (Python 3.11 recommended), install the dependencies, and launch the training script:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python stage_1_train.py
```

Set the environment variable `KERAS_BACKEND=tensorflow` in your shell if it is not already configured so that standalone Keras uses TensorFlow.
