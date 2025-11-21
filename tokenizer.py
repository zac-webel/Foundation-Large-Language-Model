"""Codon-level tokenizer for the Pathogen AI Encoder Base model.

This module exposes a `tokenizer` instance that chunks nucleotide sequences
into codon tokens drawn from the {A, T, C, G} alphabet. The vocabulary assigns
ID 0 to the padding token and enumerates all 64 possible codons starting from 1.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence


class CodonTokenizer:
    """Tokenizer that maps nucleotide sequences to codon-level token IDs."""

    PAD_TOKEN = "<PAD>"
    PAD_TOKEN_ID = 0
    CONTEXT_LENGTH = 567
    BASES = ("A", "T", "C", "G")

    def __init__(self) -> None:
        codons = ["".join(codon) for codon in itertools.product(self.BASES, repeat=3)]
        self._token_to_id: Dict[str, int] = {
            codon: index for index, codon in enumerate(codons, start=1)
        }
        self._id_to_token: Dict[int, str] = {
            index: codon for codon, index in self._token_to_id.items()
        }

    def encode(
        self,
        sequence: str,
        *,
        max_length: int | None = CONTEXT_LENGTH,
        add_padding: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        """Encode a nucleotide sequence into codon token IDs.

        Args:
            sequence: Input nucleotide string composed of characters A/T/C/G.
            max_length: Maximum number of codon tokens to emit. Defaults to
                the model context length (567). Use ``None`` to disable.
            add_padding: When True, right-pad the sequence with PAD token IDs
                up to ``max_length``.
            truncation: When True, truncate codon tokens that exceed
                ``max_length``. When False, raise an error if the sequence is
                too long.

        Returns:
            A list of integer token IDs representing the input sequence.
        """
        codons = self._to_codons(sequence)

        if max_length is None:
            target_length = len(codons)
        else:
            target_length = max_length

        if len(codons) > target_length:
            if truncation:
                codons = codons[:target_length]
            else:
                raise ValueError(
                    "Sequence produces more codons than max_length and truncation is disabled."
                )

        token_ids = [self._token_to_id[codon] for codon in codons]

        if add_padding:
            padding_target = target_length
            if len(token_ids) > padding_target:
                raise ValueError("Cannot pad because token_ids exceed padding target length.")
            padding_needed = padding_target - len(token_ids)
            token_ids.extend([self.PAD_TOKEN_ID] * padding_needed)

        return token_ids

    def decode(self, token_ids: Sequence[int], *, skip_padding: bool = True) -> str:
        """Decode codon token IDs back into a nucleotide string."""
        codons: List[str] = []
        for token_id in token_ids:
            if skip_padding and token_id == self.PAD_TOKEN_ID:
                continue
            if token_id not in self._id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            codons.append(self._id_to_token[token_id])
        return "".join(codons)

    def vocabulary(self, *, print_output: bool = True) -> Dict[int, str]:
        """Return the vocabulary mapping and optionally print it."""
        vocab = {self.PAD_TOKEN_ID: self.PAD_TOKEN}
        vocab.update(self._id_to_token)
        if print_output:
            for token_id in sorted(vocab):
                print(f"{token_id}: {vocab[token_id]}")
        return vocab

    def _to_codons(self, sequence: str) -> List[str]:
        """Split a nucleotide sequence into codons after validation."""
        cleaned = sequence.strip().upper().replace("\n", "")
        if not cleaned:
            return []
        invalid = set(cleaned) - set(self.BASES)
        if invalid:
            raise ValueError(f"Encountered invalid nucleotides: {sorted(invalid)}")
        if len(cleaned) % 3 != 0:
            raise ValueError("Sequence length must be divisible by 3 for codon tokenization.")
        return [cleaned[i : i + 3] for i in range(0, len(cleaned), 3)]

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Expose the codon to token ID mapping."""
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Expose the token ID to codon mapping."""
        return {self.PAD_TOKEN_ID: self.PAD_TOKEN} | dict(self._id_to_token)


