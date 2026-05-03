"""
Token-level augmentation for MIDI training data.

Augmentations operate on already-tokenized sequences so they're cheap to apply
on the fly during training (or precomputed into shards if we have disk to spare):

- pitch_shift: shift all NoteOn/NoteOff tokens by ±n semitones
- velocity_jitter: nudge Velocity tokens by ±k bins
- (tempo stretch is handled at the symusic level before tokenization, not here)
"""

from __future__ import annotations

import random
from typing import Sequence

from miditok import MIDILike


def pitch_shift(
    tokens: Sequence[int], tokenizer: MIDILike, semitones: int
) -> list[int]:
    """Shift all pitched note tokens by `semitones`. Drum tokens are left alone."""
    if semitones == 0:
        return list(tokens)

    vocab = tokenizer.vocab  # str -> id
    inv = {v: k for k, v in vocab.items()}

    def shift(tok_id: int) -> int:
        name = inv.get(tok_id)
        if name is None:
            return tok_id
        for prefix in ("NoteOn_", "NoteOff_"):
            if name.startswith(prefix):
                pitch = int(name[len(prefix):])
                new_pitch = pitch + semitones
                if 0 <= new_pitch <= 127:
                    new_name = f"{prefix}{new_pitch}"
                    return vocab.get(new_name, tok_id)
                return tok_id  # out of range -> leave unchanged
        return tok_id

    return [shift(t) for t in tokens]


def velocity_jitter(
    tokens: Sequence[int], tokenizer: MIDILike, max_bins: int, rng: random.Random
) -> list[int]:
    """Nudge each Velocity token by a uniform random ±max_bins."""
    if max_bins <= 0:
        return list(tokens)

    vocab = tokenizer.vocab
    inv = {v: k for k, v in vocab.items()}

    velocity_ids: dict[int, int] = {}
    for name, tid in vocab.items():
        if name.startswith("Velocity_"):
            try:
                velocity_ids[int(name[len("Velocity_"):])] = tid
            except ValueError:
                pass
    if not velocity_ids:
        return list(tokens)
    sorted_vels = sorted(velocity_ids)

    def jitter(tok_id: int) -> int:
        name = inv.get(tok_id)
        if name is None or not name.startswith("Velocity_"):
            return tok_id
        try:
            v = int(name[len("Velocity_"):])
        except ValueError:
            return tok_id
        idx = sorted_vels.index(v) if v in sorted_vels else -1
        if idx < 0:
            return tok_id
        new_idx = max(0, min(len(sorted_vels) - 1, idx + rng.randint(-max_bins, max_bins)))
        return velocity_ids[sorted_vels[new_idx]]

    return [jitter(t) for t in tokens]


def random_augment(
    tokens: Sequence[int],
    tokenizer: MIDILike,
    pitch_range: int = 6,
    velocity_bins: int = 2,
    rng: random.Random | None = None,
) -> list[int]:
    rng = rng or random.Random()
    semitones = rng.randint(-pitch_range, pitch_range)
    out = pitch_shift(tokens, tokenizer, semitones)
    out = velocity_jitter(out, tokenizer, velocity_bins, rng)
    return out
