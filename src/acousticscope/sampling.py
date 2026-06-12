"""Dynamic audio sampling logic from the AcousticScope prototype.

The original experiment captured Alexa responses with a microphone. This module
keeps that core algorithm in an importable form so the project structure mirrors
the paper without forcing every README demo to initialize audio hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    sample_rate: int = 44_100
    chunk_seconds: float = 0.35
    silence_threshold: float = 0.1
    startup_timeout_seconds: float = 10.0
    max_duration_seconds: float = 120.0
    max_silent_chunks: int = 4


def capture_until_silence(recorder, config: SamplingConfig = SamplingConfig()) -> np.ndarray:
    """Record chunks until speech starts and then sustained silence is observed.

    ``recorder`` is intentionally injected. In production it can wrap
    ``sounddevice.rec``; in tests it can be a deterministic generator.
    """
    chunks: list[np.ndarray] = []
    silent_chunks = 0
    samples_per_chunk = int(config.sample_rate * config.chunk_seconds)

    start = monotonic()
    while monotonic() - start < config.startup_timeout_seconds:
        chunk = np.asarray(recorder(samples_per_chunk, config.sample_rate)).squeeze()
        if np.max(np.abs(chunk)) >= config.silence_threshold:
            chunks.append(chunk)
            break

    if not chunks:
        return np.array([])

    while sum(len(chunk) for chunk in chunks) / config.sample_rate < config.max_duration_seconds:
        chunk = np.asarray(recorder(samples_per_chunk, config.sample_rate)).squeeze()
        chunks.append(chunk)

        if np.max(np.abs(chunk)) < config.silence_threshold:
            silent_chunks += 1
            if silent_chunks >= config.max_silent_chunks:
                break
        else:
            silent_chunks = 0

    return np.concatenate(chunks)
