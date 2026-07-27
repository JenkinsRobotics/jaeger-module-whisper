"""STT method registry — the single place that maps a method NAME to its
live-adapter factory + its bench probe.

The audio session flips methods by name (``config.stt_mode``) via
:func:`get`; the CLI bench iterates :data:`METHODS`.  Adding a method =
add a subfolder + one entry here.  Imports stay lazy so importing the
registry never pulls the model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Method:
    name: str
    desc: str
    # (config, bus, wake_phrases) -> live STTAdapter
    make: Callable[..., Any]
    # (audio, sr, ref=None) -> BenchResult
    bench: Callable[..., Any]
    available: bool = True


# ── live-adapter factories (mirror the kwargs core/audio/session.py uses) ──
#
# `aec` and `reference_buffer` used to be threaded through here to reach
# the mic. Both are gone: jaeger_os.nodes.audio_io owns the device and
# the canceller, and publishes cleaned frames. What a pipeline needs now
# is the one thing that actually carries audio — the bus.
def _make_two_pass(config, bus, wake_phrases):
    from .two_pass import WhisperSTTTwoPass
    return WhisperSTTTwoPass(
        fast_model_name=config.fast_model_name,
        accurate_model_name=config.accurate_model_name,
        require_wake_word=config.require_wake_word,
        wake_phrases=wake_phrases,
        followup_window_s=config.followup_window_s,
        bus=bus,
    )


def _make_continuous(config, bus, wake_phrases):
    from .continuous import WhisperSTTContinuous
    return WhisperSTTContinuous(
        model_name=config.fast_model_name,
        require_wake_word=config.require_wake_word,
        wake_phrases=wake_phrases,
        followup_window_s=config.followup_window_s,
        bus=bus,
    )


def _make_local_agreement(config, bus, wake_phrases):
    from .local_agreement import WhisperSTTLocalAgreement
    return WhisperSTTLocalAgreement()  # raises NotImplementedError (stub)


# ── lazy bench wrappers (keep registry import light) ──
def _bench_two_pass(*a, **k):
    from .two_pass import bench
    return bench(*a, **k)


def _bench_continuous(*a, **k):
    from .continuous import bench
    return bench(*a, **k)


def _bench_local_agreement(*a, **k):
    from .local_agreement import bench
    return bench(*a, **k)


METHODS: dict[str, Method] = {
    "two_pass": Method(
        "two_pass", "dual whisper — fast base.en gates accurate medium.en",
        _make_two_pass, _bench_two_pass, available=True),
    "continuous": Method(
        "continuous", "single model + rolling re-transcription",
        _make_continuous, _bench_continuous, available=True),
    "local_agreement": Method(
        "local_agreement", "LocalAgreement streaming (stub)",
        _make_local_agreement, _bench_local_agreement, available=False),
}


def get(name: str) -> Method:
    """The method for ``name``; falls back to two_pass on an unknown name."""
    return METHODS.get(name) or METHODS["two_pass"]
