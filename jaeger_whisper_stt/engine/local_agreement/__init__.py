"""LocalAgreement-2 streaming STT: rolling window + word-cursor commit."""

from .pipeline import WhisperSTTLocalAgreement

__all__ = ["WhisperSTTLocalAgreement", "bench"]


def bench(audio, sr, ref=None, *, model="base.en", window_s=7.0):
    """Same unit of work as ``window`` — one rolling decode.  The
    agreement rule is pure text post-processing and costs nothing
    measurable, so the interesting number is identical: does a window
    decode finish inside the cadence?"""
    from ..window import bench as window_bench
    r = window_bench(audio, sr, ref, model=model, window_s=window_s)
    r.method = "local_agreement"
    return r
