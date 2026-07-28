"""Shared helpers used by both two_pass.py and continuous.py.

Plugin-internal — not exported through __init__.py. Both STT modes need:
  • _MicStream            — mic frames off the bus, pauseable queue
  • _warm_stt             — silence-pass priming so first phrase isn't slow
  • _normalize            — lowercase + strip punctuation for wake-word match
  • _find_wake_in_text    — substring + fuzzy wake-phrase matcher
  • DEFAULT_WAKE_PHRASES  — canonical wake-phrase list (handles Whisper
                            mishearings: yeager/yager/jager/jaeger)

`_VadWorker` is two-pass specific (energy-segmentation in continuous mode
replaces it), so it stays in two_pass.py.

THIS MODULE NO LONGER OPENS THE MICROPHONE. `jaeger_os.nodes.audio_io`
owns the device and publishes echo-cancelled frames on /sense/mic/pcm;
`_MicStream` below subscribes. That is what lets a wake-word detector or
a level meter have the same microphone at the same time — under the old
shape this module held the input stream and nothing else could get it.

AEC moved with the device. It needs the mic frame and the audio being
played TOGETHER per frame, so it belongs where both are native; keeping
it here would have meant shipping the far-end reference across a
boundary 100 times a second.
"""

from __future__ import annotations

import queue
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any

import threading

import numpy as np

from jaeger_os.core.voice import is_non_speech_marker as is_non_speech_marker


_WAKE_PREFIXES = ("ok", "okay", "hey")
_ASSISTANT_NAMES = ("jaeger", "yeager", "yager", "jager")
DEFAULT_WAKE_PHRASES = tuple(f"{p} {n}" for p in _WAKE_PREFIXES for n in _ASSISTANT_NAMES)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _find_wake_in_text(
    text: str,
    wake_phrases: tuple[str, ...],
    wake_match_threshold: float,
) -> tuple[bool, str]:
    """Return (matched, remainder_after_wake). Wake phrase MUST be at
    the FIRST 2 tokens of the transcript — VOICE-3 in
    docs/ROADMAP_0.2.0.md.

    Previously the matcher looked anywhere in the utterance, which
    meant "Yes I think hey jaeger is cool" wrongly triggered. The
    new gate is "wake phrase is the opening of the sentence" —
    matches a real call to the agent without matching incidental
    mentions. Falls back to a fuzzy match on the same 2-token
    window for Whisper mishearings.
    """
    norm = _normalize(text)
    tokens = norm.split()
    if not tokens:
        return False, ""

    # All known wake phrases are 2 tokens ("hey jaeger", "ok jaeger",
    # …). If a longer phrase ever joins the set, the window grows
    # with it.
    for phrase in wake_phrases:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        if len(tokens) < n:
            continue
        head = " ".join(tokens[:n])
        if head == phrase:
            return True, " ".join(tokens[n:]).strip()

    # Fuzzy fallback — only on the head window, not anywhere in the
    # sentence. Threshold from the caller (default 0.78).
    for phrase in wake_phrases:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        if len(tokens) < n:
            continue
        head = " ".join(tokens[:n])
        if SequenceMatcher(None, head, phrase).ratio() >= wake_match_threshold:
            return True, " ".join(tokens[n:]).strip()

    return False, ""


def stt_verbose() -> bool:
    """Per-phrase STT debug prints (``[heard] ...``, ``[skipped ...]``,
    ``[follow-up] ...``) go to stdout when ``JAEGER_STT_VERBOSE=1``.

    Operator-flipped 2026-06-07 after live testing showed the debug
    output dominated the conversation pane during normal voice use.
    The boot-time setup prints (``[stt-cont] Loading...`` etc.)
    still fire regardless — they're once-per-session, not noise.

    Set the env var in the operator's shell or instance config when
    troubleshooting voice; otherwise the voice-activity log in the
    TUI (controlled by ``/quiet``) is the operator's view of the
    pipeline."""
    import os as _os
    return _os.environ.get("JAEGER_STT_VERBOSE") == "1"


def _warm_stt(model, label: str, sample_rate: int) -> None:
    """Run a 1.5-second silence transcription so the first real phrase doesn't
    pay model setup cost. Whisper rejects audio under 1000 ms (skips
    inference + warns), so 1.5 s is the safe minimum for warming."""
    warm_audio = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    print(f"[{label}] warming up...", flush=True)
    t0 = time.perf_counter()
    try:
        list(model.transcribe(warm_audio, language="en"))
    except Exception as exc:
        print(f"[{label}] warm-up skipped: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"[{label}] primed ({time.perf_counter() - t0:.1f}s).", flush=True)


#: How long to wait for the first mic frame before warning that
#: nothing is publishing. Long enough that a slow driver boot does not
#: cry wolf, short enough that an operator sees it before giving up.
LIVENESS_TIMEOUT_S = 3.0


class _MicStream:
    """Mic frames off the bus, with the queue+pause shape the pipelines
    already expect.

    Deliberately keeps `_MicStream`'s interface — `.q`, `.start()`,
    `.stop()`, `.set_paused()`, `.paused` — because the two pipelines
    are written against it and the change worth making here is WHERE
    the audio comes from, not how the VAD reads it.

    PAUSE IS LOCAL NOW. It used to stop the device. A shared microphone
    cannot honour that: one consumer pausing would starve every other
    subscriber. So a paused stream drops frames on arrival instead. The
    observable effect for this pipeline is identical, and the device
    keeps serving everyone else.
    """

    def __init__(
        self,
        *,
        bus: Any,
        sample_rate: int,
        frame_samples: int,
        max_queue_frames: int = 200,
    ) -> None:
        self.bus = bus
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_frames)
        self.paused = False
        self._subscribed = False
        self._frames_seen = 0
        self._liveness: threading.Timer | None = None
        #: Leftover samples between bus frames — see _on_frame.
        self._carry: np.ndarray = np.empty(0, dtype=np.float32)
        self._warned_rate = False

    def _on_frame(self, msg: Any) -> None:
        self._frames_seen += 1
        if self.paused:
            return
        # SAMPLE RATE is a real wiring error — correcting it means
        # resampling, a DSP decision this class must not make silently.
        rate = getattr(msg, "sample_rate", self.sample_rate)
        if rate != self.sample_rate:
            if not self._warned_rate:      # once, not 100x a second
                print(f"[mic] ignoring {rate} Hz audio (this pipeline is "
                      f"{self.sample_rate} Hz) — fix the driver's rate or "
                      f"the engine's sample_rate", file=sys.stderr)
                self._warned_rate = True
            return

        # FRAME SIZE is not. This used to drop any frame that was not
        # exactly frame_samples, which made the module deaf against a
        # perfectly good driver: jaeger_os's audio_io publishes 10 ms
        # frames because speexdsp's AEC requires fixed equal-length
        # near/far frames, while this pipeline's VAD wants 30 ms. Both
        # defaults are right for their own reasons, and neither side
        # can be declared the wiring error.
        #
        # So REGROUP instead of refusing. Buffering is lossless and
        # costs a copy — unlike resampling, which is why the original
        # refusal was correct about rate and wrong about size. Works in
        # both directions: 3 small frames in -> 1 block out, or 1 large
        # frame in -> several blocks out.
        samples = np.frombuffer(msg.samples, dtype=np.float32)
        self._carry = (samples if self._carry.size == 0
                       else np.concatenate((self._carry, samples)))
        n = self.frame_samples
        while self._carry.size >= n:
            self._offer(self._carry[:n].reshape(-1, 1))
            self._carry = self._carry[n:]

    def _offer(self, sample: np.ndarray) -> None:
        """Queue one VAD-sized block. Newest wins: a backed-up VAD
        should work on recent audio, not grind through a backlog from
        seconds ago."""
        try:
            self.q.put_nowait(sample)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(sample)
            except queue.Full:
                pass

    def start(self) -> None:
        if self._subscribed:
            return
        from jaeger_os.transport import topics
        self.bus.subscribe(topics.SENSE_MIC_PCM, self._on_frame)
        self._subscribed = True
        self._arm_liveness_check(topics.SENSE_MIC_PCM)

    def _arm_liveness_check(self, topic: str) -> None:
        """Say something if no audio ever arrives.

        This is the failure mode the driver split introduced, and it is
        the nastiest kind: subscribing to a topic nobody publishes
        SUCCEEDS. Without this the symptom is an agent that never hears
        you, reports healthy, and logs nothing — strictly worse than
        the ``OSError`` you used to get from a missing device.
        """
        def _check() -> None:
            if self._frames_seen:
                return
            print(
                f"[mic] no audio on {topic} after {LIVENESS_TIMEOUT_S:.0f}s. "
                f"Nothing is publishing it — is the audio_io driver in "
                f"this app's node list? Subscribing to an unpublished "
                f"topic succeeds silently, so this warning is the only "
                f"symptom you get.",
                file=sys.stderr, flush=True)

        self._liveness = threading.Timer(LIVENESS_TIMEOUT_S, _check)
        self._liveness.daemon = True
        self._liveness.start()

    def stop(self) -> None:
        if self._liveness is not None:
            self._liveness.cancel()
            self._liveness = None
        if not self._subscribed:
            return
        from jaeger_os.transport import topics
        try:
            self.bus.unsubscribe(topics.SENSE_MIC_PCM, self._on_frame)
        finally:
            self._subscribed = False

    def drain(self) -> None:
        with self.q.mutex:
            self.q.queue.clear()

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if not paused:
            # Anything captured during the pause boundary is stale.
            self.drain()
