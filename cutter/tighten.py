"""Cut kept words into timeline ranges snapped to silence.

A decision with ``action: drop`` removes its words. Filler drops do the same
when ``fillers.json`` is present. Each remaining run becomes one range.
Flagged decisions leave a CHECK marker for Final Cut.

Parakeet often stretches the last word of a sentence across the pause after
it. A range ends where the voice stops. With speech segments and
``tighten.use_vad``, that point is the segment around the word. Otherwise it
is measured on the analysis audio.

A clap zone is not played. A script chapter marks the chosen take.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from cutter.audio import rms_envelope, snap_time, voice_end, voice_levels
from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import (
    ChaptersConfig,
    ClapsConfig,
    Profile,
    ScriptConfig,
    TightenConfig,
    VadConfig,
    load_profile,
)
from cutter.models import (
    AlignmentArtifact,
    AlignmentData,
    AudioEventsArtifact,
    Clap,
    Decision,
    DecisionsArtifact,
    DroppedSpan,
    Marker,
    Range,
    Source,
    SourcesArtifact,
    SourcesData,
    SpeechSegment,
    TimelineArtifact,
    TimelineChapter,
    TimelineData,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)

STAGE = "tighten"
STAGE_VERSION = 4

logger = logging.getLogger("cutter.tighten")

_TINY_FRAGMENT = "CHECK: tiny fragment removed"
_CLAP_INSIDE = "CHECK: clap inside range"
_UNHEARD = "CHECK: speech without transcript"
_UNSCRIPTED = "CHECK: unscripted"
_SENTENCE_END = (".", "?", "!")

# §7.6.3. A speech-segment start within this many seconds before the first
# word of a range pulls the in-point back to that start. Not a profile key.
_VAD_START_LEAD_S = 0.2

_OPTIONAL_ARTIFACTS = ("fillers.json", "audio_events.json", "alignment.json")


def build_timeline(
    words: list[Word],
    decisions: list[Decision],
    sources: SourcesData,
    envelopes: dict[str, tuple[np.ndarray, int]],
    config: TightenConfig,
    *,
    fillers: list[Decision] | None = None,
    speech: list[SpeechSegment] | None = None,
    claps: list[Clap] | None = None,
    alignment: AlignmentData | None = None,
    vad: VadConfig | None = None,
    claps_config: ClapsConfig | None = None,
    script: ScriptConfig | None = None,
    chapters: ChaptersConfig | None = None,
) -> TimelineData:
    """Build playback ranges, dropped spans, and chapter markers.

    ``envelopes`` maps source id to ``(rms, sample_rate)``. A missing envelope
    does not snap; the raw cut point is kept.

    The keyword arguments are the Phase 2 inputs. Omit them, or pass empty
    speech and no claps, and the ranges match Phase 1. ``tighten.use_vad``
    false also keeps those cut points when speech segments are present.
    """
    filler_decisions = list(fillers or [])
    speech_segments = list(speech or [])
    use_vad = config.use_vad and bool(speech_segments)
    zones = _clap_zones(claps, claps_config)
    by_index = {word.i: word for word in words}
    by_source = _words_by_source(words)
    source_by_id = {source.id: source for source in sources.sources}
    speech_by_source = _speech_by_source(speech_segments)
    fps = Fraction(sources.fps)
    dropped_ids = _dropped_ids([*decisions, *filler_decisions])
    kept = [word for word in sorted(words, key=lambda word: word.i) if word.i not in dropped_ids]
    voices = {
        source_id: voice_levels(samples, config.rms_frame_ms)
        for source_id, (samples, _rate) in envelopes.items()
    }
    spoken_ends = _spoken_ends(
        words,
        kept,
        speech_by_source,
        voices,
        config,
        use_vad=use_vad,
    )
    groups = _group_kept(kept, dropped_ids, config.max_gap_ms / 1000, spoken_ends)
    groups = _split_on_claps(
        groups,
        zones,
        config,
        spoken_ends,
        speech_by_source,
        voices,
        use_vad=use_vad,
    )
    ranges = [
        _range_for_group(
            group,
            by_source,
            source_by_id,
            envelopes,
            voices,
            config,
            fps,
            spoken_ends,
            speech_by_source,
            zones,
            use_vad=use_vad,
        )
        for group in groups
    ]
    ranges = _merge_touching(ranges, zones, fps)
    ranges = _remove_tiny(ranges, config, zones, fps)
    _attach_flags(ranges, [*decisions, *filler_decisions])
    _mark_unscripted(ranges, alignment, by_index, dropped_ids, script)
    _mark_unheard(ranges, speech_segments, words, vad)
    _mark_claps_inside(ranges, zones, fps)
    numbered = [
        rng.model_copy(update={"id": f"r{number:03d}"})
        for number, rng in enumerate(ranges, start=1)
    ]
    return TimelineData(
        ranges=numbered,
        dropped=_dropped_spans([*decisions, *filler_decisions], by_index),
        chapters=_chapters(alignment, chapters, dropped_ids, by_index),
    )


def run_tighten(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> TimelineArtifact:
    """Stage entry for ``cutter tighten <project_dir> [--force]``.

    Reads words, decisions, sources, and each source's analysis wav. When
    ``fillers.json``, ``audio_events.json``, or ``alignment.json`` is present,
    those bytes join the input hash and the cut uses them. Writes
    ``artifacts/timeline.json``. Pass ``profile`` from ``cutter run`` when it
    is not the default ``long`` profile. A cache hit returns the artifact
    already on disk.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    sources_path = artifacts / "sources.json"
    timeline_path = artifacts / "timeline.json"
    for label, path in (
        ("words", words_path),
        ("decisions", decisions_path),
        ("sources", sources_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} artifact: {path}")

    sources_artifact = SourcesArtifact.model_validate_json(
        sources_path.read_text(encoding="utf-8")
    )
    wav_paths = [_analysis_wav(project_dir, source) for source in sources_artifact.data.sources]
    for wav_path in wav_paths:
        if not wav_path.is_file():
            raise FileNotFoundError(f"missing analysis audio: {wav_path}")

    optional_paths = [
        artifacts / name for name in _OPTIONAL_ARTIFACTS if (artifacts / name).is_file()
    ]
    hashed_inputs = inputs_hash(
        artifacts=[words_path, decisions_path, sources_path, *wav_paths, *optional_paths]
    )
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        timeline_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return TimelineArtifact.model_validate_json(timeline_path.read_text(encoding="utf-8"))

    words = WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8")).data.words
    decisions = DecisionsArtifact.model_validate_json(
        decisions_path.read_text(encoding="utf-8")
    ).data.decisions
    fillers_artifact = _load_optional(artifacts / "fillers.json", DecisionsArtifact)
    events_artifact = _load_optional(artifacts / "audio_events.json", AudioEventsArtifact)
    alignment_artifact = _load_optional(artifacts / "alignment.json", AlignmentArtifact)
    envelopes = {
        source.id: rms_envelope(wav_path, loaded.tighten.rms_frame_ms)
        for source, wav_path in zip(sources_artifact.data.sources, wav_paths, strict=True)
    }
    artifact = TimelineArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=build_timeline(
            words,
            decisions,
            sources_artifact.data,
            envelopes,
            loaded.tighten,
            fillers=None if fillers_artifact is None else fillers_artifact.data.decisions,
            speech=None if events_artifact is None else events_artifact.data.speech,
            claps=None if events_artifact is None else events_artifact.data.claps,
            alignment=None if alignment_artifact is None else alignment_artifact.data,
            vad=loaded.vad,
            claps_config=loaded.claps,
            script=loaded.script,
            chapters=loaded.chapters,
        ),
    )
    write_artifact(timeline_path, artifact)
    return artifact


def _load_optional[T: DecisionsArtifact | AudioEventsArtifact | AlignmentArtifact](
    path: Path,
    model: type[T],
) -> T | None:
    if not path.is_file():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _dropped_ids(decisions: list[Decision]) -> set[int]:
    dropped: set[int] = set()
    for decision in decisions:
        if decision.action != "drop":
            continue
        first, last = decision.dropped_words
        dropped.update(range(first, last + 1))
    return dropped


def _words_by_source(words: list[Word]) -> dict[str, list[Word]]:
    grouped: dict[str, list[Word]] = {}
    for word in sorted(words, key=lambda item: item.i):
        grouped.setdefault(word.source, []).append(word)
    return grouped


def _speech_by_source(speech: list[SpeechSegment]) -> dict[str, list[SpeechSegment]]:
    grouped: dict[str, list[SpeechSegment]] = {}
    for segment in speech:
        grouped.setdefault(segment.source, []).append(segment)
    return grouped


def _clap_zones(
    claps: list[Clap] | None,
    config: ClapsConfig | None,
) -> dict[str, list[tuple[float, float]]]:
    """Forbidden ``[t - exclude_before, t + exclude_after]`` per source, in seconds."""
    if not claps or config is None:
        return {}
    before = config.exclude_before_ms / 1000
    after = config.exclude_after_ms / 1000
    grouped: dict[str, list[tuple[float, float]]] = {}
    for clap in claps:
        grouped.setdefault(clap.source, []).append((clap.t - before, clap.t + after))
    for source_zones in grouped.values():
        source_zones.sort()
    return grouped


def _spoken_ends(
    words: list[Word],
    kept: list[Word],
    speech: dict[str, list[SpeechSegment]],
    voices: dict[str, tuple[np.ndarray, float]],
    config: TightenConfig,
    *,
    use_vad: bool,
) -> dict[int, float]:
    """Where a word's voice stops, for gaps and the next range's in-point.

    With VAD, every word uses the segment that contains its midpoint, so a
    dropped word still blocks the next in-point until its voice has stopped.
    A word in no segment keeps the Phase 1 rule: the measured voice end for
    a sentence-final word, and the transcript end otherwise. Without VAD,
    only sentence-final kept words are measured.
    """
    ends: dict[int, float] = {}
    if use_vad:
        for word in words:
            segment = _segment_at(word, speech)
            if segment is not None:
                ends[word.i] = _clamp_spoken_end(segment.end, word)
            elif word.w.endswith(_SENTENCE_END):
                ends[word.i] = _spoken_end(word, voices, config)
            else:
                ends[word.i] = word.end
        return ends
    for word in kept:
        if word.w.endswith(_SENTENCE_END):
            ends[word.i] = _spoken_end(word, voices, config)
    return ends


def _clamp_spoken_end(segment_end: float, word: Word) -> float:
    """``clamp(segment_end, midpoint, word.end)``."""
    midpoint = (word.start + word.end) / 2
    return min(word.end, max(segment_end, midpoint))


def _segment_at(
    word: Word,
    speech: dict[str, list[SpeechSegment]],
) -> SpeechSegment | None:
    """The speech segment that contains ``word``'s midpoint, if any.

    A midpoint on a shared boundary belongs to the following segment.
    """
    midpoint = (word.start + word.end) / 2
    matches = [
        segment
        for segment in speech.get(word.source, [])
        if segment.start <= midpoint <= segment.end
    ]
    if not matches:
        return None
    interior = [segment for segment in matches if segment.start <= midpoint < segment.end]
    pool = interior or matches
    return min(pool, key=lambda segment: (segment.start, segment.end))


def _group_kept(
    kept: list[Word],
    dropped_ids: set[int],
    max_gap_s: float,
    spoken_ends: dict[int, float],
) -> list[list[Word]]:
    """Split kept words into ranges.

    ``spoken_ends`` replaces the transcript end of a word whose voice stops
    earlier, so a pause hidden inside a stretched timestamp still counts.
    """
    groups: list[list[Word]] = []
    for word in kept:
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]
        crossed_drop = any(previous.i < index < word.i for index in dropped_ids)
        previous_end = spoken_ends.get(previous.i, previous.end)
        if (
            word.source != previous.source
            or crossed_drop
            or word.start - previous_end > max_gap_s
        ):
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _split_on_claps(
    groups: list[list[Word]],
    zones: dict[str, list[tuple[float, float]]],
    config: TightenConfig,
    spoken_ends: dict[int, float],
    speech: dict[str, list[SpeechSegment]],
    voices: dict[str, tuple[np.ndarray, float]],
    *,
    use_vad: bool,
) -> list[list[Word]]:
    """Split a group when a clap zone has kept-word midpoints on both sides.

    Each side must be able to sit outside the zone. Otherwise the group stays
    whole and a CHECK marker records the zone that could not be removed.
    """
    if not zones:
        return groups
    split: list[list[Word]] = []
    for group in groups:
        split.extend(
            _split_group(
                group,
                zones.get(group[0].source, []),
                config,
                spoken_ends,
                speech,
                voices,
                use_vad=use_vad,
            )
        )
    return split


def _split_group(
    group: list[Word],
    zones: list[tuple[float, float]],
    config: TightenConfig,
    spoken_ends: dict[int, float],
    speech: dict[str, list[SpeechSegment]],
    voices: dict[str, tuple[np.ndarray, float]],
    *,
    use_vad: bool,
) -> list[list[Word]]:
    pieces = [group]
    for zone_start, zone_end in zones:
        nxt: list[list[Word]] = []
        for piece in pieces:
            left = [word for word in piece if _midpoint(word) < zone_start]
            right = [word for word in piece if _midpoint(word) > zone_end]
            inside = [word for word in piece if zone_start <= _midpoint(word) <= zone_end]
            if (
                left
                and right
                and not inside
                and _sides_clear_zone(
                    left,
                    right,
                    zone_start,
                    zone_end,
                    config,
                    spoken_ends,
                    speech,
                    voices,
                    use_vad=use_vad,
                )
            ):
                nxt.append(left)
                nxt.append(right)
            else:
                nxt.append(piece)
        pieces = nxt
    return pieces


def _sides_clear_zone(
    left: list[Word],
    right: list[Word],
    zone_start: float,
    zone_end: float,
    config: TightenConfig,
    spoken_ends: dict[int, float],
    speech: dict[str, list[SpeechSegment]],
    voices: dict[str, tuple[np.ndarray, float]],
    *,
    use_vad: bool,
) -> bool:
    """True when the left side can end before the zone and the right side can start after it."""
    last = left[-1]
    last_end = spoken_ends[last.i] if use_vad else _spoken_end(last, voices, config)
    if last_end + config.min_tail_ms / 1000 > zone_start:
        return False
    anchor = _in_anchor(right[0], speech, use_vad=use_vad)
    return zone_end <= anchor - config.min_head_ms / 1000


def _range_for_group(
    group: list[Word],
    by_source: dict[str, list[Word]],
    source_by_id: dict[str, Source],
    envelopes: dict[str, tuple[np.ndarray, int]],
    voices: dict[str, tuple[np.ndarray, float]],
    config: TightenConfig,
    fps: Fraction,
    spoken_ends: dict[int, float],
    speech: dict[str, list[SpeechSegment]],
    zones: dict[str, list[tuple[float, float]]],
    *,
    use_vad: bool,
) -> Range:
    first = group[0]
    last = group[-1]
    source = source_by_id[first.source]
    previous, following = _neighbours(by_source.get(first.source, []), first.i, last.i)
    if use_vad:
        last_end = spoken_ends[last.i]
    else:
        last_end = _spoken_end(last, voices, config)
    in_s, out_s = _cut_points(
        first,
        last,
        last_end,
        previous,
        following,
        source.duration_s,
        envelopes,
        config,
        spoken_ends,
        speech,
        zones.get(first.source, []),
        use_vad=use_vad,
    )
    in_frame, out_frame = _frames(in_s, out_s, fps, source.duration_frames)
    in_frame, out_frame = _clear_zone_frames(
        in_frame,
        out_frame,
        fps,
        zones.get(first.source, []),
        source.duration_frames,
        _midpoint(first),
        _midpoint(last),
    )
    return Range(
        id="r000",
        source=first.source,
        in_s=in_s,
        out_s=out_s,
        in_frame=in_frame,
        out_frame=out_frame,
        first_word=first.i,
        last_word=last.i,
    )


def _neighbours(
    source_words: list[Word],
    first_i: int,
    last_i: int,
) -> tuple[Word | None, Word | None]:
    previous: Word | None = None
    following: Word | None = None
    for word in source_words:
        if word.i < first_i:
            previous = word
        elif word.i > last_i:
            following = word
            break
    return previous, following


def _cut_points(
    first: Word,
    last: Word,
    last_end: float,
    previous: Word | None,
    following: Word | None,
    duration_s: float,
    envelopes: dict[str, tuple[np.ndarray, int]],
    config: TightenConfig,
    spoken_ends: dict[int, float],
    speech: dict[str, list[SpeechSegment]],
    zones: list[tuple[float, float]],
    *,
    use_vad: bool,
) -> tuple[float, float]:
    """Pad, clamp to the file and the neighbouring word, then snap inside the legal interval.

    ``last_end`` is where the voice of ``last`` stops. The out-point never
    snaps later than ``pad_tail_ms`` after it, so a range does not end with
    more silence than the pad. A clap zone shrinks that interval, and snap
    skips frames inside the zone.
    """
    anchor = _in_anchor(first, speech, use_vad=use_vad)
    raw_in = _clamp(anchor - config.pad_head_ms / 1000, 0.0, duration_s)
    if use_vad and previous is not None:
        floor = max(0.0, spoken_ends.get(previous.i, previous.end))
        raw_in = _clamp(max(raw_in, floor), 0.0, duration_s)
        in_lo = floor
    elif previous is not None and previous.end <= anchor:
        raw_in = _clamp(max(raw_in, previous.end), 0.0, duration_s)
        in_lo = max(0.0, previous.end)
    else:
        in_lo = 0.0 if previous is None else max(0.0, previous.end)
    raw_out = _clamp(last_end + config.pad_tail_ms / 1000, 0.0, duration_s)
    if following is not None:
        raw_out = _clamp(min(raw_out, following.start), 0.0, duration_s)

    in_hi = min(duration_s, anchor - config.min_head_ms / 1000)
    out_lo = max(0.0, last_end + config.min_tail_ms / 1000)
    out_hi = duration_s if following is None else min(duration_s, following.start)
    out_hi = min(out_hi, raw_out)
    raw_in, raw_out, in_lo, in_hi, out_lo, out_hi = _exclude_zones(
        raw_in,
        raw_out,
        in_lo,
        in_hi,
        out_lo,
        out_hi,
        zones,
        first,
        last,
    )
    envelope = envelopes.get(first.source)
    forbidden = zones or None
    return (
        _snap(envelope, raw_in, in_lo, in_hi, config, forbidden),
        _snap(envelope, raw_out, out_lo, out_hi, config, forbidden),
    )


def _in_anchor(
    word: Word,
    speech: dict[str, list[SpeechSegment]],
    *,
    use_vad: bool,
) -> float:
    """Transcript start, or an earlier segment start within ``_VAD_START_LEAD_S``."""
    if not use_vad:
        return word.start
    segment = _segment_at(word, speech)
    if segment is None:
        return word.start
    lead = word.start - segment.start
    if 0 <= lead <= _VAD_START_LEAD_S:
        return min(word.start, segment.start)
    return word.start


def _exclude_zones(
    raw_in: float,
    raw_out: float,
    in_lo: float,
    in_hi: float,
    out_lo: float,
    out_hi: float,
    zones: list[tuple[float, float]],
    first: Word,
    last: Word,
) -> tuple[float, float, float, float, float, float]:
    """Pull an in-point or out-point off a clap zone when the words sit on one side.

    A zone that covers a kept-word midpoint is left in place. The range stays,
    and a CHECK marker is attached later.
    """
    first_mid = _midpoint(first)
    last_mid = _midpoint(last)
    for zone_start, zone_end in zones:
        if last_mid <= zone_start and out_lo <= zone_start:
            out_hi = min(out_hi, zone_start)
            raw_out = min(raw_out, zone_start)
        elif first_mid >= zone_end and zone_end <= in_hi:
            in_lo = max(in_lo, zone_end)
            raw_in = max(raw_in, zone_end)
    return raw_in, raw_out, in_lo, in_hi, out_lo, out_hi


def _snap(
    envelope: tuple[np.ndarray, int] | None,
    raw_s: float,
    earliest_s: float,
    latest_s: float,
    config: TightenConfig,
    forbidden: list[tuple[float, float]] | None,
) -> float:
    if earliest_s > latest_s or envelope is None:
        return raw_s
    samples, sample_rate = envelope
    return snap_time(
        samples,
        sample_rate=sample_rate,
        frame_ms=config.rms_frame_ms,
        raw_s=raw_s,
        window_ms=config.snap_window_ms,
        earliest_s=earliest_s,
        latest_s=latest_s,
        forbidden=forbidden,
    )


def _spoken_end(
    word: Word,
    voices: dict[str, tuple[np.ndarray, float]],
    config: TightenConfig,
) -> float:
    """End of the voice in ``word``, never before its midpoint.

    Voice is ``voice_margin_db`` above the source's background. The midpoint
    floor keeps every kept word inside its range by the rule eval and the
    property tests use.
    """
    voice = voices.get(word.source)
    if voice is None:
        return word.end
    levels, background_db = voice
    found = voice_end(
        levels,
        frame_ms=config.rms_frame_ms,
        start_s=word.start,
        end_s=word.end,
        threshold_db=background_db + config.voice_margin_db,
        quiet_ms=config.voice_quiet_ms,
    )
    if found is None:
        return word.end
    midpoint = (word.start + word.end) / 2
    return min(word.end, max(found, midpoint))


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return min(max(value, low), high)


def _frames(in_s: float, out_s: float, fps: Fraction, duration_frames: int) -> tuple[int, int]:
    in_frame = math.floor(Fraction(in_s).limit_denominator(1_000_000) * fps)
    out_frame = math.ceil(Fraction(out_s).limit_denominator(1_000_000) * fps)
    if in_frame >= out_frame:
        out_frame = in_frame + 1
    return _clamp_frames(in_frame, out_frame, duration_frames)


def _clear_zone_frames(
    in_frame: int,
    out_frame: int,
    fps: Fraction,
    zones: list[tuple[float, float]],
    duration_frames: int,
    first_mid: float,
    last_mid: float,
) -> tuple[int, int]:
    """Pull frame edges off a clap zone after rounding, when the word still fits.

    Rounding can carry a cut that sits on the zone boundary one frame into
    the zone. The pull is skipped when it would drop the word's midpoint.
    """
    if not zones:
        return in_frame, out_frame
    first_frame = math.floor(Fraction(first_mid).limit_denominator(1_000_000) * fps)
    last_frame = math.floor(Fraction(last_mid).limit_denominator(1_000_000) * fps)
    for zone_start, zone_end in zones:
        if not _frames_hit(in_frame, out_frame, fps, zone_start, zone_end):
            continue
        start_frame = math.floor(Fraction(zone_start).limit_denominator(1_000_000) * fps)
        end_frame = math.ceil(Fraction(zone_end).limit_denominator(1_000_000) * fps)
        updated_in, updated_out = in_frame, out_frame
        if last_mid <= zone_start:
            updated_out = min(out_frame, start_frame)
        elif first_mid >= zone_end:
            updated_in = max(in_frame, end_frame)
        else:
            continue
        if updated_in > first_frame or updated_out <= last_frame or updated_in >= updated_out:
            continue
        in_frame, out_frame = _clamp_frames(updated_in, updated_out, duration_frames)
    return in_frame, out_frame


def _frames_hit(
    in_frame: int,
    out_frame: int,
    fps: Fraction,
    zone_start: float,
    zone_end: float,
) -> bool:
    played_in = Fraction(in_frame) / fps
    played_out = Fraction(out_frame) / fps
    start = Fraction(zone_start).limit_denominator(1_000_000)
    end = Fraction(zone_end).limit_denominator(1_000_000)
    return played_in < end and played_out > start


def _clamp_frames(in_frame: int, out_frame: int, duration_frames: int) -> tuple[int, int]:
    if duration_frames < 1:
        return 0, 1
    in_frame = min(max(in_frame, 0), duration_frames - 1)
    out_frame = min(max(out_frame, in_frame + 1), duration_frames)
    return in_frame, out_frame


def _merge_touching(
    ranges: list[Range],
    zones: dict[str, list[tuple[float, float]]],
    fps: Fraction,
) -> list[Range]:
    merged: list[Range] = []
    for rng in ranges:
        if (
            merged
            and merged[-1].source == rng.source
            and rng.in_frame - merged[-1].out_frame <= 1
            and not _joined_hits_zone(merged[-1], rng, zones, fps)
        ):
            merged[-1] = _merge_ranges(merged[-1], rng)
        else:
            merged.append(rng)
    return merged


def _joined_hits_zone(
    earlier: Range,
    later: Range,
    zones: dict[str, list[tuple[float, float]]],
    fps: Fraction,
) -> bool:
    """True when joining the two ranges would play a clap zone."""
    for zone_start, zone_end in zones.get(earlier.source, []):
        if earlier.in_s < zone_end and later.out_s > zone_start:
            return True
        if _frames_hit(earlier.in_frame, later.out_frame, fps, zone_start, zone_end):
            return True
    return False


def _merge_ranges(earlier: Range, later: Range) -> Range:
    return earlier.model_copy(
        update={
            "out_s": later.out_s,
            "out_frame": later.out_frame,
            "last_word": later.last_word,
            "markers": [*earlier.markers, *later.markers],
        }
    )


def _remove_tiny(
    ranges: list[Range],
    config: TightenConfig,
    zones: dict[str, list[tuple[float, float]]],
    fps: Fraction,
) -> list[Range]:
    """Merge a short range into a close same-source neighbour, or drop it with a CHECK marker."""
    max_gap_s = config.max_gap_ms / 1000
    pending: list[Marker] = []
    index = 0
    while index < len(ranges):
        current = ranges[index]
        if current.out_frame - current.in_frame >= config.min_range_frames:
            if pending:
                ranges[index] = current.model_copy(
                    update={"markers": [*pending, *current.markers]}
                )
                pending = []
            index += 1
            continue
        previous = ranges[index - 1] if index else None
        nxt = ranges[index + 1] if index + 1 < len(ranges) else None
        if (
            _within_gap(previous, current, max_gap_s)
            and previous is not None
            and not _joined_hits_zone(previous, current, zones, fps)
        ):
            ranges[index - 1] = _merge_ranges(previous, current)
            del ranges[index]
            continue
        if (
            _within_gap(current, nxt, max_gap_s)
            and nxt is not None
            and not _joined_hits_zone(current, nxt, zones, fps)
        ):
            ranges[index + 1] = _merge_ranges(current, nxt)
            del ranges[index]
            continue
        pending.append(Marker(at_word=current.first_word, text=_TINY_FRAGMENT))
        del ranges[index]
    if pending and ranges:
        last = ranges[-1]
        ranges[-1] = last.model_copy(update={"markers": [*last.markers, *pending]})
    return ranges


def _within_gap(earlier: Range | None, later: Range | None, max_gap_s: float) -> bool:
    if earlier is None or later is None or earlier.source != later.source:
        return False
    return later.in_s - earlier.out_s <= max_gap_s


def _attach_flags(ranges: list[Range], decisions: list[Decision]) -> None:
    for decision in decisions:
        if not decision.flag:
            continue
        _attach_marker(
            ranges,
            Marker(
                at_word=decision.kept_from_word,
                text=f"CHECK: {decision.flag_reason} ({decision.id})",
            ),
        )


def _mark_unscripted(
    ranges: list[Range],
    alignment: AlignmentData | None,
    by_index: dict[int, Word],
    dropped_ids: set[int],
    script: ScriptConfig | None,
) -> None:
    if alignment is None or script is None:
        return
    for span in alignment.unscripted:
        duration = _span_duration(span.first_word, span.last_word, by_index)
        if duration is None or duration < script.flag_unscripted_s:
            continue
        kept_word = _first_kept_between(span.first_word, span.last_word, dropped_ids, by_index)
        if kept_word is None:
            continue
        _attach_marker(ranges, Marker(at_word=kept_word, text=_UNSCRIPTED))


def _span_duration(first_i: int, last_i: int, by_index: dict[int, Word]) -> float | None:
    first = by_index.get(first_i)
    last = by_index.get(last_i)
    if first is None or last is None:
        return None
    if first.source == last.source:
        return last.end - first.start
    total = 0.0
    for index in range(first_i, last_i + 1):
        word = by_index.get(index)
        if word is not None:
            total += word.end - word.start
    return total


def _mark_unheard(
    ranges: list[Range],
    speech: list[SpeechSegment],
    words: list[Word],
    vad: VadConfig | None,
) -> None:
    """Flag a long speech segment that contains no transcript word. It is not cut in."""
    if vad is None or not speech:
        return
    for segment in speech:
        if segment.end - segment.start < vad.flag_unheard_speech_s:
            continue
        if any(
            word.source == segment.source and segment.start <= _midpoint(word) <= segment.end
            for word in words
        ):
            continue
        _attach_unheard(ranges, segment)


def _attach_unheard(ranges: list[Range], segment: SpeechSegment) -> None:
    same = [rng for rng in ranges if rng.source == segment.source]
    if not same:
        return
    following = [rng for rng in same if rng.in_s >= segment.end]
    if following:
        target = following[0]
        at_word = target.first_word
    else:
        target = same[-1]
        at_word = target.last_word
    target.markers.append(Marker(at_word=at_word, text=_UNHEARD))


def _mark_claps_inside(
    ranges: list[Range],
    zones: dict[str, list[tuple[float, float]]],
    fps: Fraction,
) -> None:
    if not zones:
        return
    for rng in ranges:
        hits = [
            zone
            for zone in zones.get(rng.source, [])
            if _range_hits_zone(rng, zone, fps)
        ]
        if not hits:
            continue
        rng.markers.append(Marker(at_word=rng.first_word, text=_CLAP_INSIDE))
        logger.warning(
            "clap zone stays inside %s %.3f-%.3f; %s",
            rng.source,
            rng.in_s,
            rng.out_s,
            _CLAP_INSIDE,
        )


def _range_hits_zone(rng: Range, zone: tuple[float, float], fps: Fraction) -> bool:
    zone_start, zone_end = zone
    if rng.in_s < zone_end and rng.out_s > zone_start:
        return True
    return _frames_hit(rng.in_frame, rng.out_frame, fps, zone_start, zone_end)


def _chapters(
    alignment: AlignmentData | None,
    chapters: ChaptersConfig | None,
    dropped_ids: set[int],
    by_index: dict[int, Word],
) -> list[TimelineChapter]:
    """Chapter markers at the first kept word of the chosen take.

    The chapter's first sentence wins. When that sentence has no chosen take,
    the first later sentence in the chapter that has one is used. A chapter
    with none is skipped.
    """
    if alignment is None or chapters is None or not chapters.enabled:
        return []
    by_chapter: dict[str, list] = {}
    for sentence in alignment.sentences:
        if sentence.chapter is None:
            continue
        by_chapter.setdefault(sentence.chapter, []).append(sentence)
    markers: list[TimelineChapter] = []
    for chapter in alignment.chapters:
        in_chapter = sorted(by_chapter.get(chapter.id, []), key=lambda item: item.id)
        primary = [item for item in in_chapter if item.id == chapter.first_sentence]
        rest = [item for item in in_chapter if item.id != chapter.first_sentence]
        at_word: int | None = None
        for sentence in [*primary, *rest]:
            chosen = next((take for take in sentence.takes if take.chosen), None)
            if chosen is None:
                continue
            at_word = _first_kept_between(
                chosen.first_word,
                chosen.last_word,
                dropped_ids,
                by_index,
            )
            if at_word is not None:
                break
        if at_word is None:
            logger.warning(
                "chapter %s (%s) has no chosen take with a kept word; skipping",
                chapter.id,
                chapter.title,
            )
            continue
        markers.append(TimelineChapter(id=chapter.id, title=chapter.title, at_word=at_word))
    return markers


def _first_kept_between(
    first_i: int,
    last_i: int,
    dropped_ids: set[int],
    by_index: dict[int, Word],
) -> int | None:
    for index in range(first_i, last_i + 1):
        if index in dropped_ids or index not in by_index:
            continue
        return index
    return None


def _attach_marker(ranges: list[Range], marker: Marker) -> None:
    if not ranges:
        return
    for rng in ranges:
        if rng.first_word <= marker.at_word <= rng.last_word:
            rng.markers.append(marker)
            return
    for rng in ranges:
        if rng.first_word > marker.at_word:
            rng.markers.append(marker)
            return
    ranges[-1].markers.append(marker)


def _dropped_spans(decisions: list[Decision], by_index: dict[int, Word]) -> list[DroppedSpan]:
    spans: list[DroppedSpan] = []
    for decision in decisions:
        if decision.action != "drop":
            continue
        first_i, last_i = decision.dropped_words
        first = by_index[first_i]
        last = by_index[last_i]
        spans.append(
            DroppedSpan(
                source=first.source,
                in_s=first.start,
                out_s=last.end,
                decision=decision.id,
            )
        )
    return spans


def _analysis_wav(project_dir: Path, source: Source) -> Path:
    path = Path(source.analysis_wav)
    if path.is_absolute():
        return path
    return project_dir / path


def _midpoint(word: Word) -> float:
    return (word.start + word.end) / 2
