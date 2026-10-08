"""Cut kept words into timeline ranges snapped to silence.

A decision with ``action: drop`` removes its words. Each remaining run becomes
one range. Flagged decisions leave a CHECK marker for Final Cut.

Parakeet often stretches the last word of a sentence across the pause after
it. A range ends where the voice stops, measured on the audio, not where the
transcript says the word ends.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from cutter.audio import rms_envelope, snap_time, voice_end, voice_levels
from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, TightenConfig, load_profile
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DroppedSpan,
    Marker,
    Range,
    Source,
    SourcesArtifact,
    SourcesData,
    TimelineArtifact,
    TimelineData,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)

STAGE = "tighten"
STAGE_VERSION = 3

_TINY_FRAGMENT = "CHECK: tiny fragment removed"
_SENTENCE_END = (".", "?", "!")


def build_timeline(
    words: list[Word],
    decisions: list[Decision],
    sources: SourcesData,
    envelopes: dict[str, tuple[np.ndarray, int]],
    config: TightenConfig,
) -> TimelineData:
    """Build playback ranges and the dropped-span list.

    ``envelopes`` maps source id to ``(rms, sample_rate)``. A missing envelope
    does not snap; the raw cut point is kept.
    """
    by_index = {word.i: word for word in words}
    by_source = _words_by_source(words)
    source_by_id = {source.id: source for source in sources.sources}
    fps = Fraction(sources.fps)
    dropped_ids = _dropped_ids(decisions)
    kept = [word for word in sorted(words, key=lambda word: word.i) if word.i not in dropped_ids]
    voices = {
        source_id: voice_levels(samples, config.rms_frame_ms)
        for source_id, (samples, _rate) in envelopes.items()
    }
    sentence_ends = {
        word.i: _spoken_end(word, voices, config)
        for word in kept
        if word.w.endswith(_SENTENCE_END)
    }
    ranges = [
        _range_for_group(group, by_source, source_by_id, envelopes, voices, config, fps)
        for group in _group_kept(kept, dropped_ids, config.max_gap_ms / 1000, sentence_ends)
    ]
    ranges = _merge_touching(ranges)
    ranges = _remove_tiny(ranges, config)
    _attach_flags(ranges, decisions)
    numbered = [
        rng.model_copy(update={"id": f"r{number:03d}"})
        for number, rng in enumerate(ranges, start=1)
    ]
    return TimelineData(ranges=numbered, dropped=_dropped_spans(decisions, by_index))


def run_tighten(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> TimelineArtifact:
    """Stage entry for ``cutter tighten <project_dir> [--force]``.

    Reads words, decisions, sources, and each source's analysis wav. Writes
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

    hashed_inputs = inputs_hash(
        artifacts=[words_path, decisions_path, sources_path, *wav_paths]
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
        data=build_timeline(words, decisions, sources_artifact.data, envelopes, loaded.tighten),
    )
    write_artifact(timeline_path, artifact)
    return artifact


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


def _range_for_group(
    group: list[Word],
    by_source: dict[str, list[Word]],
    source_by_id: dict[str, Source],
    envelopes: dict[str, tuple[np.ndarray, int]],
    voices: dict[str, tuple[np.ndarray, float]],
    config: TightenConfig,
    fps: Fraction,
) -> Range:
    first = group[0]
    last = group[-1]
    source = source_by_id[first.source]
    previous, following = _neighbours(by_source.get(first.source, []), first.i, last.i)
    in_s, out_s = _cut_points(
        first,
        last,
        _spoken_end(last, voices, config),
        previous,
        following,
        source.duration_s,
        envelopes,
        config,
    )
    in_frame, out_frame = _frames(in_s, out_s, fps, source.duration_frames)
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
) -> tuple[float, float]:
    """Pad, clamp to the file and the neighbouring word, then snap inside the legal interval.

    ``last_end`` is where the voice of ``last`` stops. The out-point never
    snaps later than ``pad_tail_ms`` after it, so a range does not end with
    more silence than the pad.
    """
    raw_in = _clamp(first.start - config.pad_head_ms / 1000, 0.0, duration_s)
    if previous is not None and previous.end <= first.start:
        raw_in = _clamp(max(raw_in, previous.end), 0.0, duration_s)
    raw_out = _clamp(last_end + config.pad_tail_ms / 1000, 0.0, duration_s)
    if following is not None:
        raw_out = _clamp(min(raw_out, following.start), 0.0, duration_s)

    in_lo = 0.0 if previous is None else max(0.0, previous.end)
    in_hi = min(duration_s, first.start - config.min_head_ms / 1000)
    out_lo = max(0.0, last_end + config.min_tail_ms / 1000)
    out_hi = duration_s if following is None else min(duration_s, following.start)
    out_hi = min(out_hi, raw_out)
    envelope = envelopes.get(first.source)
    return (
        _snap(envelope, raw_in, in_lo, in_hi, config),
        _snap(envelope, raw_out, out_lo, out_hi, config),
    )


def _snap(
    envelope: tuple[np.ndarray, int] | None,
    raw_s: float,
    earliest_s: float,
    latest_s: float,
    config: TightenConfig,
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


def _clamp_frames(in_frame: int, out_frame: int, duration_frames: int) -> tuple[int, int]:
    if duration_frames < 1:
        return 0, 1
    in_frame = min(max(in_frame, 0), duration_frames - 1)
    out_frame = min(max(out_frame, in_frame + 1), duration_frames)
    return in_frame, out_frame


def _merge_touching(ranges: list[Range]) -> list[Range]:
    merged: list[Range] = []
    for rng in ranges:
        if (
            merged
            and merged[-1].source == rng.source
            and rng.in_frame - merged[-1].out_frame <= 1
        ):
            merged[-1] = _merge_ranges(merged[-1], rng)
        else:
            merged.append(rng)
    return merged


def _merge_ranges(earlier: Range, later: Range) -> Range:
    return earlier.model_copy(
        update={
            "out_s": later.out_s,
            "out_frame": later.out_frame,
            "last_word": later.last_word,
            "markers": [*earlier.markers, *later.markers],
        }
    )


def _remove_tiny(ranges: list[Range], config: TightenConfig) -> list[Range]:
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
        if _within_gap(previous, current, max_gap_s):
            assert previous is not None
            ranges[index - 1] = _merge_ranges(previous, current)
            del ranges[index]
            continue
        if _within_gap(current, nxt, max_gap_s):
            assert nxt is not None
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
