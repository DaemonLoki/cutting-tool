"""Speech segments and claps for one project.

Voice activity comes from ``vad.py``. Claps are detected on each 48 kHz
analysis WAV and dropped when their onset falls inside speech.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.claps import detect_claps
from cutter.config import Profile, load_profile
from cutter.models import (
    AudioEventsArtifact,
    AudioEventsData,
    Clap,
    Source,
    SourcesArtifact,
    SpeechSegment,
    make_meta,
    write_artifact,
)
from cutter.vad import speech_segments

STAGE = "audio"
STAGE_VERSION = 3


def run_audio(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> AudioEventsArtifact:
    """Stage entry for ``cutter audio <project_dir> [--profile] [--force]``.

    Reads ``artifacts/sources.json`` and each source's 16 kHz and 48 kHz WAVs.
    Writes ``artifacts/audio_events.json``. A cache hit returns the artifact
    already on disk. When VAD is disabled the backend is ``none`` and speech
    is empty. Claps are filled in when ``claps.enabled`` is true.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    sources_path = artifacts / "sources.json"
    events_path = artifacts / "audio_events.json"
    if not sources_path.is_file():
        raise FileNotFoundError(f"missing sources artifact: {sources_path}")

    sources = SourcesArtifact.model_validate_json(sources_path.read_text(encoding="utf-8"))
    wavs = _wavs(project_dir, sources.data.sources)
    hashed_inputs = inputs_hash(artifacts=[sources_path, *wavs])
    hashed_config = _config_hash(loaded)
    if not force and cache_hit(
        events_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return AudioEventsArtifact.model_validate_json(events_path.read_text(encoding="utf-8"))

    if loaded.vad.enabled:
        backend = loaded.vad.backend
        speech = _speech(project_dir, sources.data.sources, loaded)
    else:
        backend = "none"
        speech = []
    # Claps use the speech segments already in `speech` (empty when VAD is off).
    claps: list[Clap] = []
    if loaded.claps.enabled:
        speech_by_source: dict[str, list[tuple[float, float]]] = {}
        for segment in speech:
            speech_by_source.setdefault(segment.source, []).append((segment.start, segment.end))
        for source in sources.data.sources:
            hits = detect_claps(
                project_dir / source.analysis_wav,
                speech_by_source.get(source.id, []),
                loaded.claps,
            )
            claps.extend(
                Clap(source=source.id, t=hit.t, peak_db=hit.peak_db, rise_db=hit.rise_db)
                for hit in hits
            )
        claps.sort(key=lambda clap: (clap.source, clap.t))
    artifact = AudioEventsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=AudioEventsData(backend=backend, speech=speech, claps=claps),
    )
    write_artifact(events_path, artifact)
    return artifact


def _config_hash(profile: Profile) -> str:
    """Hash the audio sections, plus the margin the energy backend reads.

    ``pad_head_ms`` and the rest of ``tighten`` stay out of this hash. Only
    ``voice_margin_db`` changes an energy segmentation, and only then.
    """
    hashed = config_hash(profile, STAGE_CONFIG_SECTIONS[STAGE])
    if not profile.vad.enabled or profile.vad.backend != "energy":
        return hashed
    payload = f"{hashed}\0{profile.tighten.voice_margin_db}"
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _speech(project_dir: Path, sources: list[Source], profile: Profile) -> list[SpeechSegment]:
    margin = profile.tighten.voice_margin_db
    segments: list[SpeechSegment] = []
    for source in sources:
        wav_name = source.asr_wav if profile.vad.backend == "silero" else source.analysis_wav
        spans = speech_segments(project_dir / wav_name, profile.vad, voice_margin_db=margin)
        segments.extend(
            SpeechSegment(source=source.id, start=start, end=end) for start, end in spans
        )
    return segments


def _wavs(project_dir: Path, sources: list[Source]) -> list[Path]:
    wavs: list[Path] = []
    for source in sources:
        asr = project_dir / source.asr_wav
        analysis = project_dir / source.analysis_wav
        if not asr.is_file():
            raise FileNotFoundError(f"missing asr wav for {source.id}: {asr}")
        if not analysis.is_file():
            raise FileNotFoundError(f"missing analysis audio: {analysis}")
        wavs.extend((asr, analysis))
    return wavs
