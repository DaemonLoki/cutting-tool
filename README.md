# cutter — Phase 1 Implementation Spec

Phase 1 is implemented. Phase 2 (VAD, claps, fillers, script alignment, chapters, CFR proxy) is specified in `docs/phase-2.md`, which builds on this document.

**Goal:** A local Python CLI that takes a folder of numbered raw sources, transcribes them, removes failed takes using the transcript only, tightens the cuts, and writes an FCPXML file that Final Cut Pro imports as an editable rough cut.

**Audience:** Coding agents. Every section is meant to be implementable without further context. Where something must be verified against a real tool (FCP, parakeet-mlx), the spec says so explicitly. Do not guess in those places; verify.

---

## 0. Ground rules for implementers

1. **Never re-encode source video.** The FCPXML references the original files. ffmpeg is only used to extract audio.
2. **Bias against false cuts.** Cutting good content is far worse than leaving a duplicate in. When uncertain: keep both, add a `CHECK` marker.
3. **All FCPXML time math uses `fractions.Fraction`.** Analysis code may use float seconds; conversion to frames happens once, in the export stage.
4. **No hard-coded thresholds.** Every number in this spec marked `[cfg]` lives in the profile YAML.
5. **Each stage is pure:** reads artifacts from disk, writes one artifact, no hidden state.
6. **Verify, don't assume,** for: parakeet-mlx return types, FCPXML attributes (mirror `test/fixtures/reference.fcpxmld/Info.fcpxml`), the FCP DTD location.

---

## 1. Scope

### In scope

- Ingest numbered sources for one output video, validate formats, extract audio
- Word-level transcription of English speech (local)
- Text-only retake detection
- Optional local-LLM judge for ambiguous aborted retakes
- Tightening: gap removal, padding, snap-to-silence, frame rounding
- FCPXML export: rough-cut project + rejects project + `CHECK` markers
- Stage caching, CLI, and an eval command against a manually edited gold project

One project is one output video and lives in its own folder. Phase 1 speech is English. A `CHECK` marker is resolved by the editor in Final Cut; cutter has no accept/reject command.

### Out of scope (later phases, do not build)

Clap detection, script alignment, keywords per script paragraph, chapter markers, punch-ins, filler removal, VAD, audio volume ramps, Remotion, short-form videos, languages other than English. Interviews, multicam, and b-roll assemblies are out.

---

## 2. Prerequisites (provided by Stefan before work starts)

| Item | Path | Purpose |
| --- | --- | --- |
| Reference export from FCP | `test/fixtures/reference.fcpxmld/Info.fcpxml` | Full library export. Source of truth for `version`, `<format>` and camera `<asset>` attributes. The timeline also contains screen recordings, titles, color, and transitions; do not reproduce that timeline. |
| Sample raw sources | `test/fixtures/sample/raw/` | `1-intro.MP4` and `2-skills.MP4`, the camera sources for the gold edit. |
| Gold project | `test/gold/frontend-skills/manual.fcpxmld/Info.fcpxml` | A cut made only from those two sources. `src` attributes point at `~/Downloads/sample-videos/Videos/`; resolve them to the repo copies by filename. Used by `cutter eval`. |
| Machine | Apple Silicon, macOS, ffmpeg via Homebrew, LM Studio optional | |

Locate the DTD with:

```bash
find "/Applications/Final Cut Pro.app" -iname "*.dtd" | grep -i fcpxml
```

The reference file is FCPXML 1.14. Its DTD is `/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd`. Store that path in config (`fcpxml.dtd_path`).

---

## 3. Tech stack

- Python 3.12, managed with `uv`
- `typer` (CLI), `pydantic` v2 (models + config), `pyyaml`, `rich` (output)
- `natsort`, `rapidfuzz`, `numpy`, `soundfile`, `lxml`
- `parakeet-mlx` (ASR), `openai` (OpenAI-compatible client for LM Studio)
- ffmpeg / ffprobe via `subprocess` (no Python ffmpeg wrappers)
- Tests: `pytest`, `hypothesis`

Do not add librosa, moviepy, OpenTimelineIO or any video-processing library.

---

## 4. Repository layout

Paths below are relative to this repo root. `projects/` is local working data and is gitignored. The checked-in fixtures and gold edit live under `test/`, not `tests/`. Pytest modules live under `tests/`.

```
pyproject.toml
profiles/
  long.yaml
cutter/
  __init__.py
  cli.py            # Typer app, stage orchestration, caching
  config.py         # Pydantic config models, profile loading
  models.py         # Pydantic models for all artifacts
  cache.py          # input/config hashing, skip logic
  ingest.py
  transcribe.py
  retakes.py
  judge.py
  tighten.py
  timeline.py
  fcpxml.py
  fcpxml_parse.py   # for eval only
  evaluate.py
  llm.py
  audio.py          # RMS envelope helpers
tests/
  fixtures/
  gold/
  unit/
  e2e/
projects/<name>/    # gitignored
  raw/              # input sources: 01.mov, 02.mov, …
  artifacts/        # stage outputs (JSON + WAVs)
  out/              # <name>.fcpxml
```

---

## 5. Configuration (`profiles/long.yaml`)

```yaml
ingest:
  allowed_extensions: [".mov", ".mp4", ".m4v"]
  asr_sample_rate: 16000
  analysis_sample_rate: 48000

transcribe:
  model: "mlx-community/parakeet-tdt-0.6b-v2"
  chunk_duration_s: 120
  overlap_duration_s: 15

retakes:
  window_words: 6            # words compared at a retake candidate
  min_match_words: 3         # minimum consecutive matching words
  word_similarity: 85        # rapidfuzz ratio per word, 0–100
  max_mismatches: 1          # tolerated mismatched words inside the match
  max_lookback_s: 90
  auto_drop_max_s: 20        # longer failed segments are flagged, not auto-dropped
  missing_content_ratio: 0.3 # flag if dropped segment has this share of content words the retake lacks
  stopwords_file: null       # null = built-in English list

judge:
  enabled: true
  base_url: "http://localhost:1234/v1"
  model: "qwen3-14b"         # whatever is loaded in LM Studio
  min_confidence: 0.7
  timeout_s: 60

tighten:
  max_gap_ms: 400            # gaps longer than this get cut
  pad_head_ms: 80
  pad_tail_ms: 120
  snap_window_ms: 150
  min_head_ms: 30            # in-point never closer than this to first word
  min_tail_ms: 40            # out-point never closer than this to last word
  rms_frame_ms: 10
  min_range_frames: 6        # shorter ranges are merged into neighbours or dropped with a CHECK marker

fcpxml:
  version: "1.14"            # must match test/fixtures/reference.fcpxmld/Info.fcpxml
  dtd_path: "/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd"
  event_name: "{project} – cutter"
  project_name: "{project} – rough cut"
  rejects_project_name: "{project} – rejects"
  audio_role: "dialogue"
```

Load into Pydantic models with `extra="forbid"`. CLI flag `--set key.path=value` overrides single values (useful for eval sweeps).

---

## 6. Data models (`models.py`)

All artifacts are JSON with a common envelope:

```json
{
  "meta": {
    "stage": "transcribe",
    "stage_version": 1,
    "inputs_hash": "sha256:…",
    "config_hash": "sha256:…",
    "created_at": "2026-10-06T12:00:00Z"
  },
  "data": { }
}
```

Times in analysis artifacts are **float seconds from the start of the source file** (not timecode).

### 6.1 `artifacts/sources.json`

```json
{
  "fps": "30000/1001",
  "width": 3840,
  "height": 2160,
  "audio_rate": 48000,
  "audio_channels": 2,
  "sources": [
    {
      "id": "s01",
      "path": "/abs/path/raw/01.mov",
      "duration_s": 312.312,
      "duration_frames": 9360,
      "start_timecode": "00:00:00:00",
      "start_frames": 0,
      "vfr_warning": false,
      "asr_wav": "artifacts/audio/s01.16k.wav",
      "analysis_wav": "artifacts/audio/s01.48k.wav"
    }
  ]
}
```

### 6.2 `artifacts/words.json`

```json
{
  "words": [
    {"i": 0, "source": "s01", "w": "So,", "norm": "so", "start": 1.20, "end": 1.38, "conf": null, "sent": 0}
  ]
}
```

- `i` is a global index across all sources, in source order.
- `norm`: lowercase, punctuation stripped, Unicode NFKC, digits kept.
- `sent`: global sentence index, from punctuation in the ASR output.
- `conf` is nullable; do not depend on it in Phase 1.

### 6.3 `artifacts/decisions.json`

```json
{
  "decisions": [
    {
      "id": "d001",
      "kind": "retake",
      "dropped_words": [120, 141],
      "kept_from_word": 142,
      "match_words": 5,
      "dropped_duration_s": 7.4,
      "action": "drop",
      "flag": false,
      "flag_reason": null,
      "judge": null
    }
  ]
}
```

- `dropped_words` is an inclusive `[first, last]` word index range.
- `action`: `"drop"` (words removed) or `"keep"` (both takes kept).
- `flag: true` always produces a `CHECK` marker in the export.
- `judge`: `null` or `{"choice": "A"|"B"|"both", "confidence": 0.82, "reason": "…", "model": "…"}`.

### 6.4 `artifacts/timeline.json`

```json
{
  "ranges": [
    {
      "id": "r001",
      "source": "s01",
      "in_s": 1.12,
      "out_s": 18.91,
      "in_frame": 33,
      "out_frame": 567,
      "first_word": 0,
      "last_word": 58,
      "markers": [{"at_word": 12, "text": "CHECK: low-confidence retake (d003)"}]
    }
  ],
  "dropped": [
    {"source": "s01", "in_s": 18.91, "out_s": 26.3, "decision": "d001"}
  ]
}
```

- `in_frame`/`out_frame` are frame indices relative to the start of the source file. `out_frame` is exclusive.
- `ranges` are in playback order: source order, then time.

---

## 7. Stages

### 7.1 Ingest (`ingest.py`)

**Input:** `projects/<name>/raw/`. **Output:** `sources.json`, WAV files.

1. List files with allowed extensions, ignore hidden files, sort with `natsort.natsorted`. Assign ids `s01, s02, …` in that order.
2. Run `ffprobe -v error -print_format json -show_streams -show_format` per file. Extract:
   - video stream: `r_frame_rate`, `avg_frame_rate`, `width`, `height`, `nb_frames` if present
   - audio stream (first): `sample_rate`, `channels`
   - timecode: `tags.timecode` on the video stream, else on a `tmcd` data stream, else on format tags, else `00:00:00:00`
   - duration: from the video stream; fall back to format
3. **Validation (hard errors, clear messages):**
   - no audio stream
   - `r_frame_rate`, width, height or audio sample rate differ between files
   - drop-frame timecode (contains `;`): raise "drop-frame timecode not supported in Phase 1"
4. **Warnings:** `avg_frame_rate != r_frame_rate` → `vfr_warning: true` (typical for phone footage; FCP may conform it differently).
5. `start_frames` = timecode converted to frames at the **nominal** rate (30 for 30000/1001, 24 for 24000/1001, otherwise round(fps)).
6. `duration_frames` = `floor(duration_s * fps)`.
7. Extract audio from the first audio stream:
   ```bash
   ffmpeg -nostdin -y -i in.mov -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le artifacts/audio/s01.16k.wav
   ffmpeg -nostdin -y -i in.mov -map 0:a:0 -ac 1 -ar 48000 -c:a pcm_s16le artifacts/audio/s01.48k.wav
   ```

**Acceptance:** `10.mov` sorts after `2.mov`; mismatched fps raises; the VFR warning is printed; timecode `01:00:00:00` at 29.97 NDF gives `start_frames = 108000`.

### 7.2 Transcribe (`transcribe.py`)

**Input:** 16 kHz WAVs. **Output:** `words.json`.

1. Define a `Transcriber` protocol: `transcribe(wav_path) -> list[RawWord]` with `RawWord(text, start, end, conf | None)`.
2. Implement `ParakeetTranscriber` using `parakeet-mlx` with chunking from config.
   - **Verify the actual API and result structure** of the installed `parakeet-mlx` version before writing code (inspect the package, run it on a 10 s file, print the result). Parakeet emits subword tokens: merge them into words (a token with a leading space or SentencePiece `▁` starts a new word). Word start = first token start, end = last token end.
   - Keep punctuation in `w`; it drives sentence segmentation.
3. Sentence segmentation: a new sentence starts after a word ending in `.`, `?` or `!`, and at every source boundary.
4. Assign global `i` across sources in source order.
5. Sanity checks: monotonic non-decreasing starts within a source, `end >= start`, no word outside `[0, duration_s]`. Clamp and log tiny violations (< 50 ms), raise on larger ones.

**Acceptance:** On the sample source, word timestamps visually match the waveform within ~100 ms on 10 spot checks (write a small script that prints word, start, end for a range so this can be checked by ear in QuickTime).

### 7.3 Retake detection (`retakes.py`)

**Input:** `words.json`. **Output:** `decisions.json`.

A *retake* is when the speaker aborts a sentence and starts it again. The aborted failed take is dropped; the later take is kept ("last take wins"). A finished sentence that is said again is not a retake: it stays in the rough cut, with a `CHECK` marker.

**Algorithm**

```
dropped = set()
for i in range(n):                                   # left to right
    if i in dropped: continue
    best = None
    for j in candidates(i):                          # nearest first, see below
        m, mismatches = match_length(j, i)
        if m >= min_match_words and mismatches <= max_mismatches and has_content_word(j, m):
            best = (j, m); break
    if best:
        j, m = best
        record decision: dropped_words = [j, i-1], kept_from_word = i, match_words = m
        apply guards (below); if action == drop: dropped |= {j..i-1}
```

- `candidates(i)`: indices `j` with `j < i`, same source as `i`, `j not in dropped`, `words[i].start - words[j].start <= max_lookback_s`, ordered **nearest first** (descending `j`).
- `match_length(j, i)`: compare `norm` of `words[j+t]` and `words[i+t]` for `t = 0, 1, …` while `t < window_words` and `j + t < i`. A word matches if `rapidfuzz.fuzz.ratio >= word_similarity`. Count consecutive matches, allowing up to `max_mismatches` mismatches that are followed by a match. Stop at the first unrecoverable mismatch. Return the match count.
- `has_content_word(j, m)`: at least one matched word is not a stopword, **or** `m >= 5`. Prevents matching "and the" / "so the".
- Retakes never cross a source boundary in the main scan. A failed take at the end of `01.mov` and its retake in `02.mov` is handled separately: see *cross-file retakes* below.

A candidate is auto-dropped only when the earlier span is aborted: none of its words end in `.`, `?`, or `!`. Apply the guards in order; the first match wins.

1. The earlier span contains a sentence end → `action: keep`, `flag: true`, reason `complete_sentence`. The finished sentence stays. This reason is final: the judge must not change the action to `drop`.
2. `dropped_duration_s > auto_drop_max_s` → `action: keep`, `flag: true`, reason `long_segment`. A long aborted span is often a legitimate repetition.
3. Missing content, only reached for aborted spans: let `D` = content words (non-stopwords, normalized) in the dropped segment and `K` = content words in `words[i : i + len(dropped) + 10]`. If `|D − K| / |D| > missing_content_ratio` → `action: drop`, `flag: true`, reason `retake_missing_content`. The retake may have left something out.
4. Otherwise → `action: drop`, `flag: false`.

**Cross-file retakes**

If the first `window_words` words of source `k+1` match (same rules) a word run starting at `j` in the last `max_lookback_s` of source `k`, record a decision on `[j, last word of source k]`. Same guards apply. This covers the common workflow "messed up, stopped recording, started a new file".

**Acceptance (unit tests with synthetic word lists, no audio needed)**

| Case | Words | Expected |
| --- | --- | --- |
| Simple retake | `the agent joins the — the agent joins the call` | drop first 4 |
| Short aborted take | `so the agent so the agent joins the call` | drop first 3 |
| Chain of 3 aborted takes | takes A, B, C of the same words, no sentence end | keep C only, two decisions |
| Stopword-only repeat | `and the … and the server` | no decision |
| Legit repetition after 2 min | same sentence 120 s apart | no decision (lookback) |
| Long aborted segment | aborted span longer than 20 s | `keep`, flagged `long_segment` |
| Finished sentence | `A neuron is a weighted vote. A neuron is a weighted vote.` | `keep`, flagged `complete_sentence` |
| Cross-file | `01` ends with an aborted sentence, `02` starts with it | drop tail of `01` |
| Finished sentence, cross-file | `01` ends with a sentence, `02` starts by repeating it | `keep`, flagged `complete_sentence` |

### 7.4 Judge (`judge.py`, `llm.py`)

**Input:** `decisions.json` (flagged ones), `words.json`. **Output:** `decisions.json` updated in place (new artifact version).

- Runs only on ambiguous aborted takes: `flag_reason` of `long_segment` or `retake_missing_content`, and only if `judge.enabled` and the endpoint answers. It may turn those into a drop. `complete_sentence` is never sent to the judge and stays `action: keep` and flagged. Any decision still flagged after this stage is resolved in Final Cut. If the endpoint is unreachable: log a warning once, leave decisions unchanged.
- `llm.py`: thin wrapper around `openai.OpenAI(base_url=…, api_key="lm-studio")` with one method `complete_json(system, user, schema) -> dict`. Use `response_format={"type": "json_schema", …}`. If the server rejects structured output, retry once with plain JSON instructions and parse; on parse failure return `None`.
- Prompt input: up to 2 sentences before, segment **A** (dropped candidate), segment **B** (retake onward, same length + 1 sentence), 1 sentence after. Plain text only, no timestamps.
- System prompt: load from `skills/retake-judging/SKILL.md` if present, else a built-in default. The default must explain: A is a candidate failed attempt, B is the candidate retake; answer `"B"` if A should be dropped, `"A"` if B is the failed one and A should be kept instead, `"both"` if it's not a retake.
- Output schema:
  ```json
  {"type": "object", "required": ["choice", "confidence", "reason"],
   "properties": {
     "choice": {"enum": ["A", "B", "both"]},
     "confidence": {"type": "number", "minimum": 0, "maximum": 1},
     "reason": {"type": "string", "maxLength": 300}}}
  ```
- Applying the verdict:
  - `confidence < min_confidence` → `action: keep`, stays flagged.
  - `"B"` → `action: drop`, unflag.
  - `"both"` → `action: keep`, unflag.
  - `"A"` → drop the retake segment instead (B's matching span up to where it diverges from A is not well defined, so: `action: keep`, stays flagged with reason `judge_prefers_first_take`). Do not attempt automatic reverse cuts in Phase 1.
- Store the raw verdict in `decision.judge`.

**Acceptance:** With a mocked LLM client, each verdict path produces the expected action and flag. A `complete_sentence` decision stays kept and flagged even when the mock returns `"B"`. With LM Studio stopped, the stage completes and logs one warning.

### 7.5 Tighten (`tighten.py`, `audio.py`)

**Input:** `words.json`, `decisions.json`, `sources.json`, 48 kHz WAVs. **Output:** `timeline.json`.

1. **Kept words** = all words not covered by a decision with `action: drop`.
2. **Split into ranges:** walk kept words in order; start a new range when the source changes, when a dropped word lies between two kept words, or when the gap `next.start − prev.end > max_gap_ms`.
3. **Raw cut points:** `in = first.start − pad_head`, `out = last.end + pad_tail`. Clamp to `[0, duration_s]`, and never past the neighbouring word (previous word's end / next word's start, kept or dropped).
4. **Snap to silence:** compute an RMS envelope (`rms_frame_ms` windows) of the 48 kHz WAV. Within `±snap_window_ms` of each raw cut point, pick the lowest-RMS position, subject to:
   - in-point ≤ `first.start − min_head_ms` and ≥ previous word's end
   - out-point ≥ `last.end + min_tail_ms` and ≤ next word's start
   If the allowed interval is empty, keep the raw cut point.
5. **Frame rounding (once, last):** `in_frame = floor(in_s × fps)`, `out_frame = ceil(out_s × fps)`, using `Fraction(fps)`.
6. **Cleanup:** merge consecutive ranges of the same source whose frame gap is ≤ 1. Ranges shorter than `min_range_frames`: merge into the neighbour if the gap is ≤ `max_gap_ms`, otherwise drop the range and attach a `CHECK: tiny fragment removed` marker to the next range.
7. **Markers:** each flagged decision attaches a marker to the range containing `kept_from_word` (or the nearest following range). Text: `CHECK: <flag_reason> (<decision id>)`.
8. **Dropped list:** every dropped decision becomes an entry in `dropped` with its source time span (word boundaries, not padded).

**Acceptance**

- Property test (hypothesis, synthetic words + random RMS): no range contains a dropped word's midpoint; all ranges have `in_frame < out_frame`; no two ranges of the same source overlap; every kept word's midpoint lies inside some range.
- On the sample source: listen to 20 random cuts in FCP, no clipped word starts or ends.

### 7.6 FCPXML export (`fcpxml.py`)

**Input:** `timeline.json`, `sources.json`. **Output:** `out/<project>.fcpxml`.

**Before writing code:** open `test/fixtures/reference.fcpxmld/Info.fcpxml` and mirror its `fcpxml version`, the camera `<format>` attributes (including `name` and `colorSpace`) and the camera `<asset>` attributes. Where this spec and the reference disagree, the reference wins. Do not reproduce the reference timeline's screen recordings, titles, color, or transitions.

**Time conversion**

```python
frame_dur = Fraction(fps).denominator / Fraction(fps).numerator   # e.g. 1001/30000
def t(frames: int) -> str:
    v = Fraction(frames) * frame_dur
    return "0s" if v == 0 else f"{v.numerator}/{v.denominator}s"
```

Every time attribute written is `t(<integer frame count>)`. Never write a float.

**Structure**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="r1" name="FFVideoFormat3840x2160p2398" frameDuration="1001/24000s"
            width="3840" height="2160" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="r2" name="01" start="0s" duration="9369360/30000s"
           hasVideo="1" hasAudio="1" format="r1"
           audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///Users/stefan/projects/demo/raw/01.mov"/>
    </asset>
  </resources>
  <library>
    <event name="demo – cutter">
      <project name="demo – rough cut">
        <sequence format="r1" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="r2" name="01" offset="0s" start="33033/30000s"
                        duration="534534/30000s" format="r1" tcFormat="NDF"
                        audioRole="dialogue">
              <marker start="45045/30000s" duration="1001/30000s"
                      value="CHECK: long_segment (d003)"/>
            </asset-clip>
          </spine>
        </sequence>
      </project>
      <project name="demo – rejects"> … same shape, dropped spans … </project>
    </event>
  </library>
</fcpxml>
```

**Rules**

- One `<format>` (all sources validated identical). One `<asset>` per source.
- `asset.start = t(start_frames)`; `asset.duration = t(duration_frames)`.
- `src` = `Path(path).resolve().as_uri()` (handles spaces and umlauts).
- For each range, in order:
  - `start = t(start_frames + in_frame)` — asset-clip start is in **source timecode space**, so the asset's start timecode is added.
  - `duration = t(out_frame − in_frame)`
  - `offset` = running sum of previous durations, `t(…)`.
- Markers inside an `asset-clip` use that asset-clip's source time: `marker.start = t(start_frames + frame_of(at_word))`, where `frame_of` = floor of the word's start time in frames, clamped into the asset-clip.
- Rejects project: one `asset-clip` per dropped span, in timeline order, each with a marker `value="<decision id>: <kind>"`.
- Write with `lxml`, pretty-printed, UTF-8.
- **Validate** with `lxml.etree.DTD(dtd_path)` before writing to `out/`. On failure, write to `out/<project>.invalid.fcpxml`, print the DTD errors, exit non-zero. If `dtd_path` is not set, write the file and print a prominent warning.

**Acceptance**

- Snapshot test: a fixed synthetic `timeline.json` + `sources.json` produce byte-identical output to a committed golden file.
- Property test: all time attributes parse to multiples of `frame_dur`; `offset[n+1] == offset[n] + duration[n]`.
- The sample project imports into FCP with zero warnings (manual check, documented in the PR description with a screenshot of the timeline).
- A source path containing spaces and `ä` imports correctly.

### 7.7 Eval (`evaluate.py`, `fcpxml_parse.py`)

**Input:** `test/gold/frontend-skills/` (raw sources are `test/fixtures/sample/raw/`; the edit is `manual.fcpxmld/Info.fcpxml`). **Output:** `test/gold/frontend-skills/eval.json` + console table.

1. Run the full pipeline on the gold raw sources (cached artifacts allowed).
2. Parse `manual.fcpxml`: resolve `<asset>` ids to file paths, then collect primary-storyline clips from the first `<project>`'s `<spine>`: `asset-clip`, and `clip` elements with a nested `video`/`audio` ref. For each: source path, source start, duration → source time span. Ignore connected clips, titles, generators, gaps. Unsupported elements in the spine (`sync-clip`, `mc-clip`, `ref-clip`) → warn and skip, listing them.
3. Label each word: **gold kept** if its midpoint lies inside a gold span of its source; **predicted kept** if inside a pipeline range.
4. Metrics:
   - precision, recall, F1 for "kept" (word level)
   - `false_cut_sentences`: sentences with ≥ 1 word gold-kept but predicted-dropped
   - `missed_retake_words`: words gold-dropped but predicted-kept
   - normalize counts per 10 minutes of gold output duration
5. `cutter eval` prints a table and the diff against the previous `eval.json`.

**Acceptance:** On a gold project where `manual.fcpxml` is the pipeline's own output re-imported and re-exported from FCP, F1 = 1.0.

---

## 8. CLI (`cli.py`)

```
cutter ingest     <project_dir> [--profile long] [--force]
cutter transcribe <project_dir> [--force]
cutter retakes    <project_dir> [--force]
cutter judge      <project_dir> [--force] [--no-llm]
cutter tighten    <project_dir> [--force]
cutter export     <project_dir> [--force]
cutter run        <project_dir> [--profile long] [--no-llm] [--force] [--set key=value …]
cutter eval       <gold_dir> [--set key=value …]
cutter words      <project_dir> --from 12.0 --to 30.0   # debug: print words with times
```

- `run` executes all stages in order, skipping stages whose `inputs_hash` and `config_hash` match the existing artifact.
- `inputs_hash`: sha256 over the content hashes of input artifacts, plus file size + mtime for raw media (do not hash multi-GB video files).
- `config_hash`: sha256 over only the config sections the stage reads.
- Each stage declares `STAGE_VERSION`; bumping it invalidates the cache.
- End of `run`: print a summary: source count, raw duration, output duration, decisions dropped/kept/flagged, path to the FCPXML.
- Exit codes: 0 success, 1 validation error, 2 external tool missing (ffmpeg, model), 3 DTD validation failed.

---

## 9. End-to-end test without real footage

`tests/e2e/make_fixture.py` builds a synthetic project using macOS `say` and ffmpeg:

1. Generate speech per segment with `say -v Samantha -o seg.aiff "<text>"`, including deliberate retakes, e.g. `"The agent joins the call and then"` followed by `"The agent joins the call and then subscribes to the audio track."`
2. Concatenate segments with 300 ms silences into two files (one retake crossing the file boundary).
3. Mux with a video test pattern: `ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30000/1001 -i speech.wav -shortest -c:v prores_ks -c:a pcm_s16le 01.mov`.
4. Store the expected kept/dropped text alongside.

`pytest tests/e2e` runs the full pipeline (`--no-llm`) on it and asserts the kept transcript equals the expected text (normalized), and that the FCPXML validates against the DTD if available.

---

## 10. Task breakdown

Tasks are sized for one agent session each. `→` marks dependencies.

| # | Task | Depends on | Done when |
| --- | --- | --- | --- |
| T0 | Scaffold: `uv` project, package layout, Typer stub, pytest, ruff | — | `uv run cutter --help` works, CI-style `uv run pytest` passes |
| T1 | `config.py`, `models.py`, `cache.py`, `profiles/long.yaml`, `--set` overrides | T0 | Round-trip tests for every model; config rejects unknown keys |
| T2 | Ingest | T1 | §7.1 acceptance |
| T3 | Transcribe (verify parakeet-mlx API first) | T1 | §7.2 acceptance; `cutter words` works |
| T4 | Retake detection | T1 | All §7.3 unit cases pass |
| T5 | LLM client + judge | T1, T4 | §7.4 acceptance with mocked client |
| T6 | Tighten + RMS helpers | T1, T4 | §7.5 property tests pass |
| T7 | FCPXML export (mirror reference.fcpxml) | T1 | §7.6 snapshot + property tests; DTD validation |
| T8 | `run` orchestration + caching + summary | T2–T7 | Second `run` with no changes skips all stages |
| T9 | Eval + FCPXML parser | T7, T8 | §7.7 acceptance |
| T10 | Synthetic e2e fixture + test | T8 | §9 passes |
| T11 | Real-footage pass: run on sample + gold, tune `long.yaml`, document results | T9, T10 | Phase 1 acceptance below met, or a written list of what blocks it |

T2, T3, T4 and T7 can run in parallel after T1. T7 works from a hand-written synthetic `timeline.json`.

---

## 11. Phase 1 acceptance

- [ ] A 10-minute raw recording imports into FCP with zero warnings.
- [ ] Eval on the gold project: ≥ 90% of retake words dropped (`missed_retake_words` low), ≤ 2 false-cut sentences per 10 min.
- [ ] No clicks or clipped word starts on 20 randomly chosen cuts.
- [ ] 30 minutes of raw footage → FCPXML in under 5 minutes on Apple Silicon (excluding first-time model download).
- [ ] Every flagged decision appears as a `CHECK` marker in FCP.
- [ ] Rejects project contains every dropped span.

---

## 12. Known risks (read before starting)

- **Source timecode.** Forgetting to add the asset's start timecode to asset-clip `start` makes every asset-clip point at the wrong frames. Covered by the timecode acceptance test in §7.1 and the rule in §7.6.
- **Variable frame rate.** Phone footage is often VFR. FCP conforms it; transcript times may drift against FCP's frames on long sources. Phase 1 only warns. If drift shows up in testing, add an optional ffmpeg constant-frame-rate proxy step in Phase 2, still keeping originals in the FCPXML.
- **ASR smoothing.** Parakeet may normalize false starts ("the the" → "the"), hiding very short retakes. Accept this in Phase 1; Phase 2's clap detection covers it.
- **Missing punctuation.** A finished sentence is detected only by a word ending in `.`, `?`, or `!`. If the transcript omits that mark, the sentence is treated as aborted and can be auto-dropped. Accept this in Phase 1.
- **Format name.** FCP is picky about `<format name>`. Always copy it from the reference export rather than generating it.