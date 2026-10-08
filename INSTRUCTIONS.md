# Cutter usage guide

Cutter currently supports these pipeline stages:

1. Ingest numbered video sources and extract analysis audio.
2. Transcribe the sources into word-level timestamps.
3. Record speech segments and claps (scaffold: writes an empty artifact).
4. Align the transcript to an optional script (scaffold: writes an empty artifact).
5. Inspect transcribed words in a time range.
6. Detect aborted failed takes and decide which ones should be dropped.
7. Ask a local model about ambiguous aborted takes.
8. Mark filler words (scaffold: writes an empty artifact).
9. Tighten the kept words into a rough cut.
10. Export the timeline as editable FCPXML.
11. Run the whole pipeline.
12. Score a rough cut against a gold edit.

## Requirements

- macOS on Apple Silicon
- Python 3.12, managed by `uv`
- `ffmpeg` and `ffprobe` on `PATH`
- Final Cut Pro for importing the finished FCPXML
- Internet access the first time the Parakeet transcription model is downloaded
- Optional: a local OpenAI-compatible server (LM Studio) for the judge. Without
  it, use `--no-llm` or `--set judge.enabled=false`

Install the command and its dependencies from the repository root:

```bash
brew install ffmpeg
uv sync
uv run cutter --help
```

Run Cutter through `uv run cutter ...` from the repository root. You can also
activate `.venv` and invoke `cutter` directly.

When `--profile` is omitted, Cutter measures the frame and chooses for you.
A horizontal or square picture uses `profiles/long.yaml`. A vertical picture
uses `profiles/short.yaml`, including a phone file stored sideways with a
90° rotation tag. The command prints the choice, for example
`profile short (vertical, 1080x1920)`. Pass `--profile long` or
`--profile short` to override it.

`short` is the same pipeline with a faster pace: pauses longer than 150 ms
are cut, and the handles around each word are shorter. Every key is
explained in `profiles/long.md`. The short-form numbers are explained in
`profiles/short.md`. Change a value in the YAML, or pass
`--set key.path=value` for one run. A stage reruns when the keys it reads
have changed. List values use YAML, for example `--set 'chapters.levels=[1]'`
or `--set 'fillers.words=["um", "uh"]'`.

Phase 2 adds the sections `vad`, `claps`, `fillers`, `script`, and `chapters`,
plus `ingest.cfr_proxy` (`never`, `auto`, or `always`), `ingest.proxy_codec`
(`prores_proxy` or `h264`), `tighten.use_vad`, `fcpxml.rejects_include_fillers`,
and `fcpxml.vfr_media` (`original` or `proxy`). What each key will do is in
`profiles/long.md`. The new stages are scaffolds, so these keys do not change
the cut yet. They do change the cache for the stages that read them. `short`
uses the same values as `long` for every new key.

## Create a project

One project produces one output video. Create a project folder with a `raw`
subfolder:

```text
projects/my-video/
└── raw/
    ├── 1-intro.MP4
    ├── 2-main.MP4
    └── 10-outro.MP4
```

Supported extensions are `.mov`, `.mp4`, and `.m4v`, case-insensitively.
Hidden files are ignored. Sources are naturally sorted, so `2-main.MP4` plays
before `10-outro.MP4`.

Phase 1 expects:

- one English speaker;
- one camera;
- sources with the same frame rate, width, height, and audio sample rate;
- non-drop-frame timecode.

An optional `script.md` in the project folder is the speaker's script.
`cutter align` reads that path (`script.path`, `script.md` by default). Without
the file, alignment writes an empty artifact and logs `no script at <path>`
once.

Source files are never modified or re-encoded. Cutter only reads them and
extracts mono WAV files for analysis.

## 1. Ingest sources

```bash
uv run cutter ingest projects/my-video
```

Choose a profile or ignore the cache:

```bash
uv run cutter ingest projects/my-video --profile long
uv run cutter ingest projects/my-video --force
```

Ingest:

- probes every source with `ffprobe`;
- rejects missing audio, mismatched source formats, and drop-frame timecode;
- prints a `VFR warning` when average and nominal frame rates differ;
- extracts mono 16 kHz audio for transcription;
- extracts mono 48 kHz audio for cut-point analysis;
- writes `projects/my-video/artifacts/sources.json`.

Generated files:

```text
projects/my-video/artifacts/
├── sources.json
└── audio/
    ├── s01.16k.wav
    ├── s01.48k.wav
    ├── s02.16k.wav
    └── s02.48k.wav
```

The JSON artifact records absolute source paths, duration, frame rate, source
timecode, and extracted WAV paths. If the source metadata and profile have not
changed and the WAV files are present, the command reuses the cached artifact.

## 2. Transcribe sources

Run ingest first, then:

```bash
uv run cutter transcribe projects/my-video
```

Force a fresh transcription:

```bash
uv run cutter transcribe projects/my-video --force
```

The first run may download the configured Parakeet model:

```text
mlx-community/parakeet-tdt-0.6b-v2
```

Transcription reads `artifacts/sources.json` and the 16 kHz WAV files. It
writes:

```text
projects/my-video/artifacts/words.json
```

Each word contains:

- a global word index;
- its source id;
- original text with punctuation;
- normalized text;
- start and end in seconds from the beginning of that source;
- a global sentence index;
- model confidence when available.

The stage validates timestamp order and source duration. Violations under
50 ms are clamped and logged; larger violations stop the command. An unchanged
sources artifact, WAV input, and transcription profile produces a cache hit.

## 3. Record speech and claps

Run ingest first, then:

```bash
uv run cutter audio projects/my-video
uv run cutter audio projects/my-video --profile long --force
```

This command is a scaffold. It reads `artifacts/sources.json` and the 16 kHz
and 48 kHz WAV files, and writes:

```text
projects/my-video/artifacts/audio_events.json
```

The file records backend `none`, no speech segments, and no claps. A second
run with the same inputs and the same `vad` and `claps` settings is a cache
hit. The summary is `speech segments: 0, claps: 0`.

## 4. Align an optional script

Run transcription first, then:

```bash
uv run cutter align projects/my-video
uv run cutter align projects/my-video --profile long --force
```

This command is a scaffold. It reads `artifacts/words.json` and, when the file
exists, `<project>/script.md` (or whatever `script.path` names). It writes:

```text
projects/my-video/artifacts/alignment.json
```

The file has `script: null` and empty chapter, sentence, unscripted, and
missing lists, even when `script.md` is present. The summary is `no script`.
When a later version stores a script, the summary counts sentences, chapters,
and missing sentences. A missing script is logged once at info:
`no script at <path>`. Adding or changing the script file misses the cache.
The cache also covers `script` and `chapters`.

## 5. Inspect word timestamps

After transcription, print words whose start time falls in a half-open time
range:

```bash
uv run cutter words projects/my-video --from 12.0 --to 30.0
```

Output is one word per line:

```text
Hello 12.120 12.430
world. 12.460 12.910
```

Times are source-relative. If a project has multiple sources, the same time
window is applied independently to every source. The output currently does not
include the source id, so use `artifacts/words.json` when that distinction is
needed.

This command is useful for comparing word timestamps against the source audio
or video.

## 6. Detect retakes

Run transcription first, then:

```bash
uv run cutter retakes projects/my-video
```

Force recalculation:

```bash
uv run cutter retakes projects/my-video --force
```

The command reads `artifacts/words.json` and writes:

```text
projects/my-video/artifacts/decisions.json
```

Retake detection is conservative:

- it compares transcript words, not claps or a script;
- the later take wins when the earlier take was aborted, and when the next
  sentence repeats a finished sentence in the same words;
- a sentence counts as finished when any word in it ends in `.`, `?`, or `!`;
- a finished sentence followed by different words is kept and flagged for
  human review;
- an aborted take that the transcript marks as finished is kept the same way
  when the following words are not that same sentence, including when that
  mark is added at the end of a source;
- a long failed-take candidate is kept and flagged;
- a drop that may omit content is flagged;
- candidates more than 90 seconds apart are not treated as retakes;
- a failed take at the end of one source can match a retake at the start of the
  next source.

The command summary reports total decisions, automatic drops, and flagged
decisions. Flagged decisions become `CHECK` markers during tightening.

## 7. Judge ambiguous takes

Run retakes first, or let `cutter judge` recompute them from the transcript:

```bash
uv run cutter judge projects/my-video
```

Skip the model and keep the retake decisions as they are:

```bash
uv run cutter judge projects/my-video --no-llm
```

The judge only sees aborted takes flagged `long_segment` or
`retake_missing_content`. A finished sentence (`complete_sentence`) stays kept
and flagged, and is never sent to the model.

The default profile calls a local OpenAI-compatible server, such as LM Studio:

```text
http://localhost:1234/v1
```

The model name is `qwen3-14b`. A verdict can drop the earlier take, keep both
takes, or flag that the first take should stay (`judge_prefers_first_take`).
Answers below the profile's `min_confidence` (0.7 in `long`) stay flagged.

If that endpoint is down, Cutter logs one warning,
`judge endpoint unreachable; leaving decisions unchanged`, and leaves
`artifacts/decisions.json` as it was.

The command rewrites `artifacts/decisions.json` as a judge-stage artifact.
`--no-llm` and `judge.enabled: false` still write that artifact, without
calling the model.

## 8. Mark fillers

Run retakes or judge first, so `decisions.json` exists, then:

```bash
uv run cutter fillers projects/my-video
uv run cutter fillers projects/my-video --profile long --force
```

This command is a scaffold. It reads `artifacts/words.json` and
`artifacts/decisions.json`, and writes:

```text
projects/my-video/artifacts/fillers.json
```

The file is a decisions artifact with no decisions. The summary is
`filler decisions: 0`. A second run with the same words, decisions, and
`fillers` settings is a cache hit. `cutter run` writes this file after the
judged decisions are on disk, and before tighten.

## 9. Tighten the rough cut

Run retakes or judge first, then:

```bash
uv run cutter tighten projects/my-video
```

Force a new timeline:

```bash
uv run cutter tighten projects/my-video --force
```

The command reads `artifacts/words.json`, `artifacts/decisions.json`,
`artifacts/sources.json`, and the 48 kHz analysis wavs. It writes:

```text
projects/my-video/artifacts/timeline.json
```

Kept words become ranges. A dropped retake, a source change, or a gap longer
than 400 ms starts a new range. Cut points are padded, then snapped to the
quietest moment in the analysis audio, then rounded to frames once.

The transcript often stretches the last word of a sentence across the pause
after it. Tighten measures where the voice actually stops: the first 300 ms
that stay within 12 dB of the source's background. A range then ends at most
`pad_tail_ms` after that point, and the hidden pause counts toward the gap
that starts a new range. A word is never cut before the midpoint of its
transcript timestamps. Tune this with `tighten.voice_margin_db` and
`tighten.voice_quiet_ms`; see `profiles/long.md`. Flagged
decisions become `CHECK` markers. Dropped retakes are listed for the rejects
sequence. A very short range is merged into a neighbour or removed with a
`CHECK: tiny fragment removed` marker.

## 10. Export FCPXML

Export currently requires all of these artifacts:

```text
projects/my-video/artifacts/
├── sources.json
├── timeline.json
└── words.json        # optional, but improves CHECK marker placement
```

Create the timeline with `cutter tighten` or `cutter run`, then:

```bash
uv run cutter export projects/my-video
```

Output:

```text
projects/my-video/out/my-video.fcpxml
```

The FCPXML contains:

- one event named `my-video – cutter`;
- a `my-video – rough cut` project containing kept ranges;
- a `my-video – rejects` project containing dropped retakes;
- `CHECK` markers from the timeline;
- references to the original source files.

All FCPXML times are exact rational frame times. Source paths with spaces and
non-ASCII characters are encoded as file URIs. If `words.json` is available,
its word start times position `CHECK` markers; otherwise markers are placed at
the beginning of their range.

By default, Cutter validates the output against Final Cut Pro's FCPXML 1.14
DTD:

```text
/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd
```

If the DTD is missing, Cutter still writes the FCPXML and prints a warning. If
DTD validation fails, Cutter writes
`out/my-video.invalid.fcpxml` and exits with code 3.

Import the resulting `.fcpxml` file into Final Cut Pro. The XML references the
original files; it does not contain or replace the media.

## 11. Run the pipeline

```bash
uv run cutter run projects/my-video
```

Choose a profile, skip the judge, override one setting, or ignore the cache:

```bash
uv run cutter run projects/my-video --profile long --no-llm
uv run cutter run projects/my-video --set tighten.max_gap_ms=200
uv run cutter run projects/my-video --force
```

`run` executes ingest, transcribe, audio, align, retakes, judge, fillers,
tighten, and export. A stage whose inputs and profile section are unchanged
is skipped. A judged decisions file already includes the retake result, so a
cache hit skips both retakes and judge. Fillers runs after that file is
settled, because it hashes `decisions.json`. `--no-llm` still runs the judge
stage, without calling the model. Audio, align, and fillers are scaffolds:
they write empty artifacts and do not change which words are kept.

The project folder after a run looks like this:

```text
projects/my-video/
├── script.md                  # optional
├── artifacts/
│   ├── sources.json
│   ├── words.json
│   ├── audio_events.json      # scaffold: no speech, no claps
│   ├── alignment.json         # scaffold: no script
│   ├── decisions.json
│   ├── fillers.json           # scaffold: no filler decisions
│   ├── timeline.json
│   ├── export.json
│   └── audio/
│       ├── s01.16k.wav
│       └── s01.48k.wav
└── out/
    └── my-video.fcpxml
```

The summary reports source count, raw duration, output duration, how many
decisions were dropped, kept, and flagged, and the FCPXML path.

## 12. Score a gold edit

```bash
uv run cutter eval test/gold/frontend-skills
```

Override one profile value for the run that produces the rough cut:

```bash
uv run cutter eval test/gold/frontend-skills --set tighten.max_gap_ms=200
```

The gold folder holds the human edit at
`manual.fcpxmld/Info.fcpxml` (or `manual.fcpxml`). Raw sources are `raw/`
inside that folder, or, if that folder is absent, `test/fixtures/sample/raw/`.
Those sample files are not in git. Place `1-intro.MP4` and `2-skills.MP4`
there before running eval. Pipeline artifacts are written in the folder that
holds `raw/` (`test/fixtures/sample/artifacts/` when using the sample path).
`eval.json` is written in the gold folder.

Eval picks `long` or `short` from the frame, the same way `run` does, and
calls the judge when `judge.enabled` is true.
It has no `--no-llm` flag. Skip the model with `--set judge.enabled=false`.
If the endpoint is down, the judge warning is logged and scoring continues
with the retake decisions. Eval then compares word midpoints with the gold
storyline.

It writes `eval.json` in the gold folder and prints precision, recall, F1,
false cuts, and missed retakes, including counts per 10 minutes of gold
duration. A second run also prints the change since the previous `eval.json`.

Unsupported storyline items (`sync-clip`, `mc-clip`, `ref-clip`) are skipped
with a warning. Gaps, titles, and generators are ignored.

## Caching

Stage artifacts include metadata containing:

- stage name and implementation version;
- a hash of input artifacts;
- a hash of only the profile sections read by that stage;
- creation time.

Ingest hashes raw media by file name, size, and modification time instead of
reading multi-gigabyte source contents. Later stages hash their input artifact
and analysis files. `audio` hashes `sources.json` plus the 16 kHz and 48 kHz
WAVs, and the `vad` and `claps` sections. `align` hashes `words.json` and the
script file when it exists, and the `script` and `chapters` sections.
`fillers` hashes `words.json` and `decisions.json`, and the `fillers` section.
`retakes` also hashes `claps` and `script`. `tighten` also hashes `vad`,
`claps`, and `chapters`. Export stores a stamp at `artifacts/export.json` so an
unchanged timeline is not written again. Use `--force` on ingest, transcribe,
audio, align, retakes, judge, fillers, tighten, or run when you need to bypass
a valid cache entry.

## Current command status

| Command | Status | Main output |
| --- | --- | --- |
| `cutter ingest` | Implemented | `artifacts/sources.json`, WAV files |
| `cutter transcribe` | Implemented | `artifacts/words.json` |
| `cutter audio` | Scaffolded (writes an empty artifact) | `artifacts/audio_events.json` |
| `cutter align` | Scaffolded (writes an empty artifact) | `artifacts/alignment.json` |
| `cutter words` | Implemented | Console output |
| `cutter retakes` | Implemented | `artifacts/decisions.json` |
| `cutter judge` | Implemented | `artifacts/decisions.json` |
| `cutter fillers` | Scaffolded (writes an empty artifact) | `artifacts/fillers.json` |
| `cutter tighten` | Implemented | `artifacts/timeline.json` |
| `cutter export` | Implemented | `out/<project>.fcpxml` |
| `cutter run` | Implemented | FCPXML plus the artifacts above |
| `cutter eval` | Implemented | `<gold>/eval.json` |

## Exit codes

- `0`: success
- `1`: invalid input, missing artifact, unsupported source, or other validation
  error
- `2`: required external tool or transcription package is unavailable
- `3`: FCPXML DTD validation failed

## Troubleshooting

### `ffmpeg is not on PATH` or `ffprobe is not on PATH`

```bash
brew install ffmpeg
```

Then confirm:

```bash
ffmpeg -version
ffprobe -version
```

### `missing sources artifact`

Run ingest before transcription or export:

```bash
uv run cutter ingest projects/my-video
```

### `missing words artifact`

Run transcription before word inspection or retake detection:

```bash
uv run cutter transcribe projects/my-video
```

### `missing timeline artifact`

Run tighten, or run the whole pipeline:

```bash
uv run cutter tighten projects/my-video
```

### `judge endpoint unreachable`

The judge could not reach `judge.base_url` (by default
`http://localhost:1234/v1`). Decisions are left unchanged. Start LM Studio
with the configured model, or skip the model:

```bash
uv run cutter run projects/my-video --no-llm
uv run cutter eval test/gold/frontend-skills --set judge.enabled=false
```

### `no raw sources for eval` or `missing gold edit`

`cutter eval` needs both the human edit and the camera files. The edit in
this repo is `test/gold/frontend-skills/manual.fcpxmld/Info.fcpxml`. The
camera files belong in `test/fixtures/sample/raw/` and are not checked in.

### Re-run after changing a source in place

Ingest normally notices changes in file size or modification time. To rebuild
every stage:

```bash
uv run cutter run projects/my-video --force
```
