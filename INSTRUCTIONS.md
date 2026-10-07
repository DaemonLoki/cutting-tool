# Cutter usage guide

Cutter currently supports these pipeline stages:

1. Ingest numbered video sources and extract analysis audio.
2. Transcribe the sources into word-level timestamps.
3. Detect aborted failed takes and decide which ones should be dropped.
4. Inspect transcribed words in a time range.
5. Export an existing timeline artifact as editable FCPXML.

The `judge`, `tighten`, `run`, and `eval` commands are visible in `--help`, but
are not implemented yet. In particular, Cutter cannot yet create
`artifacts/timeline.json`, so `export` is only usable when that artifact has
been created separately.

## Requirements

- macOS on Apple Silicon
- Python 3.12, managed by `uv`
- `ffmpeg` and `ffprobe` on `PATH`
- Final Cut Pro for importing the finished FCPXML
- Internet access the first time the Parakeet transcription model is downloaded

Install the command and its dependencies from the repository root:

```bash
brew install ffmpeg
uv sync
uv run cutter --help
```

Run Cutter through `uv run cutter ...` from the repository root. You can also
activate `.venv` and invoke `cutter` directly.

The default profile is `profiles/long.yaml`. It is currently the only supplied
profile.

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

## 3. Inspect word timestamps

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

## 4. Detect retakes

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
- the later take wins only when the earlier take was aborted;
- a repeated, completed sentence is kept and flagged for human review;
- a long failed-take candidate is kept and flagged;
- a drop that may omit content is flagged;
- candidates more than 90 seconds apart are not treated as retakes;
- a failed take at the end of one source can match a retake at the start of the
  next source.

The command summary reports total decisions, automatic drops, and flagged
decisions. Flagged decisions are intended to become `CHECK` markers after the
tightening stage is implemented.

## 5. Export FCPXML

Export currently requires all of these artifacts:

```text
projects/my-video/artifacts/
├── sources.json
├── timeline.json
└── words.json        # optional, but improves CHECK marker placement
```

`timeline.json` cannot yet be generated by the CLI because `cutter tighten` is
not implemented. When a valid timeline artifact exists, run:

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

## Caching

Stage artifacts include metadata containing:

- stage name and implementation version;
- a hash of input artifacts;
- a hash of only the profile sections read by that stage;
- creation time.

Ingest hashes raw media by file name, size, and modification time instead of
reading multi-gigabyte source contents. Later stages hash their input artifact
and analysis files. Use `--force` on ingest, transcribe, or retakes when you
need to bypass a valid cache entry.

## Current command status

| Command | Status | Main output |
| --- | --- | --- |
| `cutter ingest` | Implemented | `artifacts/sources.json`, WAV files |
| `cutter transcribe` | Implemented | `artifacts/words.json` |
| `cutter words` | Implemented | Console output |
| `cutter retakes` | Implemented | `artifacts/decisions.json` |
| `cutter export` | Implemented, requires an existing timeline | `out/<project>.fcpxml` |
| `cutter judge` | Not implemented | — |
| `cutter tighten` | Not implemented | — |
| `cutter run` | Not implemented | — |
| `cutter eval` | Not implemented | — |

The unimplemented commands currently print a message and exit with code 1.

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

The tightening stage has not been implemented yet. Export needs a separately
created, model-valid `artifacts/timeline.json`.

### Re-run after changing a source in place

Ingest normally notices changes in file size or modification time. To
explicitly rebuild the audio and source artifact:

```bash
uv run cutter ingest projects/my-video --force
```

Then force dependent stages:

```bash
uv run cutter transcribe projects/my-video --force
uv run cutter retakes projects/my-video --force
```
