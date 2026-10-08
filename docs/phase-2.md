# cutter — Phase 2 Implementation Spec

**Goal:** Give the rough cut more evidence than the transcript alone: where the voice actually is (VAD), where the speaker clapped to mark a mistake, which filler words to remove, and, when a script exists, which take matches it best and where each chapter starts. Phase 2 keeps the Phase 1 shape: one English speaker, one camera, originals referenced from FCPXML, a bias against false cuts.

**Audience:** Coding agents. Read `README.md` (the Phase 1 spec) first; this document only describes what changes. Ground rules in README §0 still apply. Where a fact must be verified against a real tool, the spec says so. Do not guess there.

Decisions already verified while writing this spec (8 Oct 2026):

- `parakeet-tdt-0.6b-v2` **does** emit fillers with timestamps (`Um,` 0.16–0.64, `uh,` 1.60–2.16 on a `say` clip). A filler's `end` swallows the pause after it, like a sentence-final word.
- `FCPXMLv1_14.dtd`: `chapter-marker` is a `%marker_item;` of `asset-clip` with `start` (required), `value` (required), `duration`, `note`, `posterOffset`. `media-rep kind` is `original-media | proxy-media`.
- `pysilero-vad` 3.4.0 installs on this machine (ONNX runtime, no torch). API: `SileroVoiceActivityDetector().process_chunk(bytes) -> float` on 512-sample 16 kHz int16 chunks; `process_samples`, `process_chunks`, `reset`.
- The sample footage (`1-intro.MP4`, `2-skills.MP4`) is constant frame rate (24000/1001, HEVC). There is no VFR fixture in the repo.

---

## 0. Ground rules added in Phase 2

1. **Audio never cuts on its own.** VAD and claps refine *where* a cut lands and *lower the bar* for a transcript match. A span is dropped only when the transcript (or the script) says it is a failed take, or it is a configured filler word. (Extends ADR 0002.)
2. **The script wins over "last take wins".** With a script, the take closest to the script stays, even when it is the earlier one. Without a script, Phase 1 rules apply unchanged. (New ADR 0006.)
3. **Every new stage is optional.** No `script.md`, VAD disabled, no claps: the pipeline produces exactly the Phase 1 result. Phase 1 tests keep passing without modification, except where this spec says a model gained a field.
4. **One artifact per stage** still holds. Two new stage artifacts: `audio_events.json` and `alignment.json`, plus `fillers.json`.
5. **Each PR updates `INSTRUCTIONS.md`** for anything user-visible, `GLOSSARY.md` for new terms, and adds an ADR when it decides something this spec left open.

---

## 1. Scope

### In scope

- **VAD:** speech segments per source; used by tighten for word onsets, voice ends, gaps, and for flagging speech the ASR missed.
- **Clap detection:** short broadband transients in pauses; a clap anchors a retake candidate with a lower match threshold, is flagged when no retake follows, and never plays in the cut.
- **Filler removal:** configured filler words and phrases become `filler` decisions.
- **Script alignment:** `script.md` in the project folder; every script sentence gets its candidate takes; the best take is chosen; alternate takes become decisions; missing and unscripted passages are reported.
- **Chapter markers:** script headings become FCP chapter markers at the first kept word of the chapter.
- **CFR proxy for VFR sources:** a measurement first; an opt-in proxy step if drift is real.
- Pipeline, caching, CLI, eval, synthetic e2e fixture, and docs for all of the above.
- Stage 0: finish the Phase 1 acceptance pass (#13) on the real footage.

### Out of scope (unchanged from README §1 unless listed above)

Keywords per script paragraph, topic segmentation without a script, punch-ins, audio volume ramps, Remotion, languages other than English, interviews, multicam, b-roll.

---

## 2. Prerequisites

| Item | Who | Purpose |
| --- | --- | --- |
| A recording with 3–5 real hand claps marking mistakes (any length ≥ 2 min) in `projects/claps/raw/` | Stefan | Tune `claps.*` on real transients. Until then, synthetic claps only. |
| The script used for the sample video, as `test/gold/frontend-skills/script.md` | Stefan, if it exists | Real alignment and chapter check. Until then, T11 derives a script from the gold edit's kept words for a smoke test (circular; not a metric). |
| LM Studio as in Phase 1 | optional | judge |

---

## 3. Tech stack changes

- New runtime dependency: `pysilero-vad>=3.4` (brings `onnxruntime`). Nothing else.
- Still forbidden: librosa, torch, moviepy, OpenTimelineIO, video-processing libraries.

---

## 4. Repository layout changes

```
docs/phase-2.md         # this file
cutter/
  vad.py                # speech segments (silero + energy backend)
  claps.py              # transient detection
  events.py             # `cutter audio` stage: vad + claps -> audio_events.json
  script.py             # script.md parsing: chapters, sentences, words
  align.py              # `cutter align` stage: transcript <-> script -> alignment.json
  fillers.py            # `cutter fillers` stage -> fillers.json
projects/<name>/
  script.md             # optional input
  artifacts/
    audio_events.json
    alignment.json
    fillers.json
    proxy/s01.mov       # only when a CFR proxy is made
```

---

## 5. Configuration

New sections and keys. Every number is `[cfg]`. `short.yaml` gets the same keys; values may differ only where `profiles/short.md` explains why. Every key gets a comment in the YAML and a paragraph in `profiles/long.md`, as in Phase 1.

```yaml
ingest:
  # existing keys …
  cfr_proxy: "never"          # never | auto | always. auto = only sources with vfr_warning. T6 may change the default.
  proxy_codec: "prores_proxy" # prores_proxy | h264

vad:
  enabled: true
  backend: "silero"           # silero | energy. energy reuses audio.voice_levels.
  threshold: 0.5              # silero speech probability
  min_speech_ms: 100          # shorter speech blips are dropped (a clap is ~30 ms)
  min_silence_ms: 150         # shorter silences inside speech are bridged
  pad_ms: 30                  # added to both ends of each segment
  flag_unheard_speech_s: 2.0  # speech this long with no transcript word inside gets a CHECK marker

claps:
  enabled: true
  min_rise_db: 20             # peak above the 1 s rolling median of the 2 ms envelope
  max_duration_ms: 60         # level must fall back within this
  min_gap_ms: 300             # transients closer than this merge into one clap
  min_match_words: 2          # transcript match needed after a clap (retakes.min_match_words elsewhere)
  exclude_before_ms: 100      # no range may play [t - before, t + after]
  exclude_after_ms: 250

fillers:
  enabled: true
  words: ["um", "uh", "uhm", "umm", "erm", "hmm", "mhm", "mm", "ah", "eh", "er"]
  phrases: []                 # e.g. ["you know", "i mean"]; matched on norm, dropped when listed
  sentence_start_words: []    # e.g. ["so", "okay", "right", "well"]; dropped only as the first word of a sentence
  max_duration_s: 1.5         # a longer filler is kept and flagged filler_long

script:
  path: "script.md"           # relative to the project folder; absent file = stage is a no-op
  min_take_score: 80          # rapidfuzz ratio of a transcript window against a script sentence
  prefer: "best"              # best | last. best = highest score, ties go to the later take
  min_score_gap: 5            # smaller gap between chosen and alternate take = drop with a CHECK marker
  drop_later_takes: true      # an alternate take after the chosen one is dropped too (false = kept, flagged)
  flag_unscripted_s: 20       # an unscripted passage this long gets a CHECK marker

chapters:
  enabled: true
  levels: [1, 2]              # heading levels that become chapter markers

tighten:
  # existing keys …
  use_vad: true               # false = Phase 1 behaviour even when audio_events.json has speech

fcpxml:
  # existing keys …
  rejects_include_fillers: false
  vfr_media: "original"       # original | proxy. proxy is only valid when the source has one.
```

`STAGE_CONFIG_SECTIONS` gains: `audio: (vad, claps)`, `align: (script, chapters)`, `retakes: (retakes, claps, script)`, `fillers: (fillers,)`, `tighten: (tighten, vad, claps, chapters)`. `ingest` keeps `(ingest,)`; `export` keeps `(fcpxml,)`.

---

## 6. Data model changes (`models.py`)

Bump `STAGE_VERSION` of every stage whose artifact shape changes.

### 6.1 `sources.json`

`Source` gains `proxy: str | None = None` (path relative to the project folder).

### 6.2 `artifacts/audio_events.json` (new, stage `audio`)

```json
{
  "backend": "silero",
  "speech": [{"source": "s01", "start": 1.14, "end": 5.62}],
  "claps":  [{"source": "s01", "t": 7.312, "peak_db": -4.1, "rise_db": 31.0}]
}
```

- `speech` is sorted by source then start, non-overlapping within a source. Times are float seconds, like `words.json`.
- `claps` is sorted the same way. `t` is the onset.

### 6.3 `artifacts/alignment.json` (new, stage `align`)

```json
{
  "script": {"path": "script.md", "hash": "sha256:…"},
  "chapters":  [{"id": "c01", "title": "Intro", "level": 1, "first_sentence": 0}],
  "sentences": [
    {"id": 0, "chapter": "c01", "text": "Do you want your designs to go from this to this?",
     "takes": [{"first_word": 0, "last_word": 10, "score": 100.0, "chosen": false},
               {"first_word": 11, "last_word": 21, "score": 100.0, "chosen": true}]}
  ],
  "unscripted": [{"first_word": 40, "last_word": 55}],
  "missing": [7]
}
```

- `script: null` and empty lists when the project has no script. The stage still writes the artifact so caching is uniform.
- `takes` are sorted by `first_word`; exactly one is `chosen` per sentence that has any take. `missing` lists sentence ids with no take.
- `unscripted` spans are maximal runs of words that belong to no take of any sentence.

### 6.4 `decisions.json` and `fillers.json`

`Decision` changes:

```python
kind: Literal["retake", "filler"]
evidence: Literal["transcript", "clap", "script"] = "transcript"
clap_s: float | None = None        # evidence == "clap"
score_gap: float | None = None     # evidence == "script": chosen score − alternate score
```

- `fillers.json` is a `DecisionsArtifact` whose decisions all have `kind: "filler"`, ids `f001…`, `kept_from_word = last + 1`, `match_words = 0`.
- New `flag_reason` values: `clap_without_retake`, `script_close_call`, `filler_long`. Existing ones are unchanged.
- `kept_from_word` keeps its meaning "first word of the take that stays". For a script decision that drops a *later* take, that is the chosen take's first word, which is **before** `dropped_words`. Tighten already handles a marker whose word precedes the range: it attaches to the range containing the word or the nearest following one.

### 6.5 `timeline.json`

```json
{
  "ranges": [...],
  "dropped": [...],
  "chapters": [{"id": "c01", "title": "Intro", "at_word": 11}]
}
```

- `chapters` defaults to `[]`. `at_word` is a kept word.
- `dropped` now includes filler spans; the `decision` id tells them apart (`f…`).
- New marker texts: `CHECK: clap_without_retake (d007)`, `CHECK: script_close_call (d012)`, `CHECK: filler_long (f003)`, `CHECK: unscripted`, `CHECK: speech without transcript`.

---

## 7. Stages

Pipeline order in `run`: ingest → transcribe → **audio** → **align** → retakes → judge → **fillers** → tighten → export.

Fillers hashes `decisions.json`, and judge rewrites that file, so fillers runs after the judged artifact is on disk. The judge cache hashes every config section retakes reads (`retakes`, `claps`, `script`), not only `retakes`, so a clap or script threshold change reruns both stages.

### 7.1 Ingest: CFR proxy (`ingest.py`) — gated by T6

Do not implement before T6 has measured drift (see §10). If T6 says drift is real:

1. `cfr_proxy: auto` makes a proxy for every source with `vfr_warning`; `always` for all; `never` for none.
2. Command (verify flag names against the installed ffmpeg):
   ```bash
   ffmpeg -nostdin -y -i in.mp4 -fps_mode cfr -r 30000/1001 -c:v prores_ks -profile:v 0 -c:a copy artifacts/proxy/s01.mov
   ```
   `h264`: `-c:v libx264 -preset veryfast -crf 18`. Keep `-c:a copy`: the audio, and therefore every time in every artifact, is identical between original and proxy.
3. WAV extraction keeps reading the original.
4. `Source.proxy` is set. Proxies are cached by the original's size and mtime, like the WAVs.
5. Export: `fcpxml.vfr_media: proxy` makes the `<asset>`'s `media-rep kind="original-media" src` point at the proxy for sources that have one. Originals remain the default (ADR 0003). Record the opt-in as ADR 0008.

### 7.2 Audio events (`vad.py`, `claps.py`, `events.py`) — `cutter audio`

**Input:** `sources.json`, 16 kHz and 48 kHz WAVs. **Output:** `audio_events.json`.

**VAD (`vad.py`)**

1. `speech_segments(wav16k: Path, config: VadConfig) -> list[tuple[float, float]]`.
2. `silero` backend: read int16 mono 16 kHz; feed 512-sample chunks to `SileroVoiceActivityDetector.process_chunk`; `reset()` per file. A chunk is speech when probability ≥ `threshold`. **Verify first** that the installed `pysilero-vad` returns one float per 512-sample chunk and what it does with a short final chunk (pad with zeros).
3. `energy` backend: `audio.voice_levels` on the 48 kHz envelope with `rms_frame_ms` = 10; a frame is speech when ≥ background + `tighten.voice_margin_db`.
4. Post-process both: bridge silences < `min_silence_ms`, drop segments < `min_speech_ms`, pad by `pad_ms`, clamp to `[0, duration_s]`, merge overlaps.
5. If `vad.enabled: false`, `speech` is empty and `backend` is `"none"`.

**Claps (`claps.py`)**

1. `detect_claps(wav48k: Path, speech: list[tuple[float, float]], config: ClapsConfig) -> list[Clap]`.
2. Envelope: `audio.rms_envelope(wav, frame_ms=2)`, in dB. Background: rolling median over 1000 ms (500 frames), computed with a running window, not a per-frame sort.
3. Onset at frame `k` when `level[k] − background[k] ≥ min_rise_db` and `level[k] − level[k−3] ≥ min_rise_db / 2` (a rise within 6 ms). The event ends at the first frame where `level ≤ background + 6 dB`. Reject when the event lasts longer than `max_duration_ms`. `peak_db` = max level inside, `rise_db` = peak − background.
4. Merge events closer than `min_gap_ms` into one (keep the first onset, the max peak).
5. Drop events whose onset lies inside a VAD speech segment (a plosive is not a clap). When `speech` is empty (VAD off), keep them all.
6. If `claps.enabled: false`, `claps` is empty.

**Acceptance**

- VAD: synthetic 16 kHz WAV of tone bursts and silence; segments match the bursts within 50 ms. A `say` clip of one sentence yields one segment. Both backends.
- Claps: synthetic WAV with three 30 ms white-noise bursts at −3 dBFS over −50 dBFS noise, one pair 150 ms apart → two claps. The same bursts inside a `say` sentence (mixed) → zero when VAD is on. A `say` clip with no claps → zero.
- Stage caching: a second `cutter audio` is a hit; changing `vad.threshold` is a miss; changing `tighten.pad_head_ms` is still a hit.

### 7.3 Script alignment (`script.py`, `align.py`) — `cutter align`

**Input:** `words.json`, `<project>/script.md`. **Output:** `alignment.json`.

**Parsing (`script.py`)**

- Markdown. Ignore YAML front matter, fenced code blocks, HTML comments (`<!-- director notes -->`), and blank lines. Headings `#`…`######` become chapters when their level is in `chapters.levels`; a heading of another level is ignored as text. Text before the first heading belongs to no chapter (`chapter: null`).
- Paragraphs are split into sentences at `.`, `?`, `!` followed by whitespace or end of text. Inline markdown (`*`, `_`, `` ` ``, links) is stripped. Words are normalized with `transcribe.normalize_word`. Empty sentences are dropped.
- `ScriptInfo.hash` is the sha256 of the file; it is part of the stage's `inputs_hash`.

**Alignment (`align.py`)**

1. Transcript norms per source, in `i` order. Script sentences never match across a source boundary.
2. For each sentence `s` of length `n`, candidate windows start at every transcript index `p` where `words[p].norm` fuzzy-matches `s[0]` or `s[1]` (rapidfuzz ratio ≥ `retakes.word_similarity`). For each `p`, the window is `words[p : p + n + slack]` with `slack = max(1, n // 4)`; shrink from the right to the length that maximizes `rapidfuzz.fuzz.ratio(" ".join(window norms), " ".join(s))`. Keep windows with score ≥ `min_take_score`. Overlapping windows of the same sentence keep the higher score, ties the later one.
3. Choose one take per sentence so that chosen takes are strictly increasing in `first_word` across sentences and non-overlapping, maximizing the sum of scores. Dynamic programming over sentences × candidates (both are small). `prefer: last` replaces the score by `first_word` in the objective. Ties go to the later take. A sentence with no feasible candidate is `missing`.
4. `unscripted`: maximal runs of words not inside any take of any sentence.
5. No script: write `script: null`, everything empty, log once.

**Acceptance (unit, synthetic words)**

| Case | Expected |
| --- | --- |
| Script of 3 sentences, transcript has them once | 3 chosen takes, `missing = []`, `unscripted = []` |
| Sentence 2 spoken twice, second worse (one word wrong) | first take chosen, second is an alternate |
| Sentence 2 spoken twice, identical | later take chosen |
| Sentence 2 spoken twice, better one first, `prefer: last` | later take chosen |
| Transcript has a 15-word aside between sentences 1 and 2 | one `unscripted` span covering it |
| Sentence 4 never spoken | `missing = [3]` |
| Sentence spoken across the source boundary | `missing`, no take |
| Property: chosen takes are increasing and non-overlapping | hypothesis |

### 7.4 Retakes with claps and script (`retakes.py`)

**Input:** `words.json`, `audio_events.json`, `alignment.json`. **Output:** `decisions.json`. Both new inputs are optional: a missing file means Phase 1 behaviour.

Run in this order; each step skips words already in `claimed`.

1. **Script decisions** (only with chosen takes). For every sentence with a chosen take `C` and alternate takes `A…`:
   - Alternate before `C`: `dropped_words = [A.first, A.last]`, `kept_from_word = C.first`, `evidence: script`, `score_gap = C.score − A.score`. `action: drop`; `flag: true, flag_reason: script_close_call` when `score_gap < min_score_gap`. The Phase 1 guards do **not** apply here; the script is the evidence. This is where ADR 0004 is superseded (ADR 0006).
   - Alternate after `C` and `drop_later_takes: true`: same, with the alternate's words dropped. `false`: `action: keep`, `flag: true`, reason `script_close_call`.
   - Words between an alternate's `last` and the next take's `first` that are not in any take (the broken-off tail of a failed attempt) are included in the dropped span when they are shorter than `retakes.window_words`; otherwise left alone.
2. **Transcript scan**, as Phase 1 (`detect_retakes`), with `claimed` pre-seeded from step 1.
3. **Clap-anchored candidates.** For each clap at `t` in source `s`: `i` = first word of `s` with `start ≥ t`. Skip if `i` is already `kept_from_word` of a decision or in `claimed`. Run `_first_match` with `min_match_words = claps.min_match_words` and the normal lookback and guards; on a match record the decision with `evidence: clap`, `clap_s = t`. On no match: `dropped_words = [k, i − 1]` where `k` is the first word after the previous sentence end (or previous clap, or source start) in `s`; `action: keep`, `flag: true`, reason `clap_without_retake`, `evidence: clap`. If `k > i − 1` (the clap sits right after a finished sentence), record nothing and log it.
4. Cross-file retakes, as Phase 1.

**Acceptance**

- All Phase 1 cases in README §7.3 still pass with no events and no alignment.
- Clap, then an ASR-smoothed restart (`the agent — [clap] — the agent joins`): with `min_match_words = 3` the plain scan finds nothing; with the clap, `the agent` is dropped, `evidence: clap`.
- Clap with no repeat after it → `keep`, flagged `clap_without_retake`, span from the previous sentence end.
- Script: finished sentence, then the same sentence with one different word, script matches the first → the **second** is dropped, `evidence: script`, unflagged when the gap ≥ 5.
- Script: the two takes score within 5 → dropped, flagged `script_close_call`.
- Script: no alternates → identical to the transcript scan.
- Property: no two decisions overlap in `dropped_words`.

### 7.5 Fillers (`fillers.py`) — `cutter fillers`

**Input:** `words.json`, `decisions.json`. **Output:** `fillers.json`.

1. Kept words = words not dropped by `decisions.json`.
2. A word is a filler when its `norm` is in `fillers.words`; a run of consecutive kept filler words is one decision. `phrases` match consecutive norms exactly. `sentence_start_words` match only when the word is the first of its sentence (`sent` changes) **and** is followed by a word in the same sentence.
3. A filler run longer than `max_duration_s` (by `end − start` of the words) → `action: keep`, `flag: true`, reason `filler_long`.
4. A filler that is the only word of its sentence (`"Hmm."`) is kept and not flagged; it is content.
5. Decisions ids `f001…`, `kind: filler`, `evidence: transcript`.

**Acceptance**

- `Um, so the agent, uh, joins the call.` → two filler decisions (`Um,`, `uh,`); `so` stays unless `sentence_start_words` lists it.
- `you know` is dropped only when `phrases` contains it.
- A filler inside a dropped retake produces no decision.
- A 2 s `um` is kept and flagged.
- e2e: a `say` fixture with 5 fillers → all 5 dropped, every other word kept.

### 7.6 Tighten with VAD, claps, fillers, chapters (`tighten.py`, `audio.py`)

**Input:** `words.json`, `decisions.json`, `fillers.json` (optional), `sources.json`, `audio_events.json` (optional), `alignment.json` (optional), 48 kHz WAVs. **Output:** `timeline.json`.

1. Dropped ids = union of `decisions.json` and `fillers.json` drops.
2. **Spoken end with VAD** (`tighten.use_vad` and `speech` non-empty): for every kept word (not only sentence-final ones), `spoken_end = clamp(segment_end, midpoint, word.end)` where `segment_end` is the end of the VAD segment containing the word's midpoint. A word whose midpoint is in no segment uses the Phase 1 rule (`voice_end` for sentence-final words, `word.end` otherwise). Gaps and `prev_end` use `spoken_end`.
3. **Spoken start with VAD:** the in-point of a range is computed from `min(first.start, segment_start)` when `segment_start` is within 200 ms before `first.start`, still never before the previous word's `spoken_end`.
4. **Clap exclusion:** for each clap, `[t − exclude_before, t + exclude_after]` is forbidden. A range containing it is split there if it has kept words on both sides; otherwise its in- or out-point is moved outside the zone. Snap search excludes forbidden frames. A range that cannot avoid a zone is kept, with a `CHECK: clap inside range` marker (should not happen; log it).
5. **Unheard speech:** a VAD segment ≥ `vad.flag_unheard_speech_s` with no word midpoint inside → marker `CHECK: speech without transcript` on the next range (or the previous one at the end of a source). It is not added to the cut.
6. **Unscripted:** an `unscripted` span whose words last ≥ `script.flag_unscripted_s` → marker `CHECK: unscripted` at its first kept word.
7. **Chapters:** for each alignment chapter with `chapters.enabled`, `at_word` = first kept word of the chosen take of its first sentence; if that sentence is missing, the first sentence in that chapter that has a chosen take; if none, skip with a warning.
8. Everything else as Phase 1, including cleanup and frame rounding last.

**Acceptance**

- Phase 1 property test still holds with events; add: no range intersects a clap zone; every chapter's `at_word` is a kept word; every filler's midpoint is outside all ranges.
- With `use_vad: false` and no events the output is byte-identical to Phase 1 on the synthetic fixture.
- Sample footage: cut count and total duration change by less than 10% when VAD is enabled; listen test by Stefan.

### 7.7 Export (`fcpxml.py`)

1. Chapter markers: `<chapter-marker start="…" value="<title>"/>` inside the `asset-clip` containing `at_word`, placed like `CHECK` markers (word start frame in source timecode space). Omit `duration` and `posterOffset`. Marker items keep the DTD order: markers of all kinds are `%marker_item;*`, so interleaving is fine; write `chapter-marker` after `marker` elements for a stable snapshot.
2. Rejects project: fillers are included only when `fcpxml.rejects_include_fillers: true`. Marker value `f003: filler`.
3. `vfr_media: proxy` as §7.1.5, only if T6 lands.
4. Snapshot test gains a second golden file with chapters and fillers. DTD validation stays mandatory.

### 7.8 Run, summary, eval

- `run` adds the new stages in the order above. `--no-llm` unchanged. `cutter audio`, `cutter align`, `cutter fillers` are standalone commands with `--force`.
- Summary adds: speech share of raw duration, clap count, filler drops, script sentences found/missing, unscripted spans, chapter count.
- `cutter eval` unchanged in metrics; it prints `fillers_dropped` and `chapters` as information. Eval reads `script.md` from the gold folder when present.
- `cutter words` gains `--source s01`.

---

## 8. Synthetic e2e fixture (`tests/e2e/make_fixture.py`)

Extend `build_project` with keyword flags, default `False`, so Phase 1's test is unchanged:

- `fillers=True`: FILE_ONE's second line becomes `"Um, the cache stores the result and, uh, skips the work."`; `expected.txt` has no fillers.
- `clap=True`: a 30 ms white-noise burst (`anoisesrc`, −3 dBFS, 25 ms fade-out) between the aborted line and its retake instead of plain silence, and one burst inside a 1 s pause with no retake after it. Onset times are written to `claps.json` as `{"01.mov": [onset, onset], "02.mov": []}`, seconds from the start of each file. The second burst starts 200 ms into a 1 s silence appended after the tail cut.
- `script=True`: write `script.md` with `# Cache` and `# Agent` headings over the expected sentences, and one extra script sentence that is never spoken.

`tests/e2e/test_phase2.py` runs `cutter run --no-llm` on the full fixture and asserts: kept text equals expected; no range contains a clap; the FCPXML has two `chapter-marker`s and one `CHECK: clap_without_retake`; `missing` has one sentence; DTD valid when available.

---

## 9. Task breakdown

Sized for one agent session each. `→` is a dependency. Branch `phase-2/t<N>-<slug>`, PR title `P2-T<N>: <title>`, body `Closes #<issue>`.

| # | Task | Files | Depends on | Done when |
| --- | --- | --- | --- | --- |
| T0 | Phase 1 hardening pass (#13): run sample + gold, record §11 results, list manual checks for Stefan | none or tuning only | — | #13 has results; automatable items ticked |
| T1 | Foundations: models §6, config §5, `STAGE_CONFIG_SECTIONS`, `pysilero-vad` dep, stage skeletons for `audio`/`align`/`fillers` writing empty artifacts, run order, CLI commands, fixture flags §8, profile docs, INSTRUCTIONS skeleton, GLOSSARY terms | models, config, cache, cli, run, events/align/fillers stubs, profiles, fixture | — | Phase 1 tests pass; `cutter run` on the Phase 1 fixture is unchanged; new stages cached |
| T2 | VAD | `vad.py`, `events.py`, tests | T1 | §7.2 VAD acceptance |
| T3 | Clap detection | `claps.py`, `events.py`, tests | T1 | §7.2 claps acceptance |
| T4 | Fillers stage | `fillers.py`, tests | T1 | §7.5 acceptance |
| T5 | Script parsing + alignment | `script.py`, `align.py`, tests | T1 | §7.3 acceptance |
| T6 | VFR drift measurement (research), synthetic VFR fixture, then CFR proxy if warranted | `ingest.py`, `tests/e2e`, `docs/adr/0008` | T1 | §10 question answered in the issue; proxy implemented or explicitly not |
| T7 | Retakes: script decisions + clap anchors, ADR 0006 | `retakes.py`, tests | T3, T5 | §7.4 acceptance |
| T8 | Tighten: VAD, clap zones, fillers, chapters, unheard/unscripted markers | `tighten.py`, `audio.py`, tests | T2, T3, T4, T5 | §7.6 acceptance |
| T9 | Export: chapter markers, filler rejects, proxy media | `fcpxml.py`, tests, golden file | T1 (T6 for proxy) | §7.7 acceptance |
| T10 | Run summary, eval info, `words --source`, Phase 2 e2e test, docs pass | `run.py`, `cli.py`, `evaluate.py`, e2e, INSTRUCTIONS/README/GLOSSARY | T7, T8, T9 | §8 test passes; docs match behaviour |
| T11 | Real-footage pass with claps recording and script; tune; acceptance §11 | profiles, docs | T10 + Stefan's inputs | §11 met, or a written list of what blocks it |

**Parallel groups:** T0 ∥ T1. After T1: T2 ∥ T3 ∥ T4 ∥ T5 ∥ T6 (disjoint files; `events.py` is touched by T2 and T3 — T2 owns the file, T3 adds one call and rebases). After those: T7 ∥ T8 ∥ T9 (disjoint files). Then T10, then T11.

**Quality gates per PR:** `uv run ruff check .` and `uv run pytest -q` green; property tests for numeric code; `INSTRUCTIONS.md` updated when behaviour is user-visible; `GLOSSARY.md` for new terms; ADR for any decision this spec left open; reviewer runs the e2e tests and a Bugbot pass before merge.

---

## 10. T6 research question: does VFR drift exist in FCP?

1. Build a synthetic VFR source: two `testsrc2` segments at 30 and 24 fps with a `say` sentence every 10 s, concatenated with `-fps_mode passthrough` into `.mp4`, 3 minutes long. Verify `ffprobe` reports `r_frame_rate ≠ avg_frame_rate` and cutter prints the VFR warning.
2. Run the pipeline. Note the source time of the last sentence's first word.
3. Stefan imports the FCPXML. In FCP, does the asset-clip for the last range start on that word? Measure the offset in frames.
4. Drift ≥ 2 frames at 3 minutes → implement §7.1 with `cfr_proxy: auto` as default, ADR 0008. Otherwise: implement nothing beyond the fixture, keep `cfr_proxy: never`, record the finding in ADR 0008 and close.

---

## 11. Phase 2 acceptance

- [ ] Phase 1 acceptance (README §11) still holds on the sample footage.
- [ ] Eval on the gold project with its script: F1 ≥ Phase 1's value; `missed_retake_words` ≤ Phase 1's.
- [ ] Synthetic fixture (§8): all five fillers dropped; no other word lost; the clap never plays; the clap-anchored restart is dropped; one `clap_without_retake` marker.
- [ ] Claps recording (§2): every real clap detected; false claps per 10 minutes ≤ 1; each clap sits inside a dropped span or splits a pause.
- [ ] Chapter markers appear in FCP's chapter list at the first word of each heading's first sentence.
- [ ] With `use_vad: true`, 20 random cuts on the sample have no clipped word onsets (listen test).
- [ ] 30 minutes of raw footage → FCPXML in under 6 minutes on Apple Silicon.
- [ ] Zero FCP import warnings; DTD valid.
- [ ] T6 question answered and recorded.

---

## 12. Known risks

- **Clap false positives** from plosives, desk knocks, or door slams. VAD gating and `max_duration_ms` are the defences; a false clap only adds a marker or lowers a match threshold, never cuts alone.
- **Script drift.** Speakers paraphrase. `min_take_score: 80` is a starting point; too low and asides match script sentences, too high and every sentence is `missing`. Sweep with `cutter eval --set script.min_take_score=…` in T11.
- **Filler cuts that click.** A filler removal is a hard cut inside a phrase. Snap-to-silence and `min_range_frames` apply; if clicks show up in the listen test, raise `fillers.max_duration_s` down to keep more, or add crossfades in a later phase (audio ramps are out of scope).
- **Silero and claps.** A clap may register as a short speech blip; `min_speech_ms` filters it. Verify on the claps recording.
- **Chapter placement after cleanup.** `at_word` can be removed by `_remove_tiny`; resolve chapters after cleanup and fall back to the next kept word.
