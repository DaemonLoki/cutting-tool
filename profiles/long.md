# `long` profile

`profiles/long.yaml` is the long-form profile: one English speaker, one
camera. Cutter selects it for a horizontal or square frame. A vertical frame
selects `profiles/short.yaml` instead; see `profiles/short.md`. This file
explains every key. The numbers in the YAML are what Cutter reads. Comments
there are reminders; this file is the one to read while changing them.

Try one change without editing the file:

```bash
uv run cutter run projects/my-video --set retakes.min_match_words=2
```

Or edit `long.yaml` and run again. A stage reruns when the keys it reads have
changed. `--force` reruns every stage even when they have not.

Retakes and the judge both read the `retakes` section. After a retake change,
run `cutter retakes --force` and then `cutter judge`, or `cutter run`.

## If repetitions are missed

Detection compares transcript words. It does not hear a clap or read a script.
A repeat is a candidate only when an earlier run of words is similar to a
later run. The nearest earlier run wins. These are the knobs, in the order
worth trying:

1. `retakes.min_match_words` — lower it to accept a shorter shared opening.
   `2` catches two-word restarts. Very short matches that are only function
   words (`and the`, `so the`) are still ignored until the match is 5 words.
2. `retakes.word_similarity` — lower it when the two takes use different
   forms of the same word (`subscribe` / `subscribes`). `100` requires the
   normalized spellings to be identical.
3. `retakes.max_mismatches` — raise it when a word was inserted or swapped
   inside the repeated opening (`the agent joins` / `the agent then joins`).
4. `retakes.window_words` — raise it when the shared opening is longer than
   the current window. Comparison stops at the first difference that is not
   one of the allowed mismatches, so a longer window only helps when the
   words actually keep matching.
5. `retakes.max_lookback_s` — raise it when the second take starts more than
   90 seconds after the first take started.
6. `retakes.auto_drop_max_s` — raise it when a long aborted take is found but
   kept and flagged `long_segment`.
7. `retakes.missing_content_ratio` — raise it when a real retake is flagged
   `retake_missing_content` instead of dropped. Lower it to flag more often.

Two rules are not in this file. An exact repeat of one finished sentence is
dropped, and the later sentence stays. Any other earlier span with a word
ending in `.`, `?`, or `!` stays `keep` with reason `complete_sentence`. The
judge cannot turn that into a drop. Parakeet often adds that period at the
end of a source, so an aborted take that ends a file stays when the next
words are different. A match of fewer than 5 words must include one word
that is not a stopword.

## ingest

Used by `cutter ingest`.

`allowed_extensions`
: Source extensions in `raw/`, matched without caring about case. Hidden
  files are ignored either way. Add an extension here only when you also
  want those files probed and extracted.

`asr_sample_rate`
: Sample rate of the mono WAV sent to transcription, in Hz. `16000` matches
  Parakeet. Changing it does not resample an existing WAV until ingest runs
  again.

`analysis_sample_rate`
: Sample rate of the mono WAV used to snap cut points, in Hz. `48000` is the
  rate tighten reads. This is analysis audio only. Source video is never
  re-encoded.

`cfr_proxy`
: When a constant-frame-rate proxy is made. `never` makes none. `auto` makes
  one only for a source that printed a VFR warning. `always` makes one for
  every source. The proxy step is not implemented yet, so every value leaves
  the original files as the media Final Cut opens. The key is already part of
  the ingest cache.

`proxy_codec`
: Codec of that proxy. `prores_proxy` is a light ProRes. `h264` is a small
  H.264 file. Ignored while `cfr_proxy` is `never` and while the proxy step
  is absent.

## transcribe

Used by `cutter transcribe`.

`model`
: Hugging Face id of the Parakeet model. The first transcription downloads
  it. A different id loads a different model and invalidates `words.json`.

`chunk_duration_s`
: Length of each chunk Parakeet transcribes, in seconds. Longer chunks use
  more memory and can drift. Shorter chunks start and stop more often.

`overlap_duration_s`
: Seconds of audio repeated at each chunk boundary so a word that crosses
  the boundary is not cut in half. Must stay smaller than `chunk_duration_s`.

## retakes

Used by `cutter retakes`. The judge's cache also depends on this section,
because judging starts from these decisions.

The scan walks the transcript left to right. At each word it looks backward
for an earlier run that repeats. The dropped span is everything from that
earlier word up to, but not including, the later take. A failed take at the
end of one source can match a retake at the first words of the next source.
Lookback on that pair is measured inside the earlier source only.

`window_words`
: How many words, counting from the start of both runs, are eligible to
  match. The default `20` compares the first twenty words of the later take
  with twenty words of an earlier run. Raising it lets a longer shared
  opening count. Lowering it stops the comparison sooner, so a long repeat
  that differs
  early still needs `min_match_words` words before the difference.

`min_match_words`
: Minimum number of similar words required before the earlier run is a
  retake. The default `3` ignores one- and two-word echoes. Lower it to
  catch shorter restarts. Those short matches still need a content word
  unless at least 5 words matched.

`word_similarity`
: Rapidfuzz ratio, from 0 to 100, above which two normalized words count as
  the same word. Normalization is lowercase, punctuation stripped. `85`
  allows small spelling differences. `100` requires an exact normalized
  string. Lower values treat more pairs as repeats and also create more
  false repeats.

`max_mismatches`
: How many non-matching words may sit inside the window when the word after
  each of them matches. The default `5` allows up to five insertions or
  substitutions (`the agent joins` against `the agent then joins`). A second
  difference in a row ends the match. `0` requires an unbroken run. Raising
  it keeps matching through messier restarts.

`max_lookback_s`
: How far back, in seconds, the earlier take may start. Measured from the
  later word's start to the earlier word's start, inside one source. Across
  files it is measured from the earlier source's last word back to the
  candidate. The default `90` ignores the same sentence said again two
  minutes later. Raise it for long gaps between a failed take and the
  restart. A larger window also compares phrases that were meant to be said
  twice.

`auto_drop_max_s`
: Longest aborted span, in seconds, that is dropped automatically. The
  duration is from the first word of the earlier run to the word just before
  the later take. Longer spans stay, flagged `long_segment`, because a long
  repeat is often deliberate. The judge may still drop a `long_segment`
  decision. Raise this when real failed takes run past 20 seconds and you
  want them removed without a model.

`missing_content_ratio`
: After an aborted span qualifies, Cutter compares its content words with
  the later take. Content words are normalized words that are not stopwords.
  The later window is as long as the dropped span, plus 10 words. If the
  share of dropped content words absent from that window is greater than
  this ratio, the span is still dropped but flagged
  `retake_missing_content`. The default `0.5` flags a retake that left out
  more than half of the earlier content words. Raise it toward `1` to drop
  those takes without a flag. Lower it to flag more often. An earlier span
  with no content words skips this check.

`stopwords_file`
: `null` uses the built-in English function words: a, an, and, are, as, at,
  be, but, by, for, from, if, in, is, it, of, on, or, so, that, the, then,
  this, to, was, with. A path, absolute or relative to the directory you run
  Cutter from, replaces that list. One word per line. Stopwords do not count
  as content, and a match shorter than 5 words that is only stopwords is not
  a retake.

## judge

Used by `cutter judge` and by `cutter run` unless you pass `--no-llm`.
`cutter eval` has no `--no-llm`; set `enabled` to false, or pass
`--set judge.enabled=false`.

Only decisions flagged `long_segment` or `retake_missing_content` are sent.
An exact repeat of a finished sentence is already dropped and is not sent.
`complete_sentence` is never sent and stays kept.

`enabled`
: `true` calls the model. `false` writes the retake decisions unchanged and
  does not open a connection.

`base_url`
: OpenAI-compatible base URL. The default is LM Studio on this machine. The
  client sends `api_key` `lm-studio`, which that server ignores.

`model`
: Model name the server should use. This must be the name LM Studio has
  loaded. It is stored on each verdict.

`min_confidence`
: Verdicts below this value, from 0 to 1, stay flagged and keep the earlier
  action. The default `0.7` ignores a weak answer. Lower it to let more
  model answers change the cut. A confident `"B"` drops the earlier take. A
  confident `"both"` keeps both and clears the flag. A confident `"A"` keeps
  the earlier take and flags `judge_prefers_first_take`.

`timeout_s`
: Seconds to wait for one verdict. On timeout or connection failure the
  stage logs one warning and leaves the decisions file unchanged.

## vad

Used by `cutter audio` and, once speech segments affect cut points, by
`cutter tighten`. `cutter audio` writes one segment per run of voice and
records claps. Changing a key here still reruns `audio` and `tighten`.

`enabled`
: `false` leaves the speech list empty and records backend `none`. `true`
  asks the backend below.

`backend`
: `silero` is the neural voice detector and reads the 16 kHz WAV. `energy`
  reuses the loudness tighten already measures on the 48 kHz WAV, with
  `tighten.voice_margin_db` as the speech line.

`threshold`
: Silero speech probability, from 0 to 1. A chunk at or above this counts as
  voice. The default `0.5` is the library's usual line. Lower it when quiet
  speech is missed. Raise it when room noise becomes a segment. `energy`
  ignores this key.

`min_speech_ms`
: Speech shorter than this is dropped. The default `100` is longer than a
  clap, which is about 30 ms, so a clap does not become a speech segment.
  Lower it only when a real short word is being discarded.

`min_silence_ms`
: A silence shorter than this, inside speech, is bridged into the surrounding
  segment. The default `150` keeps a stop between syllables inside one
  segment. Raise it when a phrase is split at every breath.

`pad_ms`
: Milliseconds added to both ends of each speech segment after the other
  filters. The default `30` keeps the attack and release that the detector
  trims. The segment is still clamped to the source.

`flag_unheard_speech_s`
: A speech segment at least this many seconds long, with no transcript word
  inside it, gets a `CHECK: speech without transcript` marker. It is not
  added to the cut. The default `2` ignores a cough. Lower it to flag shorter
  gaps the transcript skipped.

## claps

Used by `cutter audio` for detection, and included in the cache of `retakes`
and `tighten` because those stages will use the onsets. Retakes and tighten
do not read the onsets yet.

`enabled`
: `false` leaves the clap list empty. `true` looks for transients in the
  pauses.

`min_rise_db`
: How far the 2 ms envelope must rise above its 1 second rolling median, in
  dB, before an onset counts. The default `20` asks for a sharp peak over the
  room. Lower it when a real clap is missed. Raise it when desk noise is
  marked.

`max_duration_ms`
: The level must fall back to near the background within this many
  milliseconds. The default `60` keeps a hand clap and rejects a thud or a
  door. Raise it when a soft clap is discarded for lasting too long.

`min_gap_ms`
: Transients closer than this are one clap. The onset kept is the first, and
  the peak kept is the louder one. The default `300` joins the two halves of
  one clap. Lower it when two deliberate claps land close together.

`min_match_words`
: How many transcript words must match after a clap before that clap anchors
  a retake. The default `2` is shorter than `retakes.min_match_words`, which
  stays the bar for a clap-free scan. Raise it when a clap is tying two
  unrelated phrases together.

`exclude_before_ms`
: Milliseconds before the clap onset that must not play. The default `100`
  keeps the attack of the clap out of the previous word.

`exclude_after_ms`
: Milliseconds after the clap onset that must not play. The default `250`
  covers the decay. A range that cannot get out of this zone stays, with a
  marker, rather than dropping speech to avoid the clap.

## fillers

Used by `cutter fillers`. That command writes `fillers.json`. Tighten does
not read it yet, so the rough cut still plays the marked words. Changing one
of these keys reruns the stage.

`enabled`
: `false` writes no filler decisions. `true` drops the words and phrases
  below, except a filler that is the only word of its sentence.

`words`
: Normalized words dropped wherever they occur. The default list is um, uh,
  and the usual hums. A filler that is the only word of its sentence stays;
  it is content. Matching ignores case and punctuation.

`phrases`
: Consecutive normalized words, matched exactly, such as `you know`. The
  default is empty, so no phrase is dropped. Add a phrase here when you want
  that pair removed and not the words alone.

`sentence_start_words`
: Normalized words dropped only when they are the first word of a sentence
  and another word in that sentence follows. The default is empty, so `so`
  and `okay` stay. A sentence-start word that is also in `words` is still
  dropped as a filler anywhere.

`max_duration_s`
: A filler run longer than this, measured from the first word's start to the
  last word's end, is kept and flagged `filler_long`. The default `1.5`
  treats a drawn-out um as something to hear. Lower it to drop those too.

## script

Used by `cutter align`, and included in the retakes cache because a script
will choose which take stays. `cutter align` reads the file and writes
`alignment.json`. A missing file is logged once and the empty artifact is
cached. Adding or editing the file misses that cache.

`path`
: Path of the script, relative to the project folder. The default
  `script.md` is optional. Point it at another name when the script is not
  called that.

`min_take_score`
: Rapidfuzz ratio, 0–100, of a transcript window against a script sentence.
  A window below this is not a take. The default `80` allows a paraphrase
  and rejects an aside. Lower it when sentences are marked missing that were
  spoken in different words. Raise it when an aside is matching a sentence.

`prefer`
: `best` keeps the highest score. `last` keeps the later take, which is the
  Phase 1 rule. Ties go to the later take either way.

`min_score_gap`
: When the chosen take beats an alternate by less than this, the drop is
  flagged `script_close_call`. The default `5` flags a near tie. Raise it to
  flag more often. `0` flags only an exact tie, and an exact tie still goes
  to the later take.

`drop_later_takes`
: `true` drops an alternate that comes after the chosen take. `false` keeps
  that later alternate and flags it. An alternate before the chosen take is
  dropped either way when the script is the evidence.

`flag_unscripted_s`
: An unscripted passage at least this many seconds long gets a
  `CHECK: unscripted` marker. The words stay in the cut. The default `20`
  ignores a short aside. Lower it to mark shorter departures from the script.

## chapters

Used by `cutter align` and by tighten's cache. `cutter align` records a
chapter for each heading whose level is listed below. Chapter markers in the
FCPXML are not written yet.

`enabled`
: `false` writes no chapter markers. `true` will place one at the first kept
  word of each heading whose level is listed below. Alignment still records
  those headings either way. A heading whose sentence was never spoken is
  skipped when markers are placed.

`levels`
: Heading levels that become markers. `1` is a line starting with `# `, and
  `2` is `## `. The default `[1, 2]` skips smaller headings. A heading of
  another level is not a chapter and is not spoken text.

## tighten

Used by `cutter tighten`. These keys move cut points. They do not decide
which words are repetitions. Words covered by a decision with `action: drop`
are already gone before this stage runs.

`max_gap_ms`
: A gap longer than this, from the end of one kept word to the start of the
  next, starts a new range. The default `400` leaves a short breath inside a
  range and cuts a longer pause. Lower it to remove more silence. Raise it
  to keep longer pauses in the rough cut.

`pad_head_ms`
: How many milliseconds before the first kept word the range starts, before
  silence snapping. The default `80` keeps the attack of the first consonant.
  The cut is also clamped so it does not include the previous word.

`pad_tail_ms`
: How many milliseconds after the end of the last kept word's voice the range
  ends. The default `120` keeps the end of the last word. Silence snapping
  may move the out-point earlier, never later, so a range does not end with
  more silence than this. The cut does not cross into the next word.

`snap_window_ms`
: Search distance, each side of the padded cut, for the quietest 10 ms
  frame. The default `150` can move a cut by up to 150 ms toward silence.
  The move still has to respect `min_head_ms`, `min_tail_ms`, and the
  neighbouring word. An out-point only moves earlier. `0` leaves the padded
  point where it is.

`min_head_ms`
: The in-point stays at least this far before the first word. The default
  `30` refuses a snap that would clip the start of the word. If the legal
  interval is empty, the padded point is kept.

`min_tail_ms`
: The out-point stays at least this far after the last word. The default
  `40` refuses a snap that would clip the end of the word.

`rms_frame_ms`
: Width of each loudness frame on the 48 kHz WAV, in milliseconds. The
  default `10` picks silence at about 10 ms resolution. Larger frames are
  smoother and less precise.

`voice_margin_db`
: Parakeet often stretches the last word of a sentence across the pause
  after it. In the sample, `with.` is stamped to 26.0 s while the voice stops
  near 25.7 s. Tighten finds where the voice stops in the analysis audio.
  The background is the quietest 10% of the source's levels, measured over
  50 ms. A level at least this many dB above the background is voice. The
  default `12` treats room tone a few dB above the floor as silence. Raise it
  when a range still ends in breath or room noise. Lower it when the end of
  a soft word is cut. A word is never cut before the midpoint of its
  transcript timestamps, so eval and the rough cut agree on which words are
  kept.

`voice_quiet_ms`
: How much background inside a word, after its voice, ends the word there.
  The default `300` is longer than the gap between two syllables. The end
  found this way sets the range out-point, and for a word ending in `.`, `?`,
  or `!`, it also counts toward `max_gap_ms`, so a pause hidden in a
  stretched timestamp can start a new range.

`min_range_frames`
: Ranges shorter than this many frames are merged into a neighbour when the
  gap is within `max_gap_ms`. Otherwise the range is removed and the next
  range gains a `CHECK: tiny fragment removed` marker. The default `6` drops
  scraps of a few frames. `1` keeps every range that has at least one frame.

`use_vad`
: `false` keeps the Phase 1 cut points even when `audio_events.json` has
  speech segments. `true` will use those segments for word onsets, voice
  ends, and gaps. Tighten does not read the segments yet, so both values
  produce the same cut. The key is already part of the tighten cache.

## fcpxml

Used by `cutter export` and by the export step of `cutter run`. These keys
name the Final Cut event and choose the DTD. They do not change which words
are kept.

`version`
: FCPXML version written on the document. `1.14` matches the reference
  export and the DTD path below.

`dtd_path`
: Final Cut Pro's FCPXML 1.14 DTD. When this file exists, export validates
  before writing `out/<project>.fcpxml`. A failure writes
  `out/<project>.invalid.fcpxml` and exits with code 3. When the path is
  missing, the FCPXML is still written and a warning is printed.

`event_name`
: Event name in the library. `{project}` is replaced with the project folder
  name. The default is `my-video – cutter` for a folder named `my-video`.

`project_name`
: Name of the project that contains the kept ranges. `{project}` is replaced
  the same way.

`rejects_project_name`
: Name of the project that contains every dropped span.

`audio_role`
: `audioRole` written on each asset clip. The default `dialogue` is the role
  Final Cut shows for this speech.

`rejects_include_fillers`
: `false` leaves filler drops out of the rejects project. `true` will include
  them, with a marker such as `f003: filler`. Export does not read filler
  decisions yet, so both values write the same rejects sequence.

`vfr_media`
: Which file the FCPXML points at when a source has a proxy. `original` keeps
  the camera file, which is the default. `proxy` will point at the proxy, and
  only for a source that has one. No proxy is made yet, so `proxy` still
  opens the original.
