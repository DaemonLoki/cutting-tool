# `long` profile

`profiles/long.yaml` is the only profile. It is the set of thresholds for one
English speaker, one camera, recorded long-form. This file explains every
key. The numbers in the YAML are what Cutter reads. Comments there are
reminders; this file is the one to read while changing them.

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

Two rules are not in this file. A finished sentence stays, even when it is
said again: any word in the earlier span ending in `.`, `?`, or `!` forces
`keep` with reason `complete_sentence`. The judge cannot turn that into a
drop. Parakeet often adds that period at the end of a source, so an aborted
take that ends a file can look finished and stay in the cut. A match of fewer
than 5 words must include one word that is not a stopword.

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
  match. The default `6` compares the first six words of the later take with
  six words of an earlier run. Raising it lets a longer shared opening count.
  Lowering it stops the comparison sooner, so a long repeat that differs
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
  each of them matches. The default `1` allows a single insertion or
  substitution (`the agent joins` against `the agent then joins`). A second
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
  `retake_missing_content`. The default `0.3` flags a retake that left out
  at least 30% of the earlier content words. Raise it toward `1` to drop
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
: How many milliseconds after the last kept word the range ends, before
  silence snapping. The default `120` keeps the end of the last word. The
  cut does not cross into the next word.

`snap_window_ms`
: Search distance, each side of the padded cut, for the quietest 10 ms
  frame. The default `150` can move a cut by up to 150 ms toward silence.
  The move still has to respect `min_head_ms`, `min_tail_ms`, and the
  neighbouring word. `0` leaves the padded point where it is.

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

`min_range_frames`
: Ranges shorter than this many frames are merged into a neighbour when the
  gap is within `max_gap_ms`. Otherwise the range is removed and the next
  range gains a `CHECK: tiny fragment removed` marker. The default `6` drops
  scraps of a few frames. `1` keeps every range that has at least one frame.

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
