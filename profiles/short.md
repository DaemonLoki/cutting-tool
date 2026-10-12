# `short` profile

`profiles/short.yaml` is the short-form profile: one English speaker, one
camera, cut for a faster pace. Every key means the same thing as in
`profiles/long.md`. This file only explains the values that differ.

`cutter run` and the other stage commands select this profile when the
displayed frame is taller than it is wide. A 90° or 270° rotation tag is
applied first, so a portrait phone file stored as landscape pixels still
counts as vertical. Pass `--profile long` to keep the long cut on a
vertical frame:

```bash
uv run cutter run projects/my-video --profile long
```

Retake detection, the judge, transcription, and the FCPXML names match
`long.yaml`. A repetition that `long` would drop is dropped here too. The
difference is how tightly the kept words are joined.

## Tighter cuts

`long` leaves a pause of up to 400 ms inside a range, and keeps 80 ms before
a word and 120 ms after it. `short` cuts sooner and holds less silence around
each word.

| Key | `long` | `short` | Why |
| --- | --- | --- | --- |
| `tighten.max_gap_ms` | 400 | 150 | A pause longer than 150 ms starts a new range, so a breath is removed. Gaps between words in one phrase are usually shorter than that and stay. |
| `tighten.pad_head_ms` | 80 | 40 | Less air before the first word. Still above `min_head_ms`, so the consonant is not the cut. |
| `tighten.pad_tail_ms` | 120 | 50 | Less air after the last word. |
| `tighten.snap_window_ms` | 150 | 60 | The quietest moment is chosen closer to the word. A wider search would walk back into the pause this profile just cut. |
| `tighten.min_head_ms` | 30 | 20 | The in-point may sit 20 ms before the first word, not 30. |
| `tighten.min_tail_ms` | 40 | 25 | The out-point may sit 25 ms after the last word, not 40. |

`rms_frame_ms` stays 10, so the snap is still measured in 10 ms steps.
`voice_margin_db` stays 12 and `voice_quiet_ms` stays 300, so a range ends
where the voice stops in both profiles.

If a cut clips a word, raise `pad_head_ms` or `min_head_ms` before raising
`max_gap_ms`. If the pace is still slow, lower `max_gap_ms` toward 100.
Below that, the gap between two words in one phrase starts to split.

## Two keys tuned against real reel edits

`test/gold/top-3-frontend-skills` and `test/gold/ChatGPT-Intelligent-UI` are
gold-standard reel edits (vertical phone footage, hand-cut in Final Cut).
Running `cutter eval` against both turned up two values where `long.yaml`'s
numbers do not fit short-form footage, independent of pacing:

| Key | `long` | `short` | Why |
| --- | --- | --- | --- |
| `claps.min_rise_db` | 20 | 36 | Neither reel used a marking clap, yet the 20 dB default measured 31 and 85 "claps" on two totally ordinary takes — all plosive consonants in normal speech, not claps. Those false claps occasionally sat inside a kept range tighten could not route around (`CHECK: clap inside range`, a path the spec calls "should not happen"), clipping into a word the gold edit kept. 36 dB cuts the false-positive rate to a handful and the clipping stopped on both reels, at no cost to word-level F1. |
| `tighten.min_range_frames` | 6 | 15 | short's tighter `max_gap_ms` and pads isolate a single short word into its own range more often than long's wider ones do. On `top-3-frontend-skills`, exactly that happened to a word the gold edit kept, and at 6 frames it was short enough to be dropped as a scrap with `CHECK: tiny fragment removed`. Sweeping `min_range_frames` from 6 to 40 on both reels: F1 keeps improving through 15 on both, then starts trading a new false cut for a smaller `missed_retake_words` count past 15 on `ChatGPT-Intelligent-UI`. 15 is the largest value that still only merges fragments rather than cutting into content. |

`max_gap_ms`, the pads, and `snap_window_ms` were swept the same way
(`max_gap_ms` 100–300, pads and `min_head_ms`/`min_tail_ms` each at a few
values) and none of them moved either reel's score — this eval scores word
midpoints, which is too coarse to see sub-100ms pacing changes. Their current
values are unchanged; judging those still needs a listen test, not `cutter
eval`.

Both reels still show a large `missed_retake_words` count (66 and 286) that
neither of these keys touches. In both gold edits the speaker re-takes an
opening line two or three times with each attempt worded slightly
differently, never as an exact repeat. `retakes.py`'s `complete_sentence`
guard only auto-drops an exact repeat of a finished sentence; a reworded
retake is kept and flagged for the editor instead, by design (ADR 0002, ADR
0005) — no config key changes that. This is a retake-detection behavior
shared with `long.yaml`, not a `short`-specific pacing gap, so it is reported
here rather than patched.
