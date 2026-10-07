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
`min_range_frames` stays 6, so a scrap of a few frames is still removed
instead of becoming a click.

If a cut clips a word, raise `pad_head_ms` or `min_head_ms` before raising
`max_gap_ms`. If the pace is still slow, lower `max_gap_ms` toward 100.
Below that, the gap between two words in one phrase starts to split.
