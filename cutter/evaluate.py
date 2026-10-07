"""Score a rough cut against a gold edit.

A false cut is speech the gold edit keeps that the rough cut removed. A
missed retake is a failed take the rough cut still plays. A word counts as
kept when its midpoint falls in a range of the same source.
"""

from __future__ import annotations

from pathlib import Path

from cutter.fcpxml_parse import SourceSpan, parse_fcpxml
from cutter.models import SourcesData, StrictModel, TimelineData, Word

_FIELDS = (
    "precision",
    "recall",
    "f1",
    "false_cut_sentences",
    "missed_retake_words",
    "false_cut_sentences_per_10min",
    "missed_retake_words_per_10min",
    "gold_duration_s",
    "kept_words",
    "gold_kept_words",
)

_COUNT_FIELDS = frozenset(
    {
        "false_cut_sentences",
        "missed_retake_words",
        "kept_words",
        "gold_kept_words",
    }
)


class Metrics(StrictModel):
    """Word-level agreement between a rough cut and a gold edit."""

    precision: float
    recall: float
    f1: float
    false_cut_sentences: int
    missed_retake_words: int
    false_cut_sentences_per_10min: float
    missed_retake_words_per_10min: float
    gold_duration_s: float
    kept_words: int
    gold_kept_words: int


def score(
    words: list[Word],
    source_names: dict[str, str],
    gold: list[SourceSpan],
    predicted: list[SourceSpan],
) -> Metrics:
    """Compare predicted ranges with a gold edit at word level.

    ``source_names`` maps a source id to its filename. A word is gold-kept
    when its midpoint lies in a gold span of that filename, half-open
    ``[in_s, out_s)``. The same test marks a word predicted-kept. A source id
    missing from ``source_names`` is dropped on both sides. A false cut is a
    sentence with at least one gold-kept word the rough cut drops. A missed
    retake is a gold-dropped word the rough cut keeps. Rates are those counts
    per 10 minutes of gold duration.
    """
    gold_by_name = _by_source(gold)
    predicted_by_name = _by_source(predicted)
    predicted_kept = 0
    gold_kept = 0
    both_kept = 0
    false_cut_sentences: set[int] = set()
    missed_retake_words = 0

    for word in words:
        in_gold = _kept(word, source_names, gold_by_name)
        in_predicted = _kept(word, source_names, predicted_by_name)
        if in_gold:
            gold_kept += 1
        if in_predicted:
            predicted_kept += 1
        if in_gold and in_predicted:
            both_kept += 1
        elif in_gold:
            false_cut_sentences.add(word.sent)
        elif in_predicted:
            missed_retake_words += 1

    if predicted_kept == 0:
        precision = 1.0 if gold_kept == 0 else 0.0
    else:
        precision = both_kept / predicted_kept
    recall = 1.0 if gold_kept == 0 else both_kept / gold_kept
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    gold_duration_s = float(sum((span.out_s - span.in_s for span in gold), 0.0))
    false_cuts = len(false_cut_sentences)
    if gold_duration_s == 0.0:
        false_cut_rate = 0.0
        missed_rate = 0.0
    else:
        per_ten_minutes = gold_duration_s / 600.0
        false_cut_rate = false_cuts / per_ten_minutes
        missed_rate = missed_retake_words / per_ten_minutes

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_cut_sentences=false_cuts,
        missed_retake_words=missed_retake_words,
        false_cut_sentences_per_10min=false_cut_rate,
        missed_retake_words_per_10min=missed_rate,
        gold_duration_s=gold_duration_s,
        kept_words=predicted_kept,
        gold_kept_words=gold_kept,
    )


def predicted_spans(timeline: TimelineData, sources: SourcesData) -> list[SourceSpan]:
    """One span per range. source_name is Path(source.path).name. in_s/out_s from the range."""
    by_id = {source.id: source for source in sources.sources}
    spans: list[SourceSpan] = []
    for item in timeline.ranges:
        source = by_id.get(item.source)
        if source is None:
            raise ValueError(f"range {item.id} source {item.source!r} is not in sources")
        spans.append(
            SourceSpan(
                source_name=Path(source.path).name,
                in_s=item.in_s,
                out_s=item.out_s,
            )
        )
    return spans


def diff_metrics(previous: Metrics | None, current: Metrics) -> str:
    """Text table of current metrics and, when previous is not None, the delta.

    Delta is current minus previous for each numeric field. The table is plain
    aligned text so the gold comparison stays stable.
    """
    header = ["metric", "current"]
    if previous is not None:
        header.append("delta")
    rows = [header]
    for field in _FIELDS:
        value = getattr(current, field)
        row = [field, _format_number(field, value)]
        if previous is not None:
            row.append(_format_delta(field, value, getattr(previous, field)))
        rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines: list[str] = []
    for row in rows:
        cells = [row[0].ljust(widths[0])]
        cells.extend(row[index].rjust(widths[index]) for index in range(1, len(row)))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def load_metrics(path: Path) -> Metrics | None:
    """None if the file is missing."""
    if not path.is_file():
        return None
    return Metrics.model_validate_json(path.read_text(encoding="utf-8"))


def write_metrics(path: Path, metrics: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")


def evaluate_gold(
    *,
    words: list[Word],
    sources: SourcesData,
    timeline: TimelineData,
    manual_fcpxml: Path,
    eval_json: Path,
) -> tuple[Metrics, str]:
    """Parse the gold FCPXML, score the rough cut, and write eval.json.

    The table is the diff against the previous eval.json when that file
    already holds a gold score. Returns the metrics and the table text.
    """
    previous = load_metrics(eval_json)
    gold = parse_fcpxml(manual_fcpxml).spans
    names = {source.id: Path(source.path).name for source in sources.sources}
    metrics = score(words, names, gold, predicted_spans(timeline, sources))
    table = diff_metrics(previous, metrics)
    write_metrics(eval_json, metrics)
    return metrics, table


def _by_source(spans: list[SourceSpan]) -> dict[str, list[SourceSpan]]:
    grouped: dict[str, list[SourceSpan]] = {}
    for span in spans:
        grouped.setdefault(span.source_name, []).append(span)
    return grouped


def _kept(
    word: Word,
    source_names: dict[str, str],
    spans_by_name: dict[str, list[SourceSpan]],
) -> bool:
    filename = source_names.get(word.source)
    if filename is None:
        return False
    midpoint = (word.start + word.end) / 2
    return any(span.in_s <= midpoint < span.out_s for span in spans_by_name.get(filename, ()))


def _format_number(field: str, value: float) -> str:
    if field in _COUNT_FIELDS:
        return str(int(value))
    return f"{float(value):.6f}"


def _format_delta(field: str, current: float, previous: float) -> str:
    if field in _COUNT_FIELDS:
        return f"{int(current) - int(previous):+d}"
    return f"{float(current) - float(previous):+.6f}"
