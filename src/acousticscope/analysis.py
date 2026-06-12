"""Utilities for summarizing AcousticScope experiment CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UNDERSTOOD_LABELS = {"understood", "understand", "correct", "success", "1", "true", "yes"}
NOT_UNDERSTOOD_LABELS = {
    "not understood",
    "not_understood",
    "not understand",
    "misunderstood",
    "failed",
    "failure",
    "0",
    "false",
    "no",
}


@dataclass(frozen=True)
class FileSummary:
    path: Path
    total: int
    understood: int
    not_understood: int
    unknown: int

    @property
    def accuracy(self) -> float:
        labeled = self.understood + self.not_understood
        if labeled == 0:
            return 0.0
        return self.understood / labeled


def normalize_label(value: str | None) -> str:
    """Map human/LLM annotations to a small shared label set."""
    if value is None:
        return "unknown"

    cleaned = value.strip().lower().replace("-", " ").replace("_", " ")
    if not cleaned:
        return "unknown"
    if cleaned in UNDERSTOOD_LABELS:
        return "understood"
    if cleaned in NOT_UNDERSTOOD_LABELS:
        return "not_understood"
    if "not" in cleaned and "understand" in cleaned:
        return "not_understood"
    if "understand" in cleaned or "understood" in cleaned:
        return "understood"
    return "unknown"


def summarize_file(path: Path, label_column: str = "label") -> FileSummary:
    counts: Counter[str] = Counter()

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return FileSummary(path=path, total=0, understood=0, not_understood=0, unknown=0)

        fields = {field.strip().lower(): field for field in reader.fieldnames if field}
        resolved_label_column = fields.get(label_column.lower())

        for row in reader:
            if resolved_label_column:
                label = normalize_label(row.get(resolved_label_column))
            else:
                label = infer_label_from_row(row)
            counts[label] += 1

    return FileSummary(
        path=path,
        total=sum(counts.values()),
        understood=counts["understood"],
        not_understood=counts["not_understood"],
        unknown=counts["unknown"],
    )


def infer_label_from_row(row: dict[str, str]) -> str:
    """Fallback for raw logs that do not have a final annotation column yet."""
    alexa_answer = " ".join(
        row.get(column, "") for column in ("Alexa_Answer", "alexa_answer", "response")
    ).lower()
    if not alexa_answer:
        return "unknown"
    failure_markers = ("listening timeout", "skill problem", "didn't understand", "sorry")
    if any(marker in alexa_answer for marker in failure_markers):
        return "not_understood"
    return "unknown"


def iter_csv_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.csv"))
        elif path.suffix.lower() == ".csv":
            yield path


def print_markdown_table(summaries: list[FileSummary]) -> None:
    print("| file | total | understood | not understood | unknown | accuracy |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        print(
            f"| {summary.path.as_posix()} | {summary.total} | {summary.understood} | "
            f"{summary.not_understood} | {summary.unknown} | {summary.accuracy:.2%} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize AcousticScope result CSVs.")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or directories to analyze")
    parser.add_argument("--label-column", default="label", help="Annotation column name")
    args = parser.parse_args(argv)

    summaries = [summarize_file(path, args.label_column) for path in iter_csv_files(args.paths)]
    if not summaries:
        parser.error("No CSV files found.")

    print_markdown_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
