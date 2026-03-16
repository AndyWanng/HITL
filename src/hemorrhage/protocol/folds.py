"""Fold construction utilities."""

from __future__ import annotations

from typing import Iterable


def serpentine_fold_assignment(items: Iterable[tuple[str, int]], num_folds: int) -> dict[str, int]:
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    if num_folds <= 1:
        return {case_id: 1 for case_id, _ in ordered}
    pattern = list(range(1, num_folds + 1)) + list(range(num_folds, 0, -1))
    assignment: dict[str, int] = {}
    for index, (case_id, _) in enumerate(ordered):
        assignment[case_id] = pattern[index % len(pattern)]
    return assignment

